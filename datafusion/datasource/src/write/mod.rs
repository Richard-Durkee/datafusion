// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Module containing helper methods/traits related to enabling
//! write support for the various file formats

use crate::file_compression_type::FileCompressionType;
use crate::file_sink_config::FileSinkConfig;
use crate::write::demux::{DemuxedStreamReceiver, FileMetadata, FileSize};
use arrow::array::RecordBatch;
use arrow::datatypes::Schema;
use bytes::Bytes;
use datafusion_common::DataFusionError;
use datafusion_common::error::Result;
use datafusion_common_runtime::{JoinSet, SpawnedTask};
use object_store::ObjectStore;
use object_store::buffered::BufWriter;
use object_store::path::Path;
use pin_project_lite::pin_project;
use std::future::Future;
use std::io::{IoSlice, Write};
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::task::{Context, Poll, ready};
use tokio::io::{AsyncWrite, AsyncWriteExt};
use tokio::sync::mpsc::Receiver;
use tokio::task::JoinError;

pub mod demux;
pub mod orchestration;

/// A buffer with interior mutability shared by the SerializedFileWriter and
/// ObjectStore writer
#[derive(Clone)]
pub struct SharedBuffer {
    /// The inner buffer for reading and writing
    ///
    /// The lock is used to obtain internal mutability, so no worry about the
    /// lock contention.
    pub buffer: Arc<futures::lock::Mutex<Vec<u8>>>,
}

impl SharedBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: Arc::new(futures::lock::Mutex::new(Vec::with_capacity(capacity))),
        }
    }
}

impl Write for SharedBuffer {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let mut buffer = self.buffer.try_lock().unwrap();
        Write::write(&mut *buffer, buf)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        let mut buffer = self.buffer.try_lock().unwrap();
        Write::flush(&mut *buffer)
    }
}

/// A trait that defines the methods required for a RecordBatch serializer.
pub trait BatchSerializer: Sync + Send {
    /// Asynchronously serializes a `RecordBatch` and returns the serialized bytes.
    /// Parameter `initial` signals whether the given batch is the first batch.
    /// This distinction is important for certain serializers (like CSV).
    fn serialize(&self, batch: RecordBatch, initial: bool) -> Result<Bytes>;
}

/// Converts table schema to writer schema, which may differ in the case
/// of hive style partitioning where some columns are removed from the
/// underlying files.
pub fn get_writer_schema(config: &FileSinkConfig) -> Arc<Schema> {
    if !config.table_partition_cols.is_empty() && !config.keep_partition_by_columns {
        let schema = config.output_schema();
        let partition_names: Vec<_> =
            config.table_partition_cols.iter().map(|(s, _)| s).collect();
        Arc::new(Schema::new_with_metadata(
            schema
                .fields()
                .iter()
                .filter(|f| !partition_names.contains(&f.name()))
                .map(|f| (**f).clone())
                .collect::<Vec<_>>(),
            schema.metadata().clone(),
        ))
    } else {
        Arc::clone(config.output_schema())
    }
}

/// A builder for an [`AsyncWrite`] that writes to an object store location.
///
/// This can be used to specify file compression on the writer. The writer
/// will have a default buffer size unless altered. The specific default size
/// is chosen by [`BufWriter::new`].
///
/// We drop the `AbortableWrite` struct and the writer will not try to cleanup on failure.
/// Users can configure automatic cleanup with their cloud provider.
#[derive(Debug)]
pub struct ObjectWriterBuilder {
    /// Compression type for object writer.
    file_compression_type: FileCompressionType,
    /// Output path
    location: Path,
    /// The related store that handles the given path
    object_store: Arc<dyn ObjectStore>,
    /// The size of the buffer for the object writer.
    buffer_size: Option<usize>,
    /// The compression level for the object writer.
    compression_level: Option<u32>,
    /// Counts the number of bytes actually written.
    bytes_written_counter: Option<Arc<AtomicU64>>,
}

impl ObjectWriterBuilder {
    /// Create a new [`ObjectWriterBuilder`] for the specified path and compression type.
    pub fn new(
        file_compression_type: FileCompressionType,
        location: &Path,
        object_store: Arc<dyn ObjectStore>,
    ) -> Self {
        Self {
            file_compression_type,
            location: location.clone(),
            object_store,
            buffer_size: None,
            compression_level: None,
            bytes_written_counter: None,
        }
    }

    /// Set buffer size in bytes for object writer.
    ///
    /// # Example
    /// ```
    /// # use datafusion_datasource::file_compression_type::FileCompressionType;
    /// # use datafusion_datasource::write::ObjectWriterBuilder;
    /// # use object_store::memory::InMemory;
    /// # use object_store::path::Path;
    /// # use std::sync::Arc;
    /// # let compression_type = FileCompressionType::UNCOMPRESSED;
    /// # let location = Path::from("/foo/bar");
    /// # let object_store = Arc::new(InMemory::new());
    /// let mut builder = ObjectWriterBuilder::new(compression_type, &location, object_store);
    /// builder.set_buffer_size(Some(20 * 1024 * 1024)); //20 MiB
    /// assert_eq!(
    ///     builder.get_buffer_size(),
    ///     Some(20 * 1024 * 1024),
    ///     "Internal error: Builder buffer size doesn't match"
    /// );
    /// ```
    pub fn set_buffer_size(&mut self, buffer_size: Option<usize>) {
        self.buffer_size = buffer_size;
    }

    /// Set buffer size in bytes for object writer, returning the builder.
    ///
    /// # Example
    /// ```
    /// # use datafusion_datasource::file_compression_type::FileCompressionType;
    /// # use datafusion_datasource::write::ObjectWriterBuilder;
    /// # use object_store::memory::InMemory;
    /// # use object_store::path::Path;
    /// # use std::sync::Arc;
    /// # let compression_type = FileCompressionType::UNCOMPRESSED;
    /// # let location = Path::from("/foo/bar");
    /// # let object_store = Arc::new(InMemory::new());
    /// let builder = ObjectWriterBuilder::new(compression_type, &location, object_store)
    ///     .with_buffer_size(Some(20 * 1024 * 1024)); //20 MiB
    /// assert_eq!(
    ///     builder.get_buffer_size(),
    ///     Some(20 * 1024 * 1024),
    ///     "Internal error: Builder buffer size doesn't match"
    /// );
    /// ```
    pub fn with_buffer_size(mut self, buffer_size: Option<usize>) -> Self {
        self.buffer_size = buffer_size;
        self
    }

    /// Currently specified buffer size in bytes.
    pub fn get_buffer_size(&self) -> Option<usize> {
        self.buffer_size
    }

    /// Set compression level for object writer.
    pub fn set_compression_level(&mut self, compression_level: Option<u32>) {
        self.compression_level = compression_level;
    }

    /// Set compression level for object writer, returning the builder.
    pub fn with_compression_level(mut self, compression_level: Option<u32>) -> Self {
        self.compression_level = compression_level;
        self
    }

    /// Currently specified compression level.
    pub fn get_compression_level(&self) -> Option<u32> {
        self.compression_level
    }

    pub fn with_bytes_written_counter(mut self, counter: FileSize) -> Self {
        self.bytes_written_counter = Some(counter);
        self
    }

    /// Return a writer object that writes to the object store location.
    ///
    /// If a buffer size has not been set, the default buffer buffer size will
    /// be used.
    ///
    /// # Errors
    /// If there is an error applying the compression type.
    pub fn build(self) -> Result<Box<dyn AsyncWrite + Send + Unpin>> {
        let Self {
            file_compression_type,
            location,
            object_store,
            buffer_size,
            compression_level,
            bytes_written_counter,
        } = self;

        let buf_writer = match buffer_size {
            Some(size) => BufWriter::with_capacity(object_store, location, size),
            None => BufWriter::new(object_store, location),
        };
        if let Some(counter) = bytes_written_counter {
            file_compression_type.convert_async_writer_with_level(
                CountingWriter {
                    inner: buf_writer,
                    counter,
                },
                compression_level,
            )
        } else {
            file_compression_type
                .convert_async_writer_with_level(buf_writer, compression_level)
        }
    }
}

pin_project! {
    /// [`AsyncWrite`] wrapper that counts the number of effectively written bytes
    struct CountingWriter<W> {
        #[pin]
        inner: W,
        counter: Arc<AtomicU64>,
    }
}

impl<W: AsyncWrite> AsyncWrite for CountingWriter<W> {
    fn poll_write(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<std::io::Result<usize>> {
        let this = self.project();
        Poll::Ready(ready!(this.inner.poll_write(cx, buf)).inspect(|count| {
            this.counter.fetch_add((*count) as u64, Ordering::Relaxed);
        }))
    }

    fn poll_flush(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<std::io::Result<()>> {
        self.project().inner.poll_flush(cx)
    }

    fn poll_shutdown(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<std::io::Result<()>> {
        self.project().inner.poll_shutdown(cx)
    }

    fn poll_write_vectored(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        bufs: &[IoSlice<'_>],
    ) -> Poll<std::io::Result<usize>> {
        let this = self.project();
        Poll::Ready(
            ready!(this.inner.poll_write_vectored(cx, bufs)).inspect(|count| {
                this.counter.fetch_add((*count) as u64, Ordering::Relaxed);
            }),
        )
    }

    fn is_write_vectored(&self) -> bool {
        self.inner.is_write_vectored()
    }
}

/// An output file whose bytes have all been written, but whose upload has not
/// been completed yet, together with what its writer returned.
///
/// Until [`AsyncWriteExt::shutdown`] is called on the writer, the file is not
/// visible in the object store, and dropping the writer leaves the target
/// location untouched.
pub type PendingFile<T> = (Box<dyn AsyncWrite + Send + Unpin>, T);

/// Writes every file opened by the demuxer, and commits them only once the
/// input has been read without error and every writer has succeeded.
///
/// `start_file` is called for each file the demuxer opens. It returns the
/// writer future, which is spawned. A writer future writes all of the file's
/// bytes and returns the writer *without* completing it (see [`PendingFile`]).
///
/// If the input or any writer fails, no file is committed: the pending writers
/// are dropped, so a failed `COPY` or `INSERT` leaves the existing files as
/// they were. A failing writer stops the write at once, and its error is
/// returned in preference to the input error it causes.
///
/// After success, the files are committed in parallel. Object stores have no
/// atomic multi-object commit, so if committing one file fails, others may
/// already be committed.
pub async fn write_files_then_commit<T, S, SFut, W>(
    mut demux_task: SpawnedTask<Result<()>>,
    mut file_stream_rx: DemuxedStreamReceiver,
    mut start_file: S,
) -> Result<Vec<T>>
where
    T: Send + 'static,
    S: FnMut(FileMetadata, Receiver<RecordBatch>) -> SFut,
    SFut: Future<Output = Result<W>>,
    W: Future<Output = Result<PendingFile<T>>> + Send + 'static,
{
    let mut writers = JoinSet::new();
    let mut pending = Vec::new();

    // Start a writer for each file the demuxer opens, and watch the writers
    // while doing so, so that a failing writer stops the write at once.
    loop {
        tokio::select! {
            biased;
            Some(joined) = writers.join_next(), if !writers.is_empty() => {
                pending.push(task_result(joined)?);
            }
            next = file_stream_rx.recv() => match next {
                Some((file, rx)) => {
                    writers.spawn(start_file(file, rx).await?);
                }
                None => break,
            }
        }
    }

    // Every file is open. Wait for the input to be read, still watching the writers.
    let input = loop {
        tokio::select! {
            biased;
            Some(joined) = writers.join_next(), if !writers.is_empty() => {
                pending.push(task_result(joined)?);
            }
            input = demux_task.join_unwind_mut() => {
                break input.map_err(|e| DataFusionError::ExecutionJoin(Box::new(e)))?;
            }
        }
    };

    // The demuxer has closed every channel, so the remaining writers finish.
    // A writer that failed makes the demuxer fail with a generic send error, so
    // the writer's error is returned first.
    while let Some(joined) = writers.join_next().await {
        pending.push(task_result(joined)?);
    }
    input?;

    let mut commits = JoinSet::new();
    for (mut writer, result) in pending {
        commits.spawn(async move {
            writer.shutdown().await?;
            Ok::<_, DataFusionError>(result)
        });
    }
    let mut results = Vec::with_capacity(commits.len());
    while let Some(joined) = commits.join_next().await {
        results.push(task_result(joined)?);
    }
    Ok(results)
}

/// Returns a joined task's result, resuming its panic if it panicked.
fn task_result<R>(joined: std::result::Result<Result<R>, JoinError>) -> Result<R> {
    match joined {
        Ok(result) => result,
        Err(e) if e.is_panic() => std::panic::resume_unwind(e.into_panic()),
        Err(e) => Err(DataFusionError::ExecutionJoin(Box::new(e))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion_common::exec_err;
    use std::sync::atomic::AtomicBool;
    use tokio::sync::mpsc;

    /// A writer that records whether it was committed (shut down).
    struct TrackedWriter(Arc<AtomicBool>);

    impl AsyncWrite for TrackedWriter {
        fn poll_write(
            self: Pin<&mut Self>,
            _cx: &mut Context<'_>,
            buf: &[u8],
        ) -> Poll<std::io::Result<usize>> {
            Poll::Ready(Ok(buf.len()))
        }

        fn poll_flush(
            self: Pin<&mut Self>,
            _cx: &mut Context<'_>,
        ) -> Poll<std::io::Result<()>> {
            Poll::Ready(Ok(()))
        }

        fn poll_shutdown(
            self: Pin<&mut Self>,
            _cx: &mut Context<'_>,
        ) -> Poll<std::io::Result<()>> {
            self.0.store(true, Ordering::SeqCst);
            Poll::Ready(Ok(()))
        }
    }

    /// Opens `n` files and returns the demuxer channel plus the batch senders.
    fn open_files(n: usize) -> (DemuxedStreamReceiver, Vec<mpsc::Sender<RecordBatch>>) {
        let (tx, rx) = mpsc::unbounded_channel();
        let mut senders = vec![];
        for i in 0..n {
            let (batch_tx, batch_rx) = mpsc::channel(1);
            let file = FileMetadata {
                path: Path::from(format!("file_{i}")),
                size: Default::default(),
            };
            tx.send((file, batch_rx)).unwrap();
            senders.push(batch_tx);
        }
        (rx, senders)
    }

    /// Runs the write. Writer `fail_writer`, if any, fails once its input ends.
    /// Returns the result and whether each file was committed.
    async fn run(
        demux: SpawnedTask<Result<()>>,
        files: DemuxedStreamReceiver,
        fail_writer: Option<usize>,
    ) -> (Result<Vec<usize>>, Vec<bool>) {
        let committed: Vec<_> =
            (0..2).map(|_| Arc::new(AtomicBool::new(false))).collect();
        let mut i = 0;
        let result = write_files_then_commit(demux, files, |_, mut rx| {
            let flag = Arc::clone(&committed[i]);
            let fail = fail_writer == Some(i);
            i += 1;
            std::future::ready(Ok(async move {
                while rx.recv().await.is_some() {}
                if fail {
                    return exec_err!("disk full");
                }
                let writer: Box<dyn AsyncWrite + Send + Unpin> =
                    Box::new(TrackedWriter(flag));
                Ok((writer, 1))
            }))
        })
        .await;
        let committed = committed.iter().map(|c| c.load(Ordering::SeqCst)).collect();
        (result, committed)
    }

    #[tokio::test]
    async fn commits_every_file_when_the_write_succeeds() {
        let (files, senders) = open_files(2);
        drop(senders);
        let demux = SpawnedTask::spawn(async { Ok(()) });
        let (result, committed) = run(demux, files, None).await;
        assert_eq!(result.unwrap(), vec![1, 1]);
        assert_eq!(committed, vec![true, true]);
    }

    #[tokio::test]
    async fn commits_nothing_when_the_input_fails() {
        let (files, senders) = open_files(2);
        let demux = SpawnedTask::spawn(async move {
            drop(senders);
            exec_err!("divide by zero")
        });
        let (result, committed) = run(demux, files, None).await;
        assert_eq!(
            result.unwrap_err().strip_backtrace(),
            "Execution error: divide by zero"
        );
        assert_eq!(committed, vec![false, false]);
    }

    #[tokio::test]
    async fn returns_the_writer_error_rather_than_the_send_error_it_causes() {
        let (files, senders) = open_files(2);
        // Like the demuxer, fail with a send error once the writer has gone away.
        let demux = SpawnedTask::spawn(async move {
            drop(senders);
            exec_err!("Error sending RecordBatch to file stream!")
        });
        let (result, committed) = run(demux, files, Some(0)).await;
        assert_eq!(
            result.unwrap_err().strip_backtrace(),
            "Execution error: disk full"
        );
        assert_eq!(committed, vec![false, false]);
    }

    #[tokio::test]
    async fn a_failing_writer_stops_the_write_while_the_input_is_still_running() {
        let (files, mut senders) = open_files(2);
        // Close the first file only. The input never finishes on its own.
        drop(senders.remove(0));
        let demux = SpawnedTask::spawn(async move {
            let _keep_open = senders;
            std::future::pending::<()>().await;
            Ok(())
        });
        let (result, committed) = run(demux, files, Some(0)).await;
        assert_eq!(
            result.unwrap_err().strip_backtrace(),
            "Execution error: disk full"
        );
        assert_eq!(committed, vec![false, false]);
    }
}
