// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Module containing helper methods/traits related to
//! orchestrating file serialization, streaming to object store,
//! parallelization, and abort handling

use std::sync::Arc;

use super::demux::DemuxedStreamReceiver;
use super::{BatchSerializer, ObjectWriterBuilder, write_files_then_commit};
use crate::file_compression_type::FileCompressionType;
use datafusion_common::error::Result;
use datafusion_common::{
    DataFusionError, exec_datafusion_err, exec_err, internal_datafusion_err,
};
use datafusion_common_runtime::SpawnedTask;
use datafusion_execution::TaskContext;

use arrow::array::RecordBatch;
use bytes::Bytes;
use object_store::ObjectStore;
use tokio::io::{AsyncWrite, AsyncWriteExt};
use tokio::sync::mpsc::{self, Receiver};

type WriterType = Box<dyn AsyncWrite + Send + Unpin>;

/// Serializes a single data stream in parallel and writes to an ObjectStore concurrently.
/// Data order is preserved.
///
/// On success, returns the writer without completing it, so the caller can
/// decide whether to commit the file. On error, the writer is dropped and the
/// file is not committed.
pub(crate) async fn serialize_rb_stream_to_object_store(
    mut data_rx: Receiver<RecordBatch>,
    serializer: Arc<dyn BatchSerializer>,
    mut writer: WriterType,
) -> Result<(WriterType, usize)> {
    let (tx, mut rx) =
        mpsc::channel::<SpawnedTask<Result<(usize, Bytes), DataFusionError>>>(100);
    let serialize_task = SpawnedTask::spawn(async move {
        // Some serializers (like CSV) handle the first batch differently than
        // subsequent batches, so we track that here.
        let mut initial = true;
        while let Some(batch) = data_rx.recv().await {
            let serializer_clone = Arc::clone(&serializer);
            let task = SpawnedTask::spawn(async move {
                let num_rows = batch.num_rows();
                let bytes = serializer_clone.serialize(batch, initial)?;
                Ok((num_rows, bytes))
            });
            if initial {
                initial = false;
            }
            tx.send(task).await.map_err(|_| {
                internal_datafusion_err!("Unknown error writing to object store")
            })?;
        }
        Ok(())
    });

    let mut row_count = 0;
    while let Some(task) = rx.recv().await {
        match task.join().await {
            Ok(Ok((cnt, bytes))) => {
                match writer.write_all(&bytes).await {
                    Ok(_) => (),
                    Err(e) => {
                        return exec_err!("Error writing to object store: {e}");
                    }
                }
                row_count += cnt;
            }
            Ok(Err(e)) => {
                return Err(e);
            }
            Err(e) => {
                // Handle task panic or cancellation
                return Err(exec_datafusion_err!(
                    "Serialization task panicked or was cancelled: {e}"
                ));
            }
        }
    }

    match serialize_task.join().await {
        Ok(Ok(_)) => (),
        Ok(Err(e)) => return Err(e),
        Err(_) => {
            return Err(internal_datafusion_err!(
                "Unknown error writing to object store"
            ));
        }
    }
    Ok((writer, row_count))
}

/// Orchestrates multipart put of a dynamic number of output files from a single input stream
/// for any statelessly serialized file type. That is, any file type for which each [RecordBatch]
/// can be serialized independently of all other [RecordBatch]s.
///
/// The files are committed only if the input and every writer succeed; see
/// [`write_files_then_commit`].
pub async fn spawn_writer_tasks_and_join(
    context: &Arc<TaskContext>,
    serializer: Arc<dyn BatchSerializer>,
    compression: FileCompressionType,
    compression_level: Option<u32>,
    object_store: Arc<dyn ObjectStore>,
    demux_task: SpawnedTask<Result<()>>,
    file_stream_rx: DemuxedStreamReceiver,
) -> Result<u64> {
    let buffer_size = context
        .session_config()
        .options()
        .execution
        .objectstore_writer_buffer_size;
    let row_counts = write_files_then_commit(
        demux_task,
        file_stream_rx,
        |file_metadata, rb_stream| {
            let writer = ObjectWriterBuilder::new(
                compression,
                &file_metadata.path,
                Arc::clone(&object_store),
            )
            .with_buffer_size(Some(buffer_size))
            .with_compression_level(compression_level)
            .with_bytes_written_counter(file_metadata.size)
            .build();
            let serializer = Arc::clone(&serializer);
            async move {
                let writer = writer?;
                Ok(serialize_rb_stream_to_object_store(
                    rb_stream, serializer, writer,
                ))
            }
        },
    )
    .await?;
    Ok(row_counts.into_iter().sum::<usize>() as u64)
}
