package com.atheer.ffi.sandbox;

/**
 * AIDL interface for the GPU execution shard worker process.
 *
 * Version: 1
 *
 * The worker process exposes a thin dispatch-only interface:
 * init() to load the model via FD, batch() for inference,
 * getInfo() for diagnostics, shutdown() for cleanup.
 */
interface IGpuExecutionShard {

    /**
     * Initialize the shard with a model file descriptor.
     *
     * @param modelFd readable ParcelFileDescriptor for the .gguf model file
     * @param modelSize size of the model file in bytes
     * @return int status code: 0 = success, 1 = probe failed, 2 = incompatible version
     */
    int init(in ParcelFileDescriptor modelFd, in long modelSize);

    /**
     * Run batched inference tokens.
     *
     * Each token is decoded sequentially inside the worker (single forward pass
     * per token). Returns logits for all tokens in one IPC round-trip.
     *
     * @param tokenIds array of token IDs to decode
     * @param positions array of position indices (same length as tokenIds)
     * @return float[][] logits, one row per token, each row of size vocabSize
     */
    float[][] batch(in long[] tokenIds, in long[] positions);

    /**
     * Get worker diagnostics.
     *
     * @return String JSON with worker info: backend, device, probe result, uptime
     */
    String getInfo();

    /**
     * Graceful shutdown. Releases GPU resources and closes FDs.
     */
    void shutdown();
}
