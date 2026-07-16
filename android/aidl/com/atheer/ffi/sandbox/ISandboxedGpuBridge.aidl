package com.atheer.ffi.sandbox;

/**
 * Callback interface from the GPU execution shard worker to the main process.
 *
 * Version: 1
 *
 * The worker calls back to report crashes and probe results.
 */
interface ISandboxedGpuBridge {

    /**
     * Called when the worker process encounters a crash / abnormal termination.
     *
     * @param reason description of the crash (e.g. "SIGSEGV", "OOM", "driver hang")
     */
    void onCrash(String reason);

    /**
     * Called with the result of the startup GPU probe attestation.
     *
     * @param passed true if the known tensor forward produced the expected output
     * @param detail machine-readable detail string (e.g. "expected=0.5, got=0.0")
     */
    void onProbeResult(boolean passed, String detail);
}
