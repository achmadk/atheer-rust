package com.atheer.ffi.sandbox

import android.app.Service
import android.content.Intent
import android.os.IBinder
import android.os.ParcelFileDescriptor
import android.os.RemoteException
import android.util.Log

/**
 * Android IsolatedService that executes GPU/NPU inference in a sandboxed process.
 *
 * Runs as `android:isolatedProcess="true"` with no network, no filesystem
 * (except the model FD passed via Binder), and no Android permissions.
 *
 * Lifecycle:
 *   1. Main process binds → init(modelFd, modelSize)
 *   2. Shard maps FD, uploads weights to GPU, runs startup probe
 *   3. If probe passes → ready for batch() calls
 *   4. If probe fails → terminate without entering ready state
 *   5. Main process unbinds → shutdown()
 */
class GpuExecutionShardService : Service() {

    companion object {
        private const val TAG = "GpuExecutionShard"
        private const val STATUS_OK = 0
        private const val STATUS_PROBE_FAILED = 1
        private const val STATUS_VERSION_MISMATCH = 2

        init {
            System.loadLibrary("atheer_gpu_shard")
        }
    }

    /** Native handle returned by JNI init. Zero means uninitialized. */
    private var nativeHandle: Long = 0
    private var isReady: Boolean = false
    private var callback: ISandboxedGpuBridge? = null

    // ─── Native methods (implemented in Rust via atheer-accel) ──────

    /** Initialize native GPU backend. Returns opaque handle or 0 on failure. */
    private external fun nativeInit(modelFd: Int, modelSize: Long): Long

    /** Run startup probe: forward known tensor, verify output. */
    private external fun nativeProbe(handle: Long): Boolean

    /** Run batched inference. Returns serialized logits as FloatArray. */
    private external fun nativeBatch(handle: Long, tokenIds: LongArray, positions: LongArray): Array<FloatArray>

    /** Get worker diagnostics JSON. */
    private external fun nativeGetInfo(handle: Long): String

    /** Release GPU resources. */
    private external fun nativeShutdown(handle: Long)

    // ─── Binder interface implementation ───────────────────────────

    private val binder = object : IGpuExecutionShard.Stub() {
        override fun init(modelFd: ParcelFileDescriptor, modelSize: Long): Int {
            Log.i(TAG, "init: modelSize=$modelSize")

            try {
                val fd = modelFd.fd
                val handle = nativeInit(fd, modelSize)
                if (handle == 0L) {
                    Log.e(TAG, "nativeInit returned 0 — backend unavailable")
                    return STATUS_PROBE_FAILED
                }
                nativeHandle = handle

                // Run startup probe attestation (T3.5/T3.6)
                val probePassed = nativeProbe(handle)
                if (!probePassed) {
                    Log.e(TAG, "Startup probe FAILED — terminating")
                    callback?.onProbeResult(false, "Tensor probe mismatch")
                    nativeShutdown(handle)
                    nativeHandle = 0
                    return STATUS_PROBE_FAILED
                }

                callback?.onProbeResult(true, "OK")
                isReady = true
                Log.i(TAG, "init complete — ready for inference")
                return STATUS_OK
            } catch (e: Exception) {
                Log.e(TAG, "init failed: ${e.message}")
                return STATUS_PROBE_FAILED
            }
        }

        override fun batch(tokenIds: LongArray, positions: LongArray): Array<FloatArray> {
            if (!isReady || nativeHandle == 0L) {
                throw RemoteException("Shard not initialized")
            }
            return nativeBatch(nativeHandle, tokenIds, positions)
        }

        override fun getInfo(): String {
            if (nativeHandle == 0L) {
                return """{"status":"uninitialized","backend":"none"}"""
            }
            return nativeGetInfo(nativeHandle)
        }

        override fun shutdown() {
            Log.i(TAG, "shutdown")
            if (nativeHandle != 0L) {
                nativeShutdown(nativeHandle)
                nativeHandle = 0
            }
            isReady = false
            // Stop the service
            stopSelf()
        }
    }

    // ─── Service lifecycle ─────────────────────────────────────────

    override fun onBind(intent: Intent): IBinder {
        Log.i(TAG, "onBind")
        callback = ISandboxedGpuBridge.Stub.asInterface(intent.extras?.getBinder("callback"))
        return binder
    }

    override fun onUnbind(intent: Intent): Boolean {
        Log.i(TAG, "onUnbind")
        if (nativeHandle != 0L) {
            nativeShutdown(nativeHandle)
            nativeHandle = 0
        }
        isReady = false
        callback = null
        return false
    }

    override fun onDestroy() {
        Log.i(TAG, "onDestroy")
        if (nativeHandle != 0L) {
            nativeShutdown(nativeHandle)
            nativeHandle = 0
        }
        isReady = false
        callback = null
        super.onDestroy()
    }
}
