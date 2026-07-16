package com.atheer.ffi.sandbox

import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.os.IBinder
import android.os.ParcelFileDescriptor
import android.os.RemoteException
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Ignore
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

/**
 * Integration tests for GPU execution shard probe failure behavior.
 *
 * Verifies that the shard correctly rejects a corrupt/unexpected model
 * and terminates without entering ready state (T3.6 attestation).
 *
 * REQUIREMENTS:
 * - Android device or emulator (API 26+)
 * - A corrupt/invalid model file at `/data/local/tmp/test_model_corrupt.gguf`
 *   (e.g., a 1KB random file — not a valid GGUF header)
 * - `libatheer_gpu_shard.so` compiled for arm64-v8a and bundled
 *
 * Run on-device via:
 *   ./gradlew :atheer-sdk:connectedAndroidTest
 */
@RunWith(AndroidJUnit4::class)
@Ignore("Requires Android device with compiled GPU shard .so and corrupt model file")
class GpuExecutionShardProbeFailureTest {

    private val context = ApplicationProvider.getApplicationContext<Context>()
    private val latch = CountDownLatch(1)
    private var shardService: IGpuExecutionShard? = null
    private var isBound = false
    private var probeResult: Boolean? = null
    private var probeDetail: String? = null

    /**
     * A minimal ISandboxedGpuBridge callback that captures probe results.
     */
    private val callbackBinder = object : ISandboxedGpuBridge.Stub() {
        override fun onCrash(reason: String?) {
            // Not expected in probe failure test — probe failure is clean
        }

        override fun onProbeResult(passed: Boolean, detail: String?) {
            probeResult = passed
            probeDetail = detail
        }
    }

    private val connection = object : ServiceConnection {
        override fun onServiceConnected(name: ComponentName?, service: IBinder?) {
            shardService = IGpuExecutionShard.Stub.asInterface(service)
            latch.countDown()
        }

        override fun onServiceDisconnected(name: ComponentName?) {
            shardService = null
            isBound = false
        }
    }

    @Before
    fun setUp() {
        val intent = Intent(context, GpuExecutionShardService::class.java).apply {
            putExtra("callback", callbackBinder)
        }
        context.bindService(intent, connection, Context.BIND_AUTO_CREATE)
        isBound = true
    }

    @After
    fun tearDown() {
        if (isBound) {
            try {
                shardService?.shutdown()
            } catch (_: RemoteException) {
                // Service may already be dead after probe failure
            }
            context.unbindService(connection)
            isBound = false
        }
    }

    @Test
    fun testProbeFailureReturnsErrorStatus() {
        assertTrue("Service should bind", latch.await(5, TimeUnit.SECONDS))
        assertNotNull("shardService should be non-null", shardService)

        // Use a corrupt/invalid model file
        val corruptFile = File("/data/local/tmp/test_model_corrupt.gguf")
        assertTrue(
            "Corrupt test model must exist at /data/local/tmp/test_model_corrupt.gguf",
            corruptFile.exists()
        )

        val modelFd = ParcelFileDescriptor.open(corruptFile, ParcelFileDescriptor.MODE_READ_ONLY)
        assertNotNull("modelFd should be non-null", modelFd)

        // init with corrupt model — should fail probe
        val status = shardService!!.init(modelFd, corruptFile.length())
        assertEquals("init with corrupt model should return STATUS_PROBE_FAILED (1)", 1, status)
    }

    @Test
    fun testProbeFailureCallbackReceived() {
        assertTrue("Service should bind", latch.await(5, TimeUnit.SECONDS))
        assertNotNull("shardService should be non-null", shardService)

        val corruptFile = File("/data/local/tmp/test_model_corrupt.gguf")
        assertTrue(
            "Corrupt test model must exist at /data/local/tmp/test_model_corrupt.gguf",
            corruptFile.exists()
        )

        val modelFd = ParcelFileDescriptor.open(corruptFile, ParcelFileDescriptor.MODE_READ_ONLY)

        // Trigger probe failure
        shardService!!.init(modelFd, corruptFile.length())

        // The callback should have been invoked with passed=false
        assertNotNull("onProbeResult should have been called", probeResult)
        assertEquals("probe should report failure", false, probeResult)
    }

    @Test
    fun testBatchAfterProbeFailureThrows() {
        assertTrue("Service should bind", latch.await(5, TimeUnit.SECONDS))
        assertNotNull("shardService should be non-null", shardService)

        val corruptFile = File("/data/local/tmp/test_model_corrupt.gguf")
        assertTrue(
            "Corrupt test model must exist at /data/local/tmp/test_model_corrupt.gguf",
            corruptFile.exists()
        )

        val modelFd = ParcelFileDescriptor.open(corruptFile, ParcelFileDescriptor.MODE_READ_ONLY)

        // init fails
        shardService!!.init(modelFd, corruptFile.length())

        // batch should throw because shard never entered ready state
        try {
            shardService!!.batch(longArrayOf(1L), longArrayOf(0L))
            org.junit.Assert.fail("batch() after probe failure should throw RemoteException")
        } catch (e: RemoteException) {
            // Expected — shard terminated after probe failure
        }
    }
}
