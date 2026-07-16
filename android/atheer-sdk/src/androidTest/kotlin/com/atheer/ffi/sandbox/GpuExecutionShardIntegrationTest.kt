package com.atheer.ffi.sandbox

import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.os.IBinder
import android.os.ParcelFileDescriptor
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.After
import org.junit.Assert.assertArrayEquals
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
 * Integration tests for the GPU execution shard worker lifecycle.
 *
 * Tests the full AIDL-wired service lifecycle on a real Android device:
 * bind -> init -> batch -> shutdown.
 *
 * REQUIREMENTS:
 * - Android device or emulator (API 26+)
 * - A small .gguf model file at `/data/local/tmp/test_model.gguf`
 * - `libatheer_gpu_shard.so` compiled for arm64-v8a and bundled
 * - GPU/NPU backend available on device
 *
 * Run on-device via:
 *   ./gradlew :atheer-sdk:connectedAndroidTest
 */
@RunWith(AndroidJUnit4::class)
@Ignore("Requires Android device with compiled GPU shard .so")
class GpuExecutionShardIntegrationTest {

    private val context = ApplicationProvider.getApplicationContext<Context>()
    private val latch = CountDownLatch(1)
    private var shardService: IGpuExecutionShard? = null
    private var isBound = false

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
        val intent = Intent(context, GpuExecutionShardService::class.java)
        context.bindService(intent, connection, Context.BIND_AUTO_CREATE)
        isBound = true
    }

    @After
    fun tearDown() {
        if (isBound) {
            shardService?.shutdown()
            context.unbindService(connection)
            isBound = false
        }
    }

    @Test
    fun testBindServiceSucceeds() {
        assertTrue("Service should bind within 5 seconds", latch.await(5, TimeUnit.SECONDS))
        assertNotNull("shardService should be non-null after bind", shardService)
    }

    @Test
    fun testServiceLifecycleInitBatchShutdown() {
        assertTrue("Service should bind", latch.await(5, TimeUnit.SECONDS))
        assertNotNull("shardService should be non-null", shardService)

        // Locate test model file
        val modelFile = File("/data/local/tmp/test_model.gguf")
        assertTrue("Test model must exist at /data/local/tmp/test_model.gguf", modelFile.exists())

        val modelFd = ParcelFileDescriptor.open(modelFile, ParcelFileDescriptor.MODE_READ_ONLY)
        assertNotNull("modelFd should be non-null", modelFd)

        // 1. init
        val status = shardService!!.init(modelFd, modelFile.length())
        assertEquals("init should return STATUS_OK (0)", 0, status)

        // 2. getInfo returns worker diagnostics
        val info = shardService!!.info
        assertNotNull("getInfo should return a JSON string", info)
        assertTrue("getInfo should contain backend info", info!!.contains("backend"))

        // 3. batch with known tokens
        val tokenIds = longArrayOf(1L, 42L, 100L)
        val positions = longArrayOf(0L, 1L, 2L)
        val logits = shardService!!.batch(tokenIds, positions)
        assertNotNull("batch should return logits", logits)
        assertEquals("logits should have one row per token", 3, logits.size)
        assertTrue("each logit row should have positive length", logits[0].size > 0)

        // 4. shutdown
        shardService!!.shutdown()
    }

    @Test
    fun testBatchBeforeInitFails() {
        assertTrue("Service should bind", latch.await(5, TimeUnit.SECONDS))
        assertNotNull("shardService should be non-null", shardService)

        // Calling batch before init should throw RemoteException
        try {
            shardService!!.batch(longArrayOf(1L), longArrayOf(0L))
            // If we reach here, the service unexpectedly accepted the call
            org.junit.Assert.fail("batch() before init() should throw RemoteException")
        } catch (e: android.os.RemoteException) {
            // Expected — service is not initialized
        }
    }
}
