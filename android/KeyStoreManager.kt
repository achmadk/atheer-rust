package com.aether.ffi

import android.content.Context
import android.security.keystore.KeyGenParameterSpec
import android.security.keystore.KeyProperties
import java.security.KeyStore
import java.util.UUID
import javax.crypto.KeyGenerator
import javax.crypto.SecretKey

/**
 * Android KeyStore wrapper for storing and retrieving model encryption keys.
 *
 * Keys are generated inside the Android KeyStore (hardware-backed when available)
 * and are never exposed as raw key material to the application process.
 */
object KeyStoreManager {

    private const val KEYSTORE_TYPE = "AndroidKeyStore"
    private const val DEVICE_UID_ALIAS = "atheer_device_uid"
    private const val PREFS_NAME = "atheer_prefs"
    private const val DEVICE_UID_KEY = "atheer_device_uid"

    // ── Model Key Storage ──────────────────────────────────────────

    /**
     * Generate and store a 256-bit AES key in the Android KeyStore.
     *
     * The key is generated inside secure hardware when available and
     * cannot be extracted as plaintext.
     */
    fun generateKey(alias: String) {
        val keyGenerator = KeyGenerator.getInstance(
            KeyProperties.KEY_ALGORITHM_AES,
            KEYSTORE_TYPE
        )
        val spec = KeyGenParameterSpec.Builder(
            alias,
            KeyProperties.PURPOSE_ENCRYPT or KeyProperties.PURPOSE_DECRYPT
        )
            .setBlockModes(KeyProperties.BLOCK_MODE_GCM)
            .setEncryptionPaddings(KeyProperties.ENCRYPTION_PADDING_NONE)
            .setKeySize(256)
            .build()

        keyGenerator.init(spec)
        keyGenerator.generateKey()
    }

    /**
     * Retrieve a SecretKey from the Android KeyStore by alias.
     * Returns `null` if no key with that alias exists.
     */
    fun getKey(alias: String): SecretKey? {
        val keyStore = KeyStore.getInstance(KEYSTORE_TYPE)
        keyStore.load(null)
        return keyStore.getEntry(alias, null) as? KeyStore.SecretKeyEntry
            ?.secretKey
    }

    /**
     * Delete a key from the Android KeyStore by alias.
     */
    fun deleteKey(alias: String) {
        val keyStore = KeyStore.getInstance(KEYSTORE_TYPE)
        keyStore.load(null)
        keyStore.deleteEntry(alias)
    }

    // ── Device UID ─────────────────────────────────────────────────

    /**
     * Initialise (or retrieve) the device UID.
     * Must be called early in the application lifecycle (e.g. `Application.onCreate()`).
     */
    fun initDeviceUid(context: Context) {
        val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        if (!prefs.contains(DEVICE_UID_KEY)) {
            val uid = UUID.randomUUID().toString()
            prefs.edit().putString(DEVICE_UID_KEY, uid).apply()
        }
    }

    /**
     * Return the device UID initialised by [initDeviceUid].
     * Returns `null` if not yet initialised.
     */
    fun getDeviceUid(context: Context): String? {
        val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        return prefs.getString(DEVICE_UID_KEY, null)
    }
}
