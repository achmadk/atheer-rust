import Foundation
import Security

/// iOS Keychain wrapper for storing and retrieving model encryption keys.
///
/// All keys are stored with `kSecAttrAccessibleWhenUnlockedThisDeviceOnly`,
/// meaning they are not backed up to iCloud and are only accessible when
/// the device is unlocked.
enum AtheerKeychainError: Error {
    case storeFailed(OSStatus)
    case retrieveFailed(OSStatus)
    case deleteFailed(OSStatus)
    case invalidKeyData
}

final class AtheerKeychain {

    // MARK: - Model Key Storage

    /// Store a 32-byte AES-256 key under the given identifier.
    static func store(key: [UInt8], keyId: String) throws {
        guard key.count == 32 else { throw AtheerKeychainError.invalidKeyData }

        let query: [CFString: Any] = [
            kSecClass: kSecClassGenericPassword,
            kSecAttrAccount: keyId,
            kSecAttrService: "atheer-model-key",
            kSecAttrAccessible: kSecAttrAccessibleWhenUnlockedThisDeviceOnly,
            kSecValueData: Data(key),
        ]

        // Delete any existing item with this keyId first
        SecItemDelete(query as CFDictionary)

        let status = SecItemAdd(query as CFDictionary, nil)
        guard status == errSecSuccess else {
            throw AtheerKeychainError.storeFailed(status)
        }
    }

    /// Retrieve a 32-byte AES-256 key by its identifier.
    static func retrieve(keyId: String) throws -> [UInt8] {
        let query: [CFString: Any] = [
            kSecClass: kSecClassGenericPassword,
            kSecAttrAccount: keyId,
            kSecAttrService: "atheer-model-key",
            kSecReturnData: true,
            kSecMatchLimit: kSecMatchLimitOne,
        ]

        var result: AnyObject?
        let status = SecItemCopyMatching(query as CFDictionary, &result)
        guard status == errSecSuccess else {
            throw AtheerKeychainError.retrieveFailed(status)
        }

        guard let data = result as? Data, data.count == 32 else {
            throw AtheerKeychainError.invalidKeyData
        }

        return [UInt8](data)
    }

    /// Delete a key by its identifier.
    static func delete(keyId: String) throws {
        let query: [CFString: Any] = [
            kSecClass: kSecClassGenericPassword,
            kSecAttrAccount: keyId,
            kSecAttrService: "atheer-model-key",
        ]

        let status = SecItemDelete(query as CFDictionary)
        guard status == errSecSuccess || status == errSecItemNotFound else {
            throw AtheerKeychainError.deleteFailed(status)
        }
    }

    // MARK: - Device UID

    private static let deviceUidKey = "atheer_device_uid"

    /// Returns the device UID, generating and storing a new UUID if none exists.
    static func deviceUid() -> String {
        if let existing = try? retrieve(keyId: deviceUidKey) {
            return String(decoding: Data(existing), as: UTF8.self)
        }

        let newUid = UUID().uuidString
        // Store as UTF-8 bytes under the reserved key
        if let uidData = newUid.data(using: .utf8) {
            let bytes = [UInt8](uidData)
            try? store(key: bytes + [UInt8](repeating: 0, count: max(0, 32 - bytes.count)),
                       keyId: deviceUidKey)
        }
        return newUid
    }
}
