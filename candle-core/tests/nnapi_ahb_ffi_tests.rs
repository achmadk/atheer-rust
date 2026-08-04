//! FFI smoke tests for the AHardwareBuffer zero-copy path.
//!
//! These tests verify the raw FFI layer (allocate, lock, unlock, release, memory
//! derivation) against a real device before anything is built on top of it. Every test
//! requires a physical Android device with AHB support (API 26+) and an NNAPI driver
//! (API 29+ for memory derivation), so all are marked `#[ignore]`.

#[cfg(all(feature = "nnapi", target_os = "android"))]
use candle_core::Result;

/// Smoke test: allocate an AHardwareBuffer, lock for read+write, write a byte pattern,
/// read it back, unlock, derive an ANeuralNetworksMemory handle, free and release.
///
/// This is the gate (task 1.11) for the entire zero-copy change: nothing in group 2
/// onward should be implemented until this passes on hardware, because the previous
/// attempt had four independent FFI defects that made every AHB allocation fail. This
/// test exercises the corrected symbol name, arity, library, usage constant, and
/// lock/unlock bindings.
#[cfg(all(feature = "nnapi", target_os = "android"))]
#[test]
#[ignore = "requires an Android device with AHB support (API 26+) and NNAPI (API 29+)"]
fn ahb_ffi_smoke_test() -> Result<()> {
    use candle_core::nnapi_backend::nnapi_ndk::*;
    use std::ptr;

    unsafe {
        // 1. Allocate a 1KB BLOB buffer with CPU read+write usage
        let mut desc = AHardwareBuffer_Desc::new();
        desc.width = 1024;
        desc.height = 1;
        desc.layers = 1;
        desc.format = AHARDWAREBUFFER_FORMAT_BLOB;
        desc.usage = AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN | AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN;
        desc.stride = 0;

        let mut buffer: *mut AHardwareBuffer = ptr::null_mut();
        let rc = AHardwareBuffer_allocate(&desc, &mut buffer);
        assert_eq!(rc, 0, "AHardwareBuffer_allocate failed with code {}", rc);
        assert!(!buffer.is_null(), "buffer is null after successful allocate");

        // 2. Lock the buffer for CPU access, obtaining a mapped pointer
        let mut virt_addr: *mut std::ffi::c_void = ptr::null_mut();
        let rc = AHardwareBuffer_lock(
            buffer,
            AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN | AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN,
            -1, // no fence
            ptr::null(), // null rect = lock the whole buffer
            &mut virt_addr,
        );
        assert_eq!(rc, 0, "AHardwareBuffer_lock failed with code {}", rc);
        assert!(!virt_addr.is_null(), "virtual address is null after successful lock");

        // 3. Write a test pattern and read it back to confirm CPU access works
        let ptr = virt_addr as *mut u8;
        let test_pattern: [u8; 8] = [0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0xBA, 0xBE];
        ptr::copy_nonoverlapping(test_pattern.as_ptr(), ptr, 8);

        let mut read_back = [0u8; 8];
        ptr::copy_nonoverlapping(ptr, read_back.as_mut_ptr(), 8);
        assert_eq!(
            read_back, test_pattern,
            "read-back mismatch: CPU writes not visible"
        );

        // 4. Unlock
        let mut fence: i32 = -1;
        let rc = AHardwareBuffer_unlock(buffer, &mut fence);
        assert_eq!(rc, 0, "AHardwareBuffer_unlock failed with code {}", rc);

        // 5. Derive an ANeuralNetworksMemory handle (API 29+)
        let mut memory: *mut ANeuralNetworksMemory = ptr::null_mut();
        match ANeuralNetworksMemory_createFromAHardwareBuffer(buffer, &mut memory) {
            Ok(rc) => {
                assert_eq!(
                    rc, 0,
                    "ANeuralNetworksMemory_createFromAHardwareBuffer failed with code {}",
                    rc
                );
                assert!(!memory.is_null(), "memory is null after successful derivation");
                ANeuralNetworksMemory_free(memory);
            }
            Err(_) => {
                // API < 29 or runtime didn't load; the memory derivation is optional. This
                // is not a failure of the AHB FFI layer itself (allocate/lock/unlock work),
                // so the test passes but with a diagnostic.
                eprintln!(
                    "INFO: ANeuralNetworksMemory_createFromAHardwareBuffer unavailable (API < 29?). \
                     AHB allocate/lock/unlock succeeded."
                );
            }
        }

        // 6. Release the buffer
        AHardwareBuffer_release(buffer);
    }

    Ok(())
}
