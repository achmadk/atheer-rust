#!/bin/bash
# Generate mobile bindings for aether-ffi
# Usage: ./scripts/generate-bindings.sh [swift|kotlin|all]
#
# NOTE: This script now uses UniFFI's library mode to generate bindings
#       directly from the compiled library (no UDL required).

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SWIFT_DIR="$PROJECT_DIR/ios"
KOTLIN_DIR="$PROJECT_DIR/android/kotlin"

echo "=== Aether FFI Bindings Generator (proc-macro mode) ==="
echo "Swift: $SWIFT_DIR"
echo "Kotlin: $KOTLIN_DIR"
echo ""

# Build release library first
echo "Building release library..."
cd "$PROJECT_DIR"
cargo build --release -p atheer-ffi

# Create directories
mkdir -p "$SWIFT_DIR"
mkdir -p "$KOTLIN_DIR"

# Get the library path for library mode
LIB_PATH="target/release/libatheer_ffi.so"
if [ ! -f "$LIB_PATH" ]; then
    LIB_PATH="target/release/libatheer_ffi.dylib"
fi

# generate_swift() {
#     echo "Generating Swift bindings..."
#     cargo run -p atheer-bindgen -- library "$LIB_PATH" swift "$SWIFT_DIR"
#     echo "Swift bindings written to $SWIFT_DIR"
# }

# generate_kotlin() {
#     echo "Generating Kotlin bindings..."
#     cargo run -p atheer-bindgen -- library "$LIB_PATH" kotlin "$KOTLIN_DIR"
#     echo "Kotlin bindings written to $KOTLIN_DIR"
# }

# case "${1:-all}" in
#     swift)
#         generate_swift
#         ;;
#     kotlin)
#         generate_kotlin
#         ;;
#     all)
#         generate_swift
#         generate_kotlin
#         ;;
#     *)
#         echo "Usage: $0 [swift|kotlin|all]"
#         exit 1
#         ;;
# esac

echo ""
echo "=== Done ==="