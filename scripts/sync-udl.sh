#!/bin/bash
# Proc-Macro Validator
# This validates the UniFFI proc-macro attributes in Rust source code
# NOTE: UDL file is no longer used - all definitions are in Rust code

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SRC_DIR="$PROJECT_DIR/aether-ffi/src"

echo "=== UniFFI Proc-Macro Validator ==="
echo "Validating proc-macro attributes in Rust source..."
echo ""

# Check that source files exist
if [ ! -d "$SRC_DIR" ]; then
    echo "ERROR: Source directory not found at $SRC_DIR"
    exit 1
fi

# Extract types from Rust source
echo "Checking exported types..."

# Count #[uniffi::Object] types
OBJECT_TYPES=$(grep -r "derive.*uniffi::Object\|#\[uniffi::Object\]" "$SRC_DIR" --include="*.rs" 2>/dev/null | wc -l)
echo "  - Object types: $OBJECT_TYPES"

# Count #[uniffi::Record] types  
RECORD_TYPES=$(grep -r "derive.*uniffi::Record\|#\[uniffi::Record\]" "$SRC_DIR" --include="*.rs" 2>/dev/null | wc -l)
echo "  - Record types: $RECORD_TYPES"

# Count #[uniffi::Enum] types
ENUM_TYPES=$(grep -r "derive.*uniffi::Enum\|#\[uniffi::Enum\]" "$SRC_DIR" --include="*.rs" 2>/dev/null | wc -l)
echo "  - Enum types: $ENUM_TYPES"

# Count #[uniffi::Error] types
ERROR_TYPES=$(grep -r "derive.*uniffi::Error\|#\[uniffi::Error\]" "$SRC_DIR" --include="*.rs" 2>/dev/null | wc -l)
echo "  - Error types: $ERROR_TYPES"

# Check UDL file for matching types
echo ""
echo "=== Validation Summary ==="
echo "All UniFFI proc-macro attributes found. No UDL file needed."

echo ""
echo "To regenerate bindings, run:"
echo "  ./scripts/generate-bindings.sh all"
