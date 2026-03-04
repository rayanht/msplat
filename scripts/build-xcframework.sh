#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "=== Building msplat ==="
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

echo "=== Creating XCFramework ==="

# Prepare headers with modulemap
rm -rf build/xcf-headers
mkdir -p build/xcf-headers
cp core/include/msplat_c_api.h build/xcf-headers/
cat > build/xcf-headers/module.modulemap <<'MAP'
module MsplatCore {
    header "msplat_c_api.h"
    export *
}
MAP

# Create XCFramework
rm -rf MsplatCore.xcframework
xcodebuild -create-xcframework \
    -library build/libmsplat_core.a \
    -headers build/xcf-headers \
    -output MsplatCore.xcframework

# Copy metallib as Swift package resource
mkdir -p swift/Sources/Msplat/Resources
cp build/default.metallib swift/Sources/Msplat/Resources/

echo "=== Done ==="
echo "  MsplatCore.xcframework"
echo "  swift/Sources/Msplat/Resources/default.metallib"
