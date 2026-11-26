#!/bin/sh

# Build only the 3D scalar CPU propagator. CUDA is not supported on current
# macOS systems. Run this script from the deepwave3D directory.
set -eu

CFLAGS="-Wall -Wextra -pedantic -std=c99 -fPIC -Ofast"
BUILD_DIR=$(mktemp -d)
trap 'rm -rf "$BUILD_DIR"' EXIT
OBJECTS=""

for accuracy in 2 4 6 8; do
    for dtype in float double; do
        object="$BUILD_DIR/scalar_cpu_iso_3d_${accuracy}_${dtype}.o"
        clang $CFLAGS -DDW_ACCURACY=$accuracy -DDW_DTYPE=$dtype \
            -c scalar.c -o "$object"
        OBJECTS="$OBJECTS $object"
    done
done

machine=$(uname -m)
case "$machine" in
    arm64)
        output="libdeepwave_cpu_macos_arm64.dylib"
        ;;
    x86_64)
        output="libdeepwave_cpu_macos_x86_64.dylib"
        ;;
    *)
        echo "Unsupported macOS architecture: $machine" >&2
        exit 1
        ;;
esac

clang -dynamiclib $OBJECTS -o "$BUILD_DIR/$output"
cp "$BUILD_DIR/$output" "$output"
echo "Built $output"
