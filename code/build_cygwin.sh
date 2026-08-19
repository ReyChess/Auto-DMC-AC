#!/usr/bin/env bash
set -euo pipefail

unset GCC_EXEC_PREFIX COMPILER_PATH CPATH CPLUS_INCLUDE_PATH C_INCLUDE_PATH LIBRARY_PATH

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$ROOT_DIR/build"
SRC_DIR="$ROOT_DIR/src"
CC=/usr/bin/gcc
CXX=/usr/bin/g++

for tool in "$CC" "$CXX"; do
  if [[ ! -x "$tool" ]]; then
    echo "ERROR: $tool is not available in Cygwin." >&2
    echo "Install/update gcc-core and gcc-g++." >&2
    exit 127
  fi
done

# Compile in Cygwin's local temporary filesystem first.  This avoids
# intermittent NTFS/Cygwin failures while creating or opening intermediate files.
TMP_BUILD="$(/usr/bin/mktemp -d /tmp/auto_dmc_build.XXXXXX)"
cleanup() { /usr/bin/rm -rf "$TMP_BUILD"; }
trap cleanup EXIT

# The Windows launcher also creates BUILD_DIR, but create/check it here for
# direct invocations of this script.
/usr/bin/mkdir -p "$BUILD_DIR"
if [[ ! -d "$BUILD_DIR" ]]; then
  echo "ERROR: build directory could not be created: $BUILD_DIR" >&2
  exit 1
fi
/usr/bin/rm -f "$BUILD_DIR"/*.exe 2>/dev/null || true

CFLAGS=(-O3 -msse4.2 -std=gnu11)
CXXFLAGS=(-O3 -std=gnu++17)

CXX_TARGET="$($CXX -dumpmachine)"
CXX_VERSION="$($CXX -dumpversion)"
CXX_INCLUDE="/usr/lib/gcc/${CXX_TARGET}/${CXX_VERSION}/include/c++"
CXX_TARGET_INCLUDE="${CXX_INCLUDE}/${CXX_TARGET}"
CXX_BACKWARD_INCLUDE="${CXX_INCLUDE}/backward"

if [[ ! -f "${CXX_INCLUDE}/filesystem" ]]; then
  echo "ERROR: C++ standard header not found: ${CXX_INCLUDE}/filesystem" >&2
  echo "Detected compiler: $($CXX --version | head -n 1)" >&2
  echo "Reinstall the Cygwin gcc-g++ package." >&2
  exit 1
fi

CXXFLAGS+=( -isystem "$CXX_INCLUDE" )
[[ -d "$CXX_TARGET_INCLUDE" ]] && CXXFLAGS+=( -isystem "$CXX_TARGET_INCLUDE" )
[[ -d "$CXX_BACKWARD_INCLUDE" ]] && CXXFLAGS+=( -isystem "$CXX_BACKWARD_INCLUDE" )

FS_TEST="$TMP_BUILD/.filesystem_test.cpp"
cat > "$FS_TEST" <<'CPP'
#include <filesystem>
int main() { std::filesystem::path p{"."}; return p.empty(); }
CPP
if ! "$CXX" "${CXXFLAGS[@]}" "$FS_TEST" -o "$TMP_BUILD/.filesystem_test.exe"; then
  echo "ERROR: the Cygwin compiler could not compile a minimal <filesystem> test." >&2
  exit 1
fi

echo "Using C compiler: $($CC --version | head -n 1)"
echo "Using C++ compiler: $($CXX --version | head -n 1)"
echo "Using C++ headers: $CXX_INCLUDE"
echo
echo "Compiling mining backends with Cygwin GCC..."
"$CC" "${CFLAGS[@]}" -DM4_DIRECT_MAXIMAL_POS_NEG=0 -DM4_GENERATE_NEGATIVES=0 \
  "$SRC_DIR/miner_global.c" -lm -o "$TMP_BUILD/miner_global_p.exe"
"$CC" "${CFLAGS[@]}" -DM4_DIRECT_MAXIMAL_POS_NEG=1 -DM4_GENERATE_NEGATIVES=1 \
  "$SRC_DIR/miner_global.c" -lm -o "$TMP_BUILD/miner_global_pn.exe"
"$CC" "${CFLAGS[@]}" -DM4_DIRECT_MAXIMAL_POS_NEG=0 -DM4_GENERATE_NEGATIVES=0 \
  "$SRC_DIR/miner_cns.c" -lm -o "$TMP_BUILD/miner_cns_p.exe"
"$CC" "${CFLAGS[@]}" -DM4_DIRECT_MAXIMAL_POS_NEG=1 -DM4_GENERATE_NEGATIVES=1 \
  "$SRC_DIR/miner_cns.c" -lm -o "$TMP_BUILD/miner_cns_pn.exe"

echo "Compiling dispatcher and automatic selector with Cygwin G++..."
"$CXX" "${CXXFLAGS[@]}" "$SRC_DIR/dmc_miner_dispatcher.cpp" -o "$TMP_BUILD/dmc_miner_unified.exe"
"$CXX" "${CXXFLAGS[@]}" "$SRC_DIR/auto_dmc_ac.cpp" -o "$TMP_BUILD/auto_dmc_ac.exe"

echo
echo "Copying verified executables to: $BUILD_DIR"
expected=(
  miner_global_p.exe miner_global_pn.exe
  miner_cns_p.exe miner_cns_pn.exe
  dmc_miner_unified.exe auto_dmc_ac.exe
)
for exe in "${expected[@]}"; do
  if [[ ! -f "$TMP_BUILD/$exe" ]]; then
    echo "ERROR: compiler did not produce expected executable: $exe" >&2
    exit 1
  fi
  /usr/bin/cp -f "$TMP_BUILD/$exe" "$BUILD_DIR/$exe"
done

echo
echo "Cygwin compilation completed successfully."
echo "Executables generated in: $BUILD_DIR"
for exe in "${expected[@]}"; do
  if [[ ! -f "$BUILD_DIR/$exe" ]]; then
    echo "ERROR: expected executable was not generated: $exe" >&2
    exit 1
  fi
  size="$(/usr/bin/stat -c '%s' "$BUILD_DIR/$exe" 2>/dev/null || echo '?')"
  printf '  OK: %-25s %s bytes\n' "$exe" "$size"
done
