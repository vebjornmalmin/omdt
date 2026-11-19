#!/bin/bash
# Build and run the Mac-side checks (TSDF accuracy, PLY and scan bundle round trips).
set -euo pipefail
cd "$(dirname "$0")"
OUT="${TMPDIR:-/tmp}/lidarscan-tests"
swiftc -O Sources/TSDFVolume.swift Sources/MeshExporter.swift Sources/ScanBundle.swift Sources/RoomShell.swift Tests/main.swift -o "$OUT"
"$OUT"
