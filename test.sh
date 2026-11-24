#!/bin/bash
# Build and run the Mac-side checks (TSDF accuracy, PLY and scan bundle round trips).
set -euo pipefail
cd "$(dirname "$0")"
OUT="${TMPDIR:-/tmp}/lidarscan-tests"
swiftc -O Sources/TSDFVolume.swift Sources/GPUFusion.swift Sources/MeshExporter.swift Sources/ScanBundle.swift Sources/RoomShell.swift Sources/ObjectBoxes.swift Sources/RoomEdits.swift Sources/RoomScene.swift Sources/RoomCorrection.swift Sources/RoomReview.swift Sources/BoxViews.swift Sources/ObjectModeler.swift Sources/RoomObjects.swift Tests/main.swift -o "$OUT"
"$OUT" "$@"
