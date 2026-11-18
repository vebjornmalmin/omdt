#!/bin/bash
# Build, install and launch LidarScan on the connected iPhone.
set -euo pipefail
cd "$(dirname "$0")"
DEVICE="${1:-00008150-00124C113443C01C}"
BUNDLE=com.vebjorn.LidarScan
xcodegen generate >/dev/null
xcodebuild -project LidarScan.xcodeproj -scheme LidarScan -configuration Release \
  -destination "id=$DEVICE" -derivedDataPath build -allowProvisioningUpdates -quiet build
APP=build/Build/Products/Release-iphoneos/LidarScan.app
xcrun devicectl device install app --device "$DEVICE" "$APP"
xcrun devicectl device process launch --device "$DEVICE" --terminate-existing "$BUNDLE"
