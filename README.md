# omdt

LiDAR room scanner for iPhone Pro. Fuses LiDAR depth and camera color into a multi-resolution TSDF
(2 mm voxels up close, coarser further away) and exports a colored PLY mesh.

Next step: turn a scan into a movable 3D room of labeled boxes, see [ROOM_MODEL_PLAN.md](ROOM_MODEL_PLAN.md).

## Build

Requires [XcodeGen](https://github.com/yonaskolb/XcodeGen) and an iPhone with LiDAR.

```bash
./run.sh <device-udid>
```
