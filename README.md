# omdt

LiDAR room scanner for iPhone Pro. Scan a room, and the app turns it into a movable 3D model:
walls and floor, plus one named box per piece of furniture. Everything runs on the phone.

## What it does

- **Scan.** Fuses LiDAR depth and camera color into a multi-resolution TSDF on the GPU (2 mm voxels
  up close, coarser further away). While scanning it keeps sharp photos with their camera poses,
  finds furniture live and names it with Apple's on-device model.
- **Save and continue.** Each scan is a folder with the mesh, photos, depth, the TSDF volume, the
  ARKit world map and GPS, so a scan can be picked up later in the same room.
- **Room.** Fits floor, ceiling and walls (snapped to the room's main directions). Things in the
  room become upright boxes: touching furniture is split at tabletop height, tall things along a
  wall (kitchen cabinets, curtains) are kept apart from what stands in front of them.
- **Names.** Each box is named from the best photos of it (Vision classifier first, then the
  on-device language model). Clean up merges fragments and removes wall, door and window pieces;
  Review lets the model read the whole room and suggest merges, renames and removals, which the
  geometry checks before applying.
- **3D Room screen.** Move (snaps to walls), turn, rename, merge, split and delete boxes, with undo.
  Tap a box to see the photos it was named from or ask the AI again. Export as USDZ or JSON.

Plan and background: [ROOM_MODEL_PLAN.md](ROOM_MODEL_PLAN.md).

## Limits

- One room per scan works best. A walk through a whole flat gives one outline around everything.
- Naming needs Apple Intelligence on the phone; without it boxes get the image classifier's guess.

## Build

Requires iOS 27, [XcodeGen](https://github.com/yonaskolb/XcodeGen) and an iPhone with LiDAR.

```bash
./run.sh <device-udid>
```

Mac checks (room fitting, boxes, clean up, review, edits, file formats) on synthetic rooms; pass
`room.json` paths to see what clean up and review would do with a real scan:

```bash
./test.sh
```
