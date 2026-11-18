# Room model plan: scan → boxes → labels → movable 3D room

Principle: **geometry proposes, images decide.** The scan (TSDF mesh) gives exact positions and sizes but
no meaning. The camera photos give meaning but are poor at geometry. So geometry code builds the boxes,
and the on-device model only *names* them and *suggests* edits (merge, split, relabel, delete) that the
geometry code checks before applying. The model never invents a box or moves one.

Everything runs on the phone. Labeling uses Apple's on-device Foundation Models, which takes image input
from iOS 27 (`Attachment(cgImage)` in a prompt, `@Generable` structured output, `ImageReference` to point
at a specific photo). Vision's on-device classifier gives a cheap first guess and a fallback.

## Pipeline

```
 Scan (existing)          1 Room shell        2 Object boxes       3 Pick photos       4 Name boxes        5 Reason loop       6 Room editor
 TSDF mesh + keyframes →  floor, walls,   →  cluster leftovers →  best 2–3 views  →  Vision prior +  →  merge / split /  →  movable 3D boxes,
 (photo, depth, pose)     ceiling, openings  into upright boxes    per box, cropped    Foundation Models   relabel, ≤3 rounds   save + export
```

### 0. Capture: keep the photos (change to the existing scan)
- While scanning, store a **keyframe** when the camera has moved ~25 cm or turned ~15°, tracking is
  normal and the image is sharp (low angular velocity): a 1024 px JPEG, the 256×192 depth map, the
  camera pose and intrinsics. Cap around 200 frames.
- Save a **scan bundle** per scan (`Documents/Scans/<id>/`: mesh, frames, poses, later `room.json`).
  Every later stage can be rerun from a saved bundle without rescanning, on the phone or on the Mac.
- ARKit's world frame is already gravity-aligned (y up), which every later step relies on.

### 1. Room shell (geometry only)
- Sample a point cloud with normals from the TSDF mesh, downsampled to a 3 cm grid.
- **Floor**: lowest large horizontal plane. **Ceiling**: highest downward-facing plane (often partial).
- **Walls**: RANSAC vertical planes; find the room's dominant direction from a histogram of wall
  normals and snap walls to it (Manhattan), keeping off-axis walls only when strongly supported.
- Intersect wall lines into a closed **floor polygon**; walls become thin boxes, floor to ceiling.
- **Opening candidates**: rectangular gaps in each wall's 2D occupancy grid (doors, windows, passages),
  labeled later from photos.

### 2. Object boxes (geometry only)
- Remove points within ~4 cm of floor, walls and ceiling; what is left is "stuff".
- Detect mid-height horizontal planes (tabletops, shelves, beds) as **support surfaces**.
- Cluster the remainder in a 5 cm occupancy grid (footprint on the floor first, then split vertically at
  support surfaces), so a lamp on a desk becomes its own box resting on the desk box.
- Fit an **upright oriented box** per cluster: yaw from the minimum-area rectangle of the footprint,
  snapped to the room axes when within ~10°; height from its support to the 98th-percentile top.
- Drop noise (tiny or sparse clusters). Output a **scene graph**: room shell plus boxes with id, center,
  size, yaw, what it rests on, which wall it touches, point count.

### 3. Pick photos per box (images, not mesh)
- Score every keyframe for every box: whole box in frame, projected size, frontal view of the box's
  largest face, sharpness, and **occlusion** (box surface depth vs. the keyframe's depth map).
- Keep the best 2–3 views at least 30° apart. Crop the projected box with ~20% context, resize to ~512 px,
  and draw the box outline on the crop so the model knows *which* thing is meant.
- Also one wide view per wall (room type) and per opening candidate (door vs. window).

### 4. Name each box (on-device AI)
- **Vision prior** (fast, deterministic): `ClassifyImageRequest` top-5 labels on the clean crop; the
  foreground-instance mask tells whether the crop holds one object or several (a split hint).
- **Foundation Models** (one short session per box, to stay inside the small context window):
  the 2–3 outlined crops as image attachments, plus facts as text (size in cm, height above floor,
  rests on, touches wall, Vision's guesses). Guided output into a fixed type:

  ```swift
  @Generable struct BoxLabel {
      var category: Category          // fixed list: sofa, chair, table, desk, bed, cabinet, shelf,
                                      // wardrobe, tv, lamp, plant, rug, fridge, oven, sink, toilet,
                                      // door, window, radiator, clutter, unknown …
      var name: String                // "grey three-seat sofa"
      var color: String
      var confidence: Confidence      // low / medium / high
      var containsSeveralObjects: Bool  // → split candidate
      var objectExtendsOutsideBox: Bool // → merge candidate
  }
  ```
- Results stream in, so boxes get their names one by one on screen while the rest are still running.
- If Apple Intelligence is off or unavailable, the Vision label is used and marked low confidence.

### 5. Reason over the whole room (the iteration loop)
- Give the model a compact text version of the scene graph (one line per box: label, confidence,
  size, position, neighbours) plus tools it can call:
  `inspectBox(id)` (re-ask with other views), `compareBoxes(a, b)`.
- It returns a list of **edits**: `merge(a, b)`, `split(a)`, `relabel(a, category)`, `delete(a)`,
  `markOpening(wall, door|window)`, plus a room type ("bedroom", "office").
- Geometry checks each edit before applying it (merge only if adjacent and the union still fits the
  points; split by re-clustering that box tighter, guided by the instance mask; delete only sparse
  boxes). Only changed boxes go back through steps 3–4.
- Up to 3 rounds, or stop when no edits come back.

### 6. Movable 3D room (new screen)
- New **Room** screen in RealityKit (SceneKit is deprecated; the live scan view stays as it is):
  floor, walls with the near walls cut away, boxes colored by category with name tags, and an optional
  ghost of the original scan mesh to compare against.
- Tap a box: name, confidence, the photos it was named from, change category, rename, delete,
  "ask again", merge/split. Drag boxes along the floor with snapping to walls and 90° rotation,
  rotate with two fingers, resize with handles. Undo.
- Save as `room.json` in the scan bundle; export **USDZ** (boxes named by label) and JSON.

## Build order (each step ends in something you can try on the phone)

| # | Milestone | You can try |
|---|-----------|-------------|
| 1 | Put the app in git, raise target to iOS 27, keyframe capture, save/load scan bundles | Scan, reopen saved scans |
| 2 | Room shell + basic Room screen | See floor and walls snapped from a scan |
| 3 | Object boxes + move/rotate/delete | Unlabeled boxes you can drag around |
| 4 | Photo selection + crops + Vision prior | Tap a box, see its best photos and a first guess |
| 5 | Foundation Models naming | Boxes get real names live |
| 6 | Reasoning loop | Wrong splits/merges get fixed automatically |
| 7 | Polish: USDZ/JSON export, undo, mesh overlay, debug view of each stage | Share the room model |

Testing: a Mac test target (like `Tests/main.swift`) runs steps 1–3 on synthetic rooms with known boxes
and on saved scan bundles. Foundation Models also runs on macOS 27, so prompts can be tuned on the Mac
against saved bundles instead of rescanning each time.

## Risks
- **Splitting touching objects** (chairs pushed under a table, a sofa against a shelf) is the hard part;
  step 5 exists mainly for this.
- **Context window** is small and images cost tokens: one session per box, 2–3 crops, measured with
  `tokenCount`.
- **Speed/heat**: 20–40 boxes × a few seconds each. Runs after the scan with progress, results streaming.
- **Requires iOS 27 + Apple Intelligence** on the phone for the model step; Vision-only fallback otherwise.

## Alternative considered
Apple's **RoomPlan** already outputs walls, doors, windows and ~16 furniture box types. It would get
steps 1–2 working faster, but it runs its own capture (sharing it with our TSDF fusion is uncertain), has a
fixed category list and gives no control over splitting. Plan uses our own geometry; RoomPlan can be a
one-day comparison spike if the shell/box quality disappoints.
