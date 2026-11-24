import Foundation
import simd

// Synthetic check: a 10 cm sphere at 0.5 m, observed from far and then from close.
let W = 256, H = 192
let fx: Float = 200, fy: Float = 200, cx: Float = 128, cy: Float = 96
let radius: Float = 0.05

func render(sphereCenter c: SIMD3<Float>) -> ([Float], [UInt8]) {
    var depth = [Float](repeating: 0, count: W * H)
    for v in 0..<H { for u in 0..<W {
        let dir = SIMD3<Float>((Float(u) - cx) / fx, -(Float(v) - cy) / fy, -1)
        // |d*dir - c|^2 = r^2
        let a = simd_dot(dir, dir), b = -2 * simd_dot(dir, c), cc = simd_dot(c, c) - radius * radius
        let disc = b * b - 4 * a * cc
        if disc > 0 {
            let d = (-b - disc.squareRoot()) / (2 * a)
            depth[v * W + u] = d      // z-depth since dir.z = -1
        } else {
            depth[v * W + u] = 1.5    // back wall
        }
    } }
    return (depth, [UInt8](repeating: 2, count: W * H))
}

func run(distance: Float) {
    let vol = TSDFVolume()
    let center = SIMD3<Float>(0, 0, -distance)
    var (depth, conf) = render(sphereCenter: center)
    let t0 = Date()
    for _ in 0..<5 {
        depth.withUnsafeBufferPointer { dp in conf.withUnsafeBufferPointer { cp in
            let f = DepthFrame(depth: dp.baseAddress!, conf: cp.baseAddress!, width: W, height: H,
                               depthStride: W, confStride: W, fx: fx, fy: fy, cx: cx, cy: cy,
                               camToWorld: matrix_identity_float4x4)
            vol.integrate(depth: f, color: nil)
        } }
    }
    let tInt = Date().timeIntervalSince(t0) / 5 * 1000
    let t1 = Date()
    let meshes = vol.mesh(blocks: vol.allBlocks)
    let tMesh = Date().timeIntervalSince(t1) * 1000

    var tris = [Int](repeating: 0, count: 5)
    var sumSq: Double = 0, n = 0, maxErr: Float = 0
    for m in meshes {
        tris[m.level] += m.idx.count / 3
        for p in m.pos where simd_distance(p, center) < radius + 0.05 {   // sphere vertices only
            let e = abs(simd_distance(p, center) - radius)
            sumSq += Double(e * e); n += 1; maxErr = max(maxErr, e)
        }
    }
    let rms = n > 0 ? (sumSq / Double(n)).squareRoot() * 1000 : -1
    print(String(format: "dist %.2fm: blocks=%d tris/level=%@ sphereVerts=%d rms=%.2fmm max=%.2fmm  integrate=%.1fms mesh=%.1fms",
                 distance, vol.blockCount, "\(tris)", n, rms, maxErr * 1000, tInt, tMesh))
}

run(distance: 1.5)
run(distance: 0.7)
run(distance: 0.35)
run(distance: 0.2)

// MARK: scan bundles

func check(_ ok: Bool, _ what: String) {
    print((ok ? "ok    " : "FAIL  ") + what)
    if !ok { failures += 1 }
}
var failures = 0

// PLY round trip.
do {
    let vol = TSDFVolume()
    var (depth, conf) = render(sphereCenter: SIMD3(0, 0, -0.5))
    depth.withUnsafeBufferPointer { dp in conf.withUnsafeBufferPointer { cp in
        vol.integrate(depth: DepthFrame(depth: dp.baseAddress!, conf: cp.baseAddress!, width: W, height: H,
                                        depthStride: W, confStride: W, fx: fx, fy: fy, cx: cx, cy: cy,
                                        camToWorld: matrix_identity_float4x4), color: nil)
    } }
    let meshes = vol.mesh(blocks: vol.allBlocks)
    let url = FileManager.default.temporaryDirectory.appendingPathComponent("roundtrip.ply")
    try MeshExporter.plyData(meshes).write(to: url)
    let back = try MeshExporter.readPLY(url)
    let V = meshes.reduce(0) { $0 + $1.pos.count }, F = meshes.reduce(0) { $0 + $1.idx.count / 3 }
    let flatPos = meshes.flatMap(\.pos)
    check(back.pos.count == V && back.idx.count == F * 3, "PLY round trip keeps \(V) vertices, \(F) faces")
    check(back.pos == flatPos, "PLY round trip keeps vertex positions exactly")
    let firstMesh = meshes.first { !$0.isEmpty }!
    check(back.idx[0] == firstMesh.idx[0] + UInt32(meshes.prefix { $0.idx != firstMesh.idx }.reduce(0) { $0 + $1.pos.count }),
          "PLY round trip offsets face indices per block")
} catch { check(false, "PLY round trip threw \(error)") }

// Keyframe selection on a synthetic walk.
do {
    func pose(x: Float, yaw: Float) -> simd_float4x4 {
        var t = simd_float4x4(simd_quatf(angle: yaw, axis: SIMD3(0, 1, 0)))
        t.columns.3 = SIMD4(x, 1.5, 0, 1)
        return t
    }
    var sel = KeyframeSelector()
    var taken = 0
    for i in 0..<600 {                     // 10 s at 60 fps, walking 2 m at 0.2 m/s
        let t = Double(i) / 60, p = pose(x: Float(t) * 0.2, yaw: 0)
        if sel.consider(p, time: t) { sel.didCapture(p, time: t); taken += 1 }
    }
    check((8...10).contains(taken), "slow 2 m walk gives one keyframe per ~25 cm (got \(taken))")

    sel.reset(); taken = 0
    for i in 0..<120 {                     // spinning at 3 rad/s: every frame is blurry
        let t = Double(i) / 60, p = pose(x: 0, yaw: Float(t) * 3)
        if sel.consider(p, time: t) { sel.didCapture(p, time: t); taken += 1 }
    }
    check(taken == 0, "fast spin gives no keyframes (got \(taken))")

    sel.reset(); taken = 0
    for i in 0..<600 {                     // slow turn on the spot, 90° in 10 s
        let t = Double(i) / 60, p = pose(x: 0, yaw: Float(t) * .pi / 20)
        if sel.consider(p, time: t) { sel.didCapture(p, time: t); taken += 1 }
    }
    check((6...7).contains(taken), "slow 90° turn gives one keyframe per ~15° (got \(taken))")
}

// Bundle save, list and load.
do {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent("ScansTest-\(UUID().uuidString)")
    ScanBundle.root = root
    defer { try? FileManager.default.removeItem(at: root) }

    let rec = try ScanBundle.makeRecordingDirectory()
    var depth = (0..<(W * H)).map { Float($0 % 4000) / 1000 }
    try depth.withUnsafeMutableBufferPointer {
        try ScanBundle.writeDepth($0.baseAddress!, width: W, height: H, stride: W,
                                  to: rec.appendingPathComponent(ScanBundle.depthPath(0)))
    }
    let kf = Keyframe(id: 0, timestamp: 1, image: ScanBundle.imagePath(0), depth: ScanBundle.depthPath(0),
                      confidence: nil, width: 1024, height: 768, depthWidth: W, depthHeight: H,
                      intrinsics: [800, 800, 512, 384],
                      camToWorld: Keyframe.flatten(simd_float4x4(simd_quatf(angle: 0.3, axis: SIMD3(0, 1, 0)))),
                      sharpness: 12)
    var m = MeshOut()
    m.pos = [SIMD3(0, 0, 0), SIMD3(4, 0, 0), SIMD3(0, 2.5, 3)]
    m.nrm = Array(repeating: SIMD3(0, 0, 1), count: 3)
    m.col = Array(repeating: SIMD4(200, 100, 50, 255), count: 3)
    m.idx = [0, 1, 2]

    ScanBundle.removeStaleRecordings()
    check(!FileManager.default.fileExists(atPath: rec.path), "stale recordings are removed on launch")
    _ = try ScanBundle.makeRecordingDirectory()     // the stale one is gone; record again
    let rec2 = try ScanBundle.makeRecordingDirectory()
    try depth.withUnsafeMutableBufferPointer {
        try ScanBundle.writeDepth($0.baseAddress!, width: W, height: H, stride: W,
                                  to: rec2.appendingPathComponent(ScanBundle.depthPath(0)))
    }
    let date = Date(timeIntervalSince1970: 1_800_000_000)
    let saved = try ScanBundle.finalize(recording: rec2, meshes: [m], keyframes: [kf], created: date)
    let again = try ScanBundle.finalize(recording: try ScanBundle.makeRecordingDirectory(), meshes: [m],
                                        keyframes: [], created: date)

    let listed = ScanBundle.list()
    check(listed.count == 2, "list shows saved scans and hides recordings (got \(listed.count))")
    check(saved.meta.id != again.meta.id, "two scans saved in the same second get different ids")
    let loaded = try ScanBundle.load(saved.url)
    check(loaded.meta == saved.meta, "scan.json round trip")
    check(loaded.meta.boundsMax == [4, 2.5, 3] && loaded.meta.triangleCount == 1, "bounds and triangle count")
    let kfs = try loaded.keyframes()
    check(kfs == [kf], "keyframes.json round trip")
    check(simd_almost_equal_elements(kfs[0].transform, simd_float4x4(simd_quatf(angle: 0.3, axis: SIMD3(0, 1, 0))), 1e-6),
          "keyframe pose round trip")
    let d = try loaded.depth(kfs[0])
    let maxErr = zip(d, depth).map { abs($0 - $1) }.max()!
    check(maxErr < 0.002, String(format: "Float16 depth within 2 mm up to 4 m (max error %.2f mm)", maxErr * 1000))
    check(try MeshExporter.readPLY(loaded.meshURL).pos == m.pos, "bundle mesh readable")
} catch { check(false, "bundle test threw \(error)") }

// MARK: snap to room

/// Synthetic room surfaces as mesh vertices with normals. Room frame (u, v) is rotated by `yaw`
/// and shifted, like a real scan whose walls don't line up with ARKit's axes.
struct RoomBuilder {
    var yaw: Float = 0
    var shift = SIMD2<Float>(0, 0)
    var step: Float = 0.015
    var mesh = MeshOut()

    mutating func add(_ u: Float, _ y: Float, _ v: Float, n: SIMD3<Float>) {
        let c = cos(yaw), s = sin(yaw)
        let noise = SIMD3<Float>(.random(in: -0.004...0.004), .random(in: -0.004...0.004), .random(in: -0.004...0.004))
        mesh.pos.append(SIMD3(u * c - v * s + shift.x, y, u * s + v * c + shift.y) + noise)
        mesh.nrm.append(SIMD3(n.x * c - n.z * s, n.y, n.x * s + n.z * c))
    }
    /// Horizontal patch at height y over a rectangle, skipping `hole`.
    mutating func plane(y: Float, u: ClosedRange<Float>, v: ClosedRange<Float>, up: Bool,
                        hole: (ClosedRange<Float>, ClosedRange<Float>)? = nil) {
        for a in stride(from: u.lowerBound, to: u.upperBound, by: step) {
            for b in stride(from: v.lowerBound, to: v.upperBound, by: step) {
                if let h = hole, h.0.contains(a), h.1.contains(b) { continue }
                add(a, y, b, n: SIMD3(0, up ? 1 : -1, 0))
            }
        }
    }
    /// Vertical wall along u at v (facing ±v) or along v at u (facing ±u).
    mutating func wall(alongU: Bool, at: Float, from: Float, to: Float, y: ClosedRange<Float>, facing: Float) {
        for a in stride(from: from, to: to, by: step) {
            for h in stride(from: y.lowerBound, to: y.upperBound, by: step) {
                if alongU { add(a, h, at, n: SIMD3(0, 0, facing)) } else { add(at, h, a, n: SIMD3(facing, 0, 0)) }
            }
        }
    }
}

func near(_ p: [Float], _ q: SIMD2<Float>, _ tol: Float) -> Bool { simd_distance(SIMD2(p[0], p[1]), q) < tol }

do {
    // L-shaped room, 5 × 4 m minus a 1.5 × 1.5 m corner, 2.5 m high, turned 20°.
    // The wall at v = 0 is only scanned for its first 3 m; a table and a cabinet stand inside.
    var r = RoomBuilder(yaw: 20 * .pi / 180, shift: SIMD2(0.7, -1.3))
    let f: Float = -1.4, top: Float = 1.1
    let notch = (Float(3.5)...5, Float(2.5)...4)
    r.plane(y: f, u: 0...5, v: 0...4, up: true, hole: notch)
    r.plane(y: top, u: 0...5, v: 0...4, up: false, hole: notch)
    r.wall(alongU: true, at: 0, from: 0, to: 3, y: f...top, facing: 1)
    r.wall(alongU: false, at: 5, from: 0, to: 2.5, y: f...top, facing: -1)
    r.wall(alongU: true, at: 2.5, from: 3.5, to: 5, y: f...top, facing: -1)
    r.wall(alongU: false, at: 3.5, from: 2.5, to: 4, y: f...top, facing: -1)
    r.wall(alongU: true, at: 4, from: 0, to: 3.5, y: f...top, facing: -1)
    r.wall(alongU: false, at: 0, from: 0, to: 4, y: f...top, facing: 1)
    r.plane(y: f + 0.75, u: 1...2, v: 1...1.8, up: true)                       // table top
    r.wall(alongU: false, at: 4.5, from: 0.5, to: 1.5, y: f...(f + 0.9), facing: -1)   // cabinet front
    r.plane(y: f + 0.9, u: 4.5...5, v: 0.5...1.5, up: true)
    r.mesh.idx = [0, 1, 2]

    let room = try RoomFitter.fit([r.mesh])
    print("  " + room.notes)
    check(abs(room.floorY - f) < 0.02, String(format: "floor height (%.3f, want %.3f)", room.floorY, f))
    check(room.ceilingObserved && abs(room.ceilingY - top) < 0.02, String(format: "ceiling height (%.3f)", room.ceilingY))
    check(abs(room.yaw * 180 / .pi - 20) < 1, String(format: "room direction (%.2f°, want 20°)", room.yaw * 180 / .pi))
    check(room.outline.count == 6, "L-shaped outline has 6 corners (got \(room.outline.count))")
    let c = cos(r.yaw), s = sin(r.yaw)
    let truth: [SIMD2<Float>] = [(0, 0), (5, 0), (5, 2.5), (3.5, 2.5), (3.5, 4), (0, 4)].map { (u: Float, v: Float) in
        SIMD2(u * c - v * s, u * s + v * c) + r.shift
    }
    let matched = truth.filter { t in room.outline.contains { near($0, t, 0.05) } }.count
    check(matched == 6, "every true corner within 5 cm of a fitted corner (\(matched)/6)")
    check(abs(room.area - 17.75) < 0.4, String(format: "floor area (%.2f m², want 17.75)", room.area))
    let partial = room.walls.filter { $0.coverage < 0.9 }
    check(partial.count == 1 && abs(partial[0].coverage - 0.6) < 0.1 && abs(partial[0].length - 5) < 0.05,
          "only the half-scanned wall is marked partial (\(room.walls.map { String(format: "%.2f", $0.coverage) }))")
    func contains(_ poly: [[Float]], _ p: SIMD2<Float>) -> Bool {
        var inside = false
        for i in 0..<poly.count {
            let a = poly[i], b = poly[(i + 1) % poly.count]
            if (a[1] > p.y) != (b[1] > p.y), p.x < (b[0] - a[0]) * (p.y - a[1]) / (b[1] - a[1]) + a[0] { inside.toggle() }
        }
        return inside
    }
    let inward = room.walls.allSatisfy { w in
        let mid = SIMD2((w.start[0] + w.end[0]) / 2, (w.start[1] + w.end[1]) / 2), n = SIMD2(w.normal[0], w.normal[1])
        return contains(room.outline, mid + n * 0.2) && !contains(room.outline, mid - n * 0.2)
    }
    check(inward, "wall normals point into the room")

    // Rectangle with one wall never scanned and no ceiling: the open side follows the floor.
    var q = RoomBuilder(yaw: -0.5)
    q.plane(y: 0, u: 0...4, v: 0...3, up: true)
    q.wall(alongU: true, at: 0, from: 0, to: 4, y: 0...2.6, facing: 1)
    q.wall(alongU: true, at: 3, from: 0, to: 4, y: 0...2.6, facing: -1)
    q.wall(alongU: false, at: 0, from: 0, to: 3, y: 0...2.6, facing: 1)
    q.mesh.idx = [0, 1, 2]
    let open = try RoomFitter.fit([q.mesh])
    print("  " + open.notes)
    check(open.outline.count == 4, "open-sided room still gets 4 corners (\(open.outline.count))")
    check(!open.ceilingObserved && abs(open.ceilingY - 2.6) < 0.05,
          String(format: "no ceiling: height from wall tops (%.2f)", open.ceilingY))
    check(abs(open.area - 12) < 0.4, String(format: "open-sided area (%.2f m², want 12)", open.area))
    check(open.walls.filter { $0.coverage < 0.1 }.count == 1, "the unscanned wall has no coverage")
    check(abs(open.extent.max()! - 4) < 0.08 && abs(open.extent.min()! - 3) < 0.08, "extent is 4 × 3 m (\(open.extent))")

    // Floor mostly hidden under a bed: only a 30 × 30 cm patch of floor was scanned.
    var b = RoomBuilder(yaw: 0.1)
    b.plane(y: 0, u: 3...3.3, v: 0.5...0.8, up: true)
    b.plane(y: 0.55, u: 0.5...2.5, v: 0.5...2.1, up: true)                  // bed
    b.plane(y: 2.5, u: 0...4, v: 0...3, up: false)
    b.wall(alongU: true, at: 0, from: 0, to: 4, y: 0...2.5, facing: 1)
    b.wall(alongU: true, at: 3, from: 0, to: 4, y: 0...2.5, facing: -1)
    b.wall(alongU: false, at: 0, from: 0, to: 3, y: 0...2.5, facing: 1)
    b.wall(alongU: false, at: 4, from: 0, to: 3, y: 0...2.5, facing: -1)
    b.mesh.idx = [0, 1, 2]
    let hidden = try RoomFitter.fit([b.mesh])
    check(abs(hidden.floorY) < 0.03, String(format: "floor under a bed is found from the wall bottoms (%.2f)", hidden.floorY))

    // Objects: a cabinet against a wall, a table with legs, a picture on a wall.
    var o = RoomBuilder(yaw: 0.3, shift: SIMD2(1, 2))
    o.plane(y: 0, u: 0...4, v: 0...3, up: true)
    o.plane(y: 2.5, u: 0...4, v: 0...3, up: false)
    o.wall(alongU: true, at: 0, from: 0, to: 4, y: 0...2.5, facing: 1)
    o.wall(alongU: true, at: 3, from: 0, to: 4, y: 0...2.5, facing: -1)
    o.wall(alongU: false, at: 0, from: 0, to: 3, y: 0...2.5, facing: 1)
    o.wall(alongU: false, at: 4, from: 0, to: 3, y: 0...2.5, facing: -1)
    o.wall(alongU: false, at: 3.4, from: 0.5, to: 1.5, y: 0...0.9, facing: -1)          // cabinet front
    o.wall(alongU: true, at: 0.5, from: 3.4, to: 4, y: 0...0.9, facing: -1)             // cabinet sides
    o.wall(alongU: true, at: 1.5, from: 3.4, to: 4, y: 0...0.9, facing: 1)
    o.plane(y: 0.9, u: 3.4...4, v: 0.5...1.5, up: true)                                 // cabinet top
    o.plane(y: 0.75, u: 1...2, v: 1...1.8, up: true)                                    // table top
    for (lu, lv) in [(Float(1.05), Float(1.05)), (1.95, 1.05), (1.95, 1.75), (1.05, 1.75)] {
        o.wall(alongU: true, at: lv, from: lu - 0.02, to: lu + 0.02, y: 0...0.75, facing: 1)   // legs
    }
    o.wall(alongU: true, at: 2.98, from: 1.5, to: 2.2, y: 1.4...1.9, facing: -1)          // picture on the wall
    o.mesh.idx = [0, 1, 2]
    let oRoom = try RoomFitter.fit([o.mesh])
    let objs = ObjectFinder.find(RoomFitter.sample([o.mesh], cell: 0.03), shell: oRoom)
    print("  objects: " + objs.map { String(format: "%.2f×%.2f×%.2f", $0.size[0], $0.size[1], $0.size[2]) }.joined(separator: ", "))
    check(objs.count == 2, "finds the cabinet and the table, not the picture (\(objs.count) boxes)")
    if objs.count == 2 {
        let cab = objs.first { abs($0.size[1] - 0.9) < 0.05 }, table = objs.first { abs($0.size[1] - 0.75) < 0.05 }
        check(cab != nil && abs(cab!.size[0] - 0.6) < 0.08 && abs(cab!.size[2] - 1.0) < 0.08, "cabinet box is 0.6 × 0.9 × 1.0 m")
        check(table != nil && abs(table!.size[0] - 1.0) < 0.08 && abs(table!.size[2] - 0.8) < 0.08
              && abs(table!.center[1] - 0.375) < 0.03, "table box is 1.0 × 0.75 × 0.8 m and stands on the floor")
    }
    var tracker = BoxTracker()
    var first = tracker.update(objs)
    first[0].label = "cabinet"
    tracker = BoxTracker(seed: first)
    var moved = objs.map { b -> ObjectBox in var b = b; b.center[0] += 0.05; b.size[1] += 0.03; b.id = 99; return b }
    moved.append(ObjectBox(id: 99, center: [5, 1, 5], size: [0.3, 0.3, 0.3], yaw: 0, points: 50))
    let second = tracker.update(moved)
    check(Set(second.prefix(2).map(\.id)) == Set(first.map(\.id)) && second[2].id > first.map(\.id).max()!,
          "tracker keeps ids of moved boxes and gives new ones a new id")
    check(second.first { $0.id == first[0].id }?.label == "cabinet", "tracker carries labels over")
    let third = tracker.update([])
    check(third.count == 3, "a box missing for one update is kept for a while")

    // Window wall: the real wall surface at v = 0.25, window glass and recesses at v = 0 behind it
    // (seen only in a few places). The wall line goes on the real surface, not the glass.
    var w = RoomBuilder(yaw: 0.2)
    w.plane(y: 0, u: 0...6, v: 0...4, up: true)
    w.plane(y: 2.5, u: 0...6, v: 0.25...4, up: false)
    w.wall(alongU: true, at: 0.25, from: 0, to: 6, y: 0...2.5, facing: 1)
    for (a, b) in [(Float(0.5), Float(1.5)), (2.5, 3.5), (4.5, 5.5)] {
        w.wall(alongU: true, at: 0, from: a, to: b, y: 0.9...2.1, facing: 1)                // glass
        w.plane(y: 0.9, u: a...b, v: 0...0.25, up: true)                                    // sill
    }
    w.wall(alongU: true, at: 4, from: 0, to: 6, y: 0...2.5, facing: -1)
    w.wall(alongU: false, at: 0, from: 0.25, to: 4, y: 0...2.5, facing: 1)
    w.wall(alongU: false, at: 6, from: 0.25, to: 4, y: 0...2.5, facing: -1)
    w.mesh.idx = [0, 1, 2]
    let wRoom = try RoomFitter.fit([w.mesh])
    let wObjs = ObjectFinder.find(RoomFitter.sample([w.mesh], cell: 0.03), shell: wRoom)
    print("  window room: " + wRoom.notes + String(format: ", %.2f × %.2f m, %d boxes", wRoom.extent[0], wRoom.extent[1], wObjs.count))
    check(abs(wRoom.extent.min()! - 3.75) < 0.06, String(format: "the window wall is on the wall surface, not the glass (%.2f m deep, want 3.75)", wRoom.extent.min()!))
    check(wObjs.isEmpty, "window sills and glass behind the wall are not furniture (\(wObjs.count) boxes)")

    // Desk island: two desks back to back with a divider between them, no legs scanned, and a
    // chair touching the front of one. Found as one cluster; split into two desks and a chair.
    var d = RoomBuilder(yaw: 0.3)
    d.plane(y: 0, u: 0...6, v: 0...5, up: true)
    d.plane(y: 2.5, u: 0...6, v: 0...5, up: false)
    d.wall(alongU: true, at: 0, from: 0, to: 6, y: 0...2.5, facing: 1)
    d.wall(alongU: true, at: 5, from: 0, to: 6, y: 0...2.5, facing: -1)
    d.wall(alongU: false, at: 0, from: 0, to: 5, y: 0...2.5, facing: 1)
    d.wall(alongU: false, at: 6, from: 0, to: 5, y: 0...2.5, facing: -1)
    d.plane(y: 0.74, u: 1.5...3.5, v: 1.5...2.3, up: true)                                // desk A
    d.plane(y: 0.74, u: 1.5...3.5, v: 2.36...3.2, up: true)                               // desk B
    d.wall(alongU: true, at: 2.33, from: 1.5, to: 3.5, y: 0.7...1.2, facing: 1)           // divider
    d.plane(y: 0.45, u: 2.2...2.65, v: 1.0...1.5, up: true)                               // chair seat
    d.wall(alongU: true, at: 1.0, from: 2.2, to: 2.65, y: 0.45...0.95, facing: 1)        // chair back
    for (lu, lv) in [(Float(2.22), Float(1.02)), (2.63, 1.02), (2.63, 1.48), (2.22, 1.48)] {
        d.wall(alongU: true, at: lv, from: lu - 0.02, to: lu + 0.02, y: 0...0.45, facing: 1)  // chair legs
    }
    d.mesh.idx = [0, 1, 2]
    let dRoom = try RoomFitter.fit([d.mesh])
    let dObjs = ObjectFinder.find(RoomFitter.sample([d.mesh], cell: 0.03), shell: dRoom)
    let dBounds = dObjs.map { RoomCorrector.roomBounds($0, shell: dRoom) }
    print("  desk island: " + dBounds.map { String(format: "%.2f×%.2f×%.2f", $0.hi.x - $0.lo.x, $0.hi.z - $0.lo.z, $0.hi.y - $0.lo.y) }.joined(separator: ", "))
    let desks = dBounds.filter { abs($0.hi.x - $0.lo.x - 2.0) < 0.1 && $0.hi.z - $0.lo.z < 1.0 }
    check(dObjs.count == 3 && desks.count == 2, "a desk island is split into two desks and a chair (\(dObjs.count) boxes)")
    check(desks.allSatisfy { abs($0.lo.y - dRoom.floorY) < 0.01 }, "desks seen without legs stand on the floor")
    check(dBounds.contains { abs($0.hi.x - $0.lo.x - 0.45) < 0.08 && abs($0.hi.y - $0.lo.y - 0.95) < 0.06 }, "the chair is its own box")

    // Kitchen: cabinets up to 2.4 m along one wall with a sofa pushed against their front, and a
    // lowered ceiling in one corner. The cabinets and the sofa are two boxes; the ceiling is none.
    var k = RoomBuilder(yaw: -0.2)
    k.plane(y: 0, u: 0...6, v: 0...5, up: true)
    k.plane(y: 2.5, u: 0...4, v: 0...5, up: false)
    k.plane(y: 2.5, u: 4...6, v: 0...3, up: false)
    k.wall(alongU: true, at: 0, from: 0, to: 6, y: 0...2.5, facing: 1)
    k.wall(alongU: true, at: 5, from: 0, to: 6, y: 0...2.5, facing: -1)
    k.wall(alongU: false, at: 0, from: 0, to: 5, y: 0...2.5, facing: 1)
    k.wall(alongU: false, at: 6, from: 0, to: 5, y: 0...2.5, facing: -1)
    k.wall(alongU: true, at: 0.6, from: 0.5, to: 3.5, y: 0...2.4, facing: 1)             // cabinet fronts
    k.wall(alongU: false, at: 0.5, from: 0, to: 0.6, y: 0...2.4, facing: -1)            // cabinet ends
    k.wall(alongU: false, at: 3.5, from: 0, to: 0.6, y: 0...2.4, facing: 1)
    k.plane(y: 0.45, u: 1.5...3.0, v: 0.62...1.5, up: true)                               // sofa seat
    k.wall(alongU: true, at: 1.5, from: 1.5, to: 3.0, y: 0...0.45, facing: 1)            // sofa front
    k.wall(alongU: false, at: 1.5, from: 0.62, to: 1.5, y: 0...0.7, facing: -1)          // sofa arms
    k.wall(alongU: false, at: 3.0, from: 0.62, to: 1.5, y: 0...0.7, facing: 1)
    k.plane(y: 2.2, u: 4...6, v: 3...5, up: false)                                         // lowered ceiling
    k.wall(alongU: true, at: 3, from: 4, to: 6, y: 2.2...2.5, facing: -1)
    k.wall(alongU: false, at: 4, from: 3, to: 5, y: 2.2...2.5, facing: -1)
    k.mesh.idx = [0, 1, 2]
    let kRoom = try RoomFitter.fit([k.mesh])
    let kObjs = ObjectFinder.find(RoomFitter.sample([k.mesh], cell: 0.03), shell: kRoom)
    let kBounds = kObjs.map { RoomCorrector.roomBounds($0, shell: kRoom) }
    print("  kitchen: " + kBounds.map { String(format: "%.2f×%.2f×%.2f at (%.2f, %.2f)", $0.hi.x - $0.lo.x, $0.hi.z - $0.lo.z, $0.hi.y - $0.lo.y, $0.lo.x, $0.lo.z) }.joined(separator: ", "))
    check(kObjs.count == 2, "cabinets along the wall and the sofa in front are two boxes, the lowered ceiling none (\(kObjs.count) boxes)")
    check(kBounds.contains { abs($0.hi.x - $0.lo.x - 3.0) < 0.1 && abs($0.hi.z - $0.lo.z - 0.6) < 0.1 && abs($0.hi.y - $0.lo.y - 2.4) < 0.06 },
          "the cabinet run is its own 3.0 × 0.6 × 2.4 m box")
    check(kBounds.contains { abs($0.hi.x - $0.lo.x - 1.5) < 0.1 && $0.hi.y - $0.lo.y < 0.8 }, "the sofa is its own low box")

    // Drift on a long walk: the far half of the floor sits 12 cm higher and its ceiling is 40 cm
    // lower (a hallway), with a tall cabinet up to that ceiling. Only the cabinet is a thing.
    var dr = RoomBuilder(yaw: 0.1)
    dr.plane(y: 0, u: 0...4, v: 0...4, up: true)
    dr.plane(y: 0.12, u: 4...8, v: 0...4, up: true)
    dr.plane(y: 2.5, u: 0...4, v: 0...4, up: false)
    dr.plane(y: 2.1, u: 4...8, v: 0...4, up: false)
    dr.wall(alongU: true, at: 0, from: 0, to: 8, y: 0...2.5, facing: 1)
    dr.wall(alongU: true, at: 4, from: 0, to: 8, y: 0...2.5, facing: -1)
    dr.wall(alongU: false, at: 0, from: 0, to: 4, y: 0...2.5, facing: 1)
    dr.wall(alongU: false, at: 8, from: 0, to: 4, y: 0...2.1, facing: -1)
    dr.wall(alongU: false, at: 5, from: 1.5, to: 2.5, y: 0.12...2.08, facing: -1)        // cabinet
    dr.wall(alongU: false, at: 5.6, from: 1.5, to: 2.5, y: 0.12...2.08, facing: 1)
    dr.wall(alongU: true, at: 1.5, from: 5, to: 5.6, y: 0.12...2.08, facing: -1)
    dr.wall(alongU: true, at: 2.5, from: 5, to: 5.6, y: 0.12...2.08, facing: 1)
    dr.mesh.idx = [0, 1, 2]
    let fRoom = try RoomFitter.fit([dr.mesh])
    let fObjs = ObjectFinder.find(RoomFitter.sample([dr.mesh], cell: 0.03), shell: fRoom)
    let fB = fObjs.map { RoomCorrector.roomBounds($0, shell: fRoom) }
    print("  drift: " + fB.map { String(format: "%.2f×%.2f×%.2f", $0.hi.x - $0.lo.x, $0.hi.z - $0.lo.z, $0.hi.y - $0.lo.y) }.joined(separator: ", "))
    check(fB.count == 1 && fB.allSatisfy { $0.hi.x - $0.lo.x < 0.8 && $0.hi.z - $0.lo.z < 1.2 },
          "a floor or ceiling a little off the fitted level is not furniture (\(fObjs.count) boxes)")

    // room.json round trip.
    let dir = try ScanBundle.makeRecordingDirectory()
    let bundle = ScanBundle(url: dir, meta: ScanMeta(id: "x", version: 1, created: Date(), keyframeCount: 0,
                                                     triangleCount: 0, boundsMin: [], boundsMax: [], coverKeyframe: nil))
    try bundle.saveRoom(RoomModel(shell: room))
    check(bundle.loadRoom()?.shell == room, "room.json round trip")
} catch { check(false, "room fit threw \(error)") }

// MARK: volume save and load
do {
    let vol = TSDFVolume()
    let (depth, conf) = render(sphereCenter: SIMD3(0.1, 0, -0.8))
    depth.withUnsafeBufferPointer { dp in conf.withUnsafeBufferPointer { cp in
        vol.integrate(depth: DepthFrame(depth: dp.baseAddress!, conf: cp.baseAddress!, width: W, height: H,
                                        depthStride: W, confStride: W, fx: fx, fy: fy, cx: cx, cy: cy,
                                        camToWorld: matrix_identity_float4x4), color: nil)
    } }
    let url = FileManager.default.temporaryDirectory.appendingPathComponent("vol-\(UUID().uuidString).tsdf")
    try vol.write(to: url)
    let size = (try FileManager.default.attributesOfItem(atPath: url.path)[.size] as? Int) ?? 0
    let back = TSDFVolume()
    try back.read(from: url)
    let a = vol.mesh(blocks: vol.allBlocks).flatMap(\.pos), b = back.mesh(blocks: back.allBlocks).flatMap(\.pos)
    check(back.blockCount == vol.blockCount && a == b,
          "volume save and load gives the same mesh (\(vol.blockCount) blocks, \(size / 1024) KB on disk)")
    check(back.takeDirty(budget: 100_000, camera: .zero).count == back.blockCount, "loaded blocks are queued for meshing")
    try? FileManager.default.removeItem(at: url)
} catch { check(false, "volume save threw \(error)") }

// MARK: GPU fusion matches the CPU
do {
    let cpu = TSDFVolume(useGPU: false), gpu = TSDFVolume(useGPU: true)
    check(gpu.usesGPU, "GPU fusion is available on this Mac")
    // A few views of a sphere, with color, moving the camera a little each time.
    let iw = 512, ih = 384
    var yPlane = [UInt8](repeating: 0, count: iw * ih), cPlane = [UInt8](repeating: 0, count: iw * ih / 2)
    for i in 0..<yPlane.count { yPlane[i] = UInt8((i * 7) % 251) }
    for i in 0..<cPlane.count { cPlane[i] = UInt8((i * 13) % 253) }
    var tc: Double = 0, tg: Double = 0
    for k in 0..<6 {
        let (depth, conf) = render(sphereCenter: SIMD3(0.02 * Float(k), 0, -0.6 - 0.15 * Float(k)))
        let pose = simd_float4x4(simd_quatf(angle: 0.02 * Float(k), axis: SIMD3(0, 1, 0)))
        depth.withUnsafeBufferPointer { dp in conf.withUnsafeBufferPointer { cp in
            yPlane.withUnsafeBufferPointer { yp in cPlane.withUnsafeBufferPointer { cbp in
                let f = DepthFrame(depth: dp.baseAddress!, conf: cp.baseAddress!, width: W, height: H, depthStride: W,
                                   confStride: W, fx: fx, fy: fy, cx: cx, cy: cy, camToWorld: pose)
                let c = ColorFrame(y: yp.baseAddress!, cbcr: cbp.baseAddress!, yStride: iw, cStride: iw, width: iw, height: ih,
                                   fx: fx * 2, fy: fy * 2, cx: cx * 2, cy: cy * 2)
                var t0 = Date(); cpu.integrate(depth: f, color: c); tc += Date().timeIntervalSince(t0)
                t0 = Date(); gpu.integrate(depth: f, color: c); tg += Date().timeIntervalSince(t0)
            } }
        } }
    }
    let mc = cpu.mesh(blocks: cpu.allBlocks), mg = gpu.mesh(blocks: gpu.allBlocks)
    // Block order differs between volumes (set order), so compare vertices sorted by position.
    func sortedVerts(_ ms: [MeshOut]) -> [(SIMD3<Float>, SIMD4<UInt8>)] {
        ms.flatMap { m in zip(m.pos, m.col).map { ($0, $1) } }
            .sorted { ($0.0.x, $0.0.y, $0.0.z) < ($1.0.x, $1.0.y, $1.0.z) }
    }
    let vc = sortedVerts(mc), vg = sortedVerts(mg)
    let pc = vc.map(\.0), pg = vg.map(\.0)
    let maxDiff = zip(pc, pg).map { simd_distance($0, $1) }.max() ?? 0
    let colDiff = zip(vc.map(\.1), vg.map(\.1)).map { abs(Int($0.x) - Int($1.x)) }.max() ?? 0
    check(cpu.blockCount == gpu.blockCount && pc.count == pg.count && maxDiff < 0.0005 && colDiff <= 2,
          String(format: "GPU fusion matches CPU (%d vs %d vertices, max %.3f mm, color ±%d); CPU %.1f ms, GPU %.1f ms per frame",
                 pc.count, pg.count, maxDiff * 1000, colDiff, tc / 6 * 1000, tg / 6 * 1000))
}

// MARK: room edits

do {
    // 4 × 3 m room turned 0.3 rad and shifted; walls at u = 0, u = 4, v = 0, v = 3.
    let yaw: Float = 0.3, c = cos(yaw), s = sin(yaw)
    func w(_ u: Float, _ v: Float) -> [Float] { [u * c - v * s + 1, u * s + v * c + 2] }
    func wn(_ u: Float, _ v: Float) -> [Float] { [u * c - v * s, u * s + v * c] }
    let corners: [(Float, Float)] = [(0, 0), (4, 0), (4, 3), (0, 3)]
    let normals: [(Float, Float)] = [(0, 1), (-1, 0), (0, -1), (1, 0)]
    var walls: [RoomShell.Wall] = []
    for i in 0..<4 {
        let a = corners[i], b = corners[(i + 1) % 4]
        walls.append(.init(id: i, start: w(a.0, a.1), end: w(b.0, b.1), normal: wn(normals[i].0, normals[i].1),
                           length: simd_distance(SIMD2(a.0, a.1), SIMD2(b.0, b.1)), coverage: 1))
    }
    let shell = RoomShell(floorY: 0, ceilingY: 2.5, ceilingObserved: true, yaw: yaw,
                          outline: corners.map { w($0.0, $0.1) }, walls: walls, area: 12, extent: [4, 3], notes: "")
    func room(_ u: Float, _ v: Float) -> SIMD2<Float> {          // world xz → room uv
        let x = u - 1, z = v - 2
        return SIMD2(x * c + z * s, -x * s + z * c)
    }
    let sofaW = w(2, 1.5)
    let sofa = ObjectBox(id: 1, center: [sofaW[0], 0.4, sofaW[1]], size: [2, 0.8, 0.9], yaw: yaw, points: 500, label: "grey sofa")
    let lampW = w(3.6, 2.6)
    let lamp = ObjectBox(id: 2, center: [lampW[0], 0.8, lampW[1]], size: [0.3, 1.6, 0.3], yaw: yaw, points: 100)
    var ed = RoomEditor(room: RoomModel(shell: shell, objects: [sofa, lamp]))

    // Drag the sofa to 5 cm from the v = 0 wall: it snaps flush (center v = 0.45).
    ed.beginMove()
    let target = w(2, 0.5)
    ed.move(1, to: SIMD2(target[0], target[1]))
    var q = room(ed.box(1)!.center[0], ed.box(1)!.center[2])
    check(abs(q.y - 0.45) < 0.002 && abs(q.x - 2) < 0.002, String(format: "sofa snaps flush to the wall (v %.3f)", q.y))
    // Pushed through the wall it comes back inside.
    let through = w(2, 0.2)
    ed.move(1, to: SIMD2(through[0], through[1]))
    q = room(ed.box(1)!.center[0], ed.box(1)!.center[2])
    check(abs(q.y - 0.45) < 0.002, "a box pushed into a wall is put back against it")
    // Far from walls it stays where it is put.
    let free = w(2, 1.4)
    ed.move(1, to: SIMD2(free[0], free[1]))
    q = room(ed.box(1)!.center[0], ed.box(1)!.center[2])
    check(abs(q.y - 1.4) < 0.002, "no snap away from walls")
    check(ed.box(1)!.center[1] == 0.4, "moving keeps the height")

    // Flush against the v = 0 wall, then a quarter turn: the 2 m side now runs along v and
    // would poke through the wall, so it is pushed back in, flush.
    let nearWall = w(2, 0.45)
    ed.beginMove(); ed.move(1, to: SIMD2(nearWall[0], nearWall[1]))
    ed.rotate(1)
    let corners1 = ed.box(1)!.corners.map { room($0.x, $0.z) }
    let minV = corners1.map(\.y).min()!, spanV = corners1.map(\.y).max()! - minV
    check(abs(minV) < 0.003 && abs(spanV - 2) < 0.003,
          String(format: "rotated sofa turns 90° and stays inside, flush (min v %.3f, length along v %.2f)", minV, spanV))

    // Merge sofa and lamp, then undo and redo.
    ed.undo(); ed.undo()                                       // back to free position, unturned
    let before = ed.objects
    let merged = ed.merge(1, 2)
    let mb = ed.box(merged ?? -1)
    let mc = mb?.corners.map { room($0.x, $0.z) } ?? []
    check(ed.objects.count == 1 && merged == 1 && mb?.label == "grey sofa" && mb?.points == 600
          && abs((mc.map(\.x).max() ?? 0) - 3.75) < 0.003 && abs((mb?.size[1] ?? 0) - 1.6) < 0.003,
          "merge makes one box around both, keeping the bigger one's id and name")
    ed.undo()
    check(ed.objects == before, "undo restores both boxes")
    ed.redo()
    check(ed.objects.count == 1, "redo merges again")
    ed.undo()

    ed.rename(2, to: "  floor lamp ")
    check(ed.box(2)?.label == "floor lamp" && ed.box(2)?.source == "user", "rename trims and marks the name as yours")
    ed.delete(2)
    check(ed.box(2) == nil && ed.canUndo, "delete removes the box")
    ed.undo()
    check(ed.box(2)?.label == "floor lamp", "undo brings a deleted box back")

    // Split: a sofa (2.0 × 0.9 m, 0.85 tall) with a coffee table (1.0 × 0.6 m, 0.45 tall)
    // pushed against its front, found as one box. Points on their tops and sides.
    do {
        let yaw: Float = 0.4, c = cos(yaw), s = sin(yaw)
        func w(_ u: Float, _ y: Float, _ v: Float) -> SIMD3<Float> { SIMD3(u * c - v * s, y, u * s + v * c) }
        var pts: [SIMD3<Float>] = []
        func block(_ u0: Float, _ u1: Float, _ v0: Float, _ v1: Float, _ top: Float) {
            for u in stride(from: u0, through: u1, by: 0.03) {
                for v in stride(from: v0, through: v1, by: 0.03) { pts.append(w(u, top, v)) }
                for y in stride(from: 0.06, through: top, by: 0.03) { pts.append(w(u, y, v1)); pts.append(w(u, y, v0)) }
            }
            for v in stride(from: v0, through: v1, by: 0.03) {
                for y in stride(from: 0.06, through: top, by: 0.03) { pts.append(w(u0, y, v)); pts.append(w(u1, y, v)) }
            }
        }
        block(0, 2.0, 0, 0.9, 0.85)          // sofa
        block(0.5, 1.5, 0.9, 1.5, 0.45)      // coffee table, touching
        let mid = w(1.0, 0.425, 0.75)
        let both = ObjectBox(id: 7, center: [mid.x, 0.425, mid.z], size: [2.0, 0.85, 1.5], yaw: yaw, points: pts.count,
                             label: "sofa", confidence: 0.8, source: "model")
        var se = RoomEditor(room: RoomModel(shell: RoomShell(floorY: 0, ceilingY: 2.5, ceilingObserved: true, yaw: yaw,
                                                             outline: [], walls: [], area: 20, extent: [5, 4], notes: ""),
                                            objects: [both]))
        let ids = se.split(7, points: pts)
        let parts = se.objects.map { String(format: "%.2f×%.2f×%.2f", $0.size[0], $0.size[2], $0.size[1]) }
        print("  split: " + parts.joined(separator: ", "))
        let sofaPart = se.box(7), tablePart = ids.flatMap { se.box($0.1) }
        check(sofaPart != nil && abs(sofaPart!.size[0] - 2.0) < 0.05 && abs(sofaPart!.size[2] - 0.9) < 0.06 && sofaPart!.label == "sofa",
              "split keeps the sofa (2.0 × 0.9 m) with its name")
        check(tablePart != nil && abs(tablePart!.size[0] - 1.0) < 0.05 && abs(tablePart!.size[2] - 0.6) < 0.06
              && abs(tablePart!.size[1] - 0.45) < 0.04 && tablePart!.label == nil && abs(tablePart!.center[1] - 0.225) < 0.03,
              "split makes the coffee table its own unnamed box on the floor")
        se.undo()
        check(se.objects == [both], "undo of a split puts the one box back")
        check(se.split(7, points: []) == nil && se.objects == [both], "no points: nothing to split")
        pts = []
        block(0, 2.0, 0, 1.5, 0.75)          // one solid block: no place to cut
        check(se.split(7, points: pts) == nil, "a box that is one solid thing is not split")
    }

    // Best photos of a box: in front of the camera, from different sides, not hidden.
    do {
        func kf(_ id: Int, at p: SIMD3<Float>, turn: Float) -> Keyframe {
            // Camera looks down its -z; `turn` rotates it about y.
            let c = cos(turn), s = sin(turn)
            let t = simd_float4x4(columns: (SIMD4(c, 0, -s, 0), SIMD4(0, 1, 0, 0), SIMD4(s, 0, c, 0), SIMD4(p, 1)))
            return Keyframe(id: id, timestamp: 0, image: "", depth: "d\(id)", confidence: nil, width: 1024, height: 768,
                            depthWidth: 4, depthHeight: 3, intrinsics: [700, 700, 512, 384],
                            camToWorld: Keyframe.flatten(t), sharpness: 150)
        }
        let box = ObjectBox(id: 1, center: [0, 0.5, 0], size: [1, 1, 1], yaw: 0, points: 100)
        let front = kf(0, at: SIMD3(0, 1, 3), turn: 0)
        let away = kf(1, at: SIMD3(0, 1, 3), turn: .pi)
        let nearlySame = kf(2, at: SIMD3(0.4, 1, 3), turn: 0.13)
        let side = kf(3, at: SIMD3(3, 1, 0), turn: .pi / 2)
        let hidden = kf(4, at: SIMD3(-3, 1, 0), turn: -.pi / 2)
        let views = BoxViews.best(box, keyframes: [away, nearlySame, front, side, hidden]) { k in
            [Float](repeating: k.id == 4 ? 1.0 : 3.0, count: 12)          // something 1 m in front of camera 4
        }
        let ids = views.map(\.keyframe.id)
        print("  box photos: \(ids)")
        check(BoxViews.project(box, away) == nil, "a photo facing away doesn't show the box")
        check(ids.count == 2 && ids.contains(3) && (ids.contains(0) || ids.contains(2)) && !ids.contains(4),
              "best photos come from different sides and skip one where something is in front")
        if let v = views.first(where: { $0.keyframe.id == 0 }) {
            check(abs(v.rect.midX - 512) < 5, "the box is in the middle of the photo facing it")
        }
    }

    // A name given by hand survives a re-fit.
    var tr = BoxTracker(seed: ed.objects)
    let refound = tr.update([sofa, lamp].map { var b = $0; b.label = nil; b.source = nil; b.id = 0; return b })
    check(refound.first { $0.label == "floor lamp" }?.source == "user", "hand names carry over when the room is re-fitted")

    // Categories from names.
    let cats: [(String?, BoxCategory)] = [("desk lamp", .lamp), ("Coffee Table", .table), ("grey three-seat sofa", .sofa),
                                          ("Bookshelves", .shelf), ("tv stand", .cabinet), ("refrigerator", .appliance),
                                          ("chairs", .chair), ("something", .other), (nil, .other)]
    let wrong = cats.filter { BoxCategory.of($0.0) != $0.1 }
    check(wrong.isEmpty, "categories from names" + (wrong.isEmpty ? "" : " (wrong: \(wrong.map { $0.0 ?? "nil" }))"))

    // Floor of an L-shaped room triangulates into the right area.
    let L: [SIMD2<Float>] = [SIMD2(0, 0), SIMD2(5, 0), SIMD2(5, 4), SIMD2(3.5, 4), SIMD2(3.5, 2.5), SIMD2(0, 2.5)]
    let tri = PolygonTriangulator.triangulate(L)
    var area: Float = 0
    for t in stride(from: 0, to: tri.count, by: 3) {
        let a = L[tri[t]], b = L[tri[t + 1]], c = L[tri[t + 2]]
        area += abs((b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)) / 2
    }
    check(tri.count == 12 && abs(area - (20 - 1.5 * 1.5 * 0 - 3.5 * 1.5)) < 0.001,
          String(format: "L-shaped floor triangulates (%d triangles, %.2f m²)", tri.count / 3, area))

    // USDZ export.
    let usdz = FileManager.default.temporaryDirectory.appendingPathComponent("room-test.usdz")
    try RoomScene.writeUSDZ(ed.room, to: usdz)
    let bytes = (try? FileManager.default.attributesOfItem(atPath: usdz.path)[.size] as? Int) ?? 0
    check(bytes > 1000, "USDZ export writes a file (\(bytes) bytes)")
    check(RoomScene.exportName(ed.box(1)!) == "Grey_sofa_1", "export names are valid USD names")
} catch { check(false, "room edits threw \(error)") }

// MARK: clean up

do {
    // The newest real scan's boxes (room coordinates, floor at -0.64): a bed seen only from above
    // as four flat pieces, a wall piece and a door the model named, a small thing by the bed.
    let floorY: Float = -0.64, yaw: Float = 0.25, c = cos(yaw), s = sin(yaw)
    let shell = RoomShell(floorY: floorY, ceilingY: floorY + 2.25, ceilingObserved: true, yaw: yaw,
                          outline: [], walls: [], area: 11, extent: [3.3, 3.3], notes: "")
    func rbox(_ id: Int, _ u: (Float, Float), _ y: (Float, Float), _ v: (Float, Float), _ label: String? = nil,
              kind: String? = nil, source: String? = nil) -> ObjectBox {
        let mu = (u.0 + u.1) / 2, mv = (v.0 + v.1) / 2
        return ObjectBox(id: id, center: [mu * c - mv * s, floorY + (y.0 + y.1) / 2, mu * s + mv * c],
                         size: [u.1 - u.0, y.1 - y.0, v.1 - v.0], yaw: yaw, points: 200, label: label,
                         confidence: label == nil ? nil : 0.9, source: label == nil ? nil : (source ?? "model"), kind: kind)
    }
    let real: [ObjectBox] = [
        rbox(4, (2.13, 2.48), (0, 1.75), (-0.95, 0.18), "door", kind: "door"),
        rbox(20, (0.01, 0.77), (0.39, 0.64), (-1.22, -0.27)),
        rbox(2, (0.48, 1.15), (1.71, 2.03), (-1.89, -1.19), "lamp", kind: "object"),
        rbox(22, (0.23, 0.79), (0.46, 0.67), (-0.20, 0.37)),
        rbox(8, (0.63, 0.99), (0.41, 0.51), (-1.55, -0.93), "bed", kind: "furniture"),
        rbox(18, (-0.40, 0.30), (0.27, 0.33), (-1.59, -1.17)),
        rbox(6, (0.42, 1.42), (0.65, 1.47), (-2.92, -2.90), "wall"),          // older box: no kind yet
        rbox(21, (1.26, 1.44), (0.58, 0.74), (0.12, 0.38)),
        // Not from the scan: a desk with an unnamed lamp on it and a chair against it,
        // and a door you named yourself.
        rbox(30, (-0.8, 0.4), (0, 0.75), (-2.8, -2.2), "desk", kind: "furniture"),
        rbox(31, (-0.5, -0.3), (0.75, 1.2), (-2.7, -2.5)),
        rbox(32, (0.45, 1.0), (0, 0.9), (-2.85, -2.3), "office chair", kind: "furniture"),
        rbox(33, (2.2, 2.45), (0, 2.0), (-2.5, -1.6), "door", source: "user"),
    ]
    let (fixed, notes) = RoomCorrector.correct(RoomModel(shell: shell, objects: real))
    for n in notes { print("  " + n) }
    let objs = fixed.objects ?? []
    let bed = objs.first { $0.id == 8 }
    let bb = bed.map { RoomCorrector.roomBounds($0, shell: shell) }
    check(bed?.label == "bed" && objs.allSatisfy { ![20, 22, 18].contains($0.id) } && bb != nil
          && abs(bb!.lo.x + 0.40) < 0.01 && abs(bb!.hi.x - 0.99) < 0.01 && abs(bb!.lo.z + 1.59) < 0.01 && abs(bb!.hi.z - 0.37) < 0.01,
          "the four bed pieces become one bed, 1.4 × 2.0 m")
    check(bb != nil && abs(bb!.lo.y - floorY) < 0.001 && abs(bb!.hi.y - (floorY + 0.67)) < 0.001,
          "the bed is extended down to the floor")
    check(Set((fixed.removed ?? []).map(\.id)) == [4, 6] && fixed.removed?.first { $0.id == 6 }?.kind == "wall",
          "the wall piece and the door are removed (wall found from the name of an older box)")
    check(objs.contains { $0.id == 21 } && objs.contains { $0.id == 2 }, "the thing by the bed and the ceiling lamp stay")
    check(objs.contains { $0.id == 31 } && objs.contains { $0.id == 32 }
          && objs.first { $0.id == 30 }.map { abs($0.size[1] - 0.75) < 0.001 } == true,
          "a lamp on a desk and a chair against it are not merged into the desk")
    check(objs.contains { $0.id == 33 }, "a door you named yourself is left alone")

    // Fitted again: the wall piece comes back a bit different, unnamed. It stays out.
    var refit = fixed
    refit.objects = (fixed.objects ?? []) + [rbox(40, (0.44, 1.40), (0.66, 1.45), (-2.93, -2.90))]
    let (again, notes2) = RoomCorrector.correct(refit)
    check(!(again.objects ?? []).contains { $0.id == 40 } && notes2.count == 1, "a removed piece stays out after a re-fit")

    // Deleted by hand: stays deleted through a re-fit; undo brings it back.
    var ed = RoomEditor(room: fixed)
    ed.delete(21)
    check(ed.room.removed?.last?.kind == "deleted", "a deleted box is kept as removed")
    var refit2 = ed.room
    refit2.objects = (refit2.objects ?? []) + [rbox(41, (1.27, 1.45), (0.58, 0.73), (0.13, 0.38))]
    check(!(RoomCorrector.correct(refit2).0.objects ?? []).contains { $0.id == 41 }, "a deleted box stays deleted after a re-fit")
    ed.undo()
    check(ed.box(21) != nil && (ed.room.removed ?? []).allSatisfy { $0.id != 21 }, "undo of a delete restores it fully")
    var ed2 = RoomEditor(room: RoomModel(shell: shell, objects: real))
    let n = ed2.cleanUp()
    check(n.count == notes.count && ed2.canUndo, "clean up in the Room screen is one undoable step")
    ed2.undo()
    check(ed2.objects == real, "undo of a clean up puts everything back")
} catch { check(false, "clean up threw \(error)") }

// MARK: room review

do {
    // 4 × 3 m room turned 0.3 rad; walls at u = 0, u = 4, v = 0, v = 3.
    let yaw: Float = 0.3, c = cos(yaw), s = sin(yaw)
    func w(_ u: Float, _ v: Float) -> [Float] { [u * c - v * s, u * s + v * c] }
    let corners: [(Float, Float)] = [(0, 0), (4, 0), (4, 3), (0, 3)]
    let normals: [(Float, Float)] = [(0, 1), (-1, 0), (0, -1), (1, 0)]
    var walls: [RoomShell.Wall] = []
    for i in 0..<4 {
        let a = corners[i], b = corners[(i + 1) % 4]
        walls.append(.init(id: i, start: w(a.0, a.1), end: w(b.0, b.1), normal: w(normals[i].0, normals[i].1),
                           length: simd_distance(SIMD2(a.0, a.1), SIMD2(b.0, b.1)), coverage: 1))
    }
    let shell = RoomShell(floorY: 0, ceilingY: 2.5, ceilingObserved: true, yaw: yaw,
                          outline: corners.map { w($0.0, $0.1) }, walls: walls, area: 12, extent: [4, 3], notes: "")
    func rbox(_ id: Int, _ u: (Float, Float), _ y: (Float, Float), _ v: (Float, Float), _ label: String? = nil,
              source: String = "model", confidence: Float = 0.9, kind: String? = nil) -> ObjectBox {
        let mu = (u.0 + u.1) / 2, mv = (v.0 + v.1) / 2
        return ObjectBox(id: id, center: [mu * c - mv * s, (y.0 + y.1) / 2, mu * s + mv * c],
                         size: [u.1 - u.0, y.1 - y.0, v.1 - v.0], yaw: yaw, points: 200, label: label,
                         confidence: label == nil ? nil : confidence, source: label == nil ? nil : source, kind: kind)
    }
    let boxes: [ObjectBox] = [
        rbox(1, (0, 1.4), (0, 0.5), (0, 1.0), "bed"),                          // head half of a bed, in the corner
        rbox(2, (0, 1.4), (0, 0.48), (1.0, 2.0)),                              // foot half, unnamed
        rbox(3, (3.0, 3.5), (0, 0.75), (0, 1.2), "desk"),
        rbox(4, (3.1, 3.3), (0.75, 1.2), (0.2, 0.4), "lamp"),                  // on the desk
        rbox(5, (2.4, 2.95), (0, 0.9), (0.2, 0.7), "chair"),                   // at the desk
        rbox(6, (1.6, 2.2), (0, 0.4), (2.6, 3.0), "necktie", source: "vision", confidence: 0.5),
        rbox(7, (1.6, 2.4), (0, 0.8), (1.0, 1.5), "chair", source: "user"),
        rbox(8, (0.5, 0.6), (1.0, 1.05), (2.5, 2.55)),                         // tiny, unnamed
    ]
    let room = RoomModel(shell: shell, objects: boxes)
    let text = RoomReview.describe(room)
    print(text.split(separator: "\n").map { "  " + $0 }.joined(separator: "\n"))
    let lines = text.split(separator: "\n").map(String.init)
    check(lines.count == 9 && lines[0].hasPrefix("Room: 4.0 × 3.0 m floor, 2.5 m high. 8 things"), "the room is written out, one line per box")
    let l1 = lines.first { $0.hasPrefix("#1 ") } ?? ""
    check(l1.contains("bed (AI from a photo, 90%)") && l1.contains("140×100 cm, 50 cm tall") && l1.contains("0–50 cm above the floor")
          && l1.contains("at 0.7, 0.5 m") && l1.contains("against a wall") && l1.contains("touches #2"), "a box line has name, size, height, place, wall and neighbours")
    check((lines.first { $0.hasPrefix("#4 ") } ?? "").contains("on top of #3") && (lines.first { $0.hasPrefix("#3 ") } ?? "").contains("holds #4"),
          "the lamp is on top of the desk")
    check(!(lines.first { $0.hasPrefix("#7 ") } ?? "").contains("against a wall"), "a box in the middle is not against a wall")
    check((lines.first { $0.hasPrefix("#7 ") } ?? "").contains("named by the user"), "hand names are marked")
    check(RoomReview.describe(room, maxBoxes: 3).split(separator: "\n").count == 4
          && RoomReview.describe(room, maxBoxes: 3).contains("the 3 biggest listed"), "long rooms are cut to the biggest boxes")

    let edits: [ReviewEdit] = [
        ReviewEdit(action: .merge, box: 1, other: 2, name: "bed", reason: "two halves"),
        ReviewEdit(action: .merge, box: 3, other: 4, name: "desk"),           // the lamp stands on the desk
        ReviewEdit(action: .merge, box: 3, other: 6, name: "desk"),           // far apart
        ReviewEdit(action: .rename, box: 6, name: "Footstool!"),
        ReviewEdit(action: .rename, box: 5, name: "sofa"),                    // a confident photo name that fits
        ReviewEdit(action: .rename, box: 3, name: "chair"),
        ReviewEdit(action: .rename, box: 7, name: "sofa"),                    // named by the user
        ReviewEdit(action: .remove, box: 5),                                  // AI is sure it's a chair
        ReviewEdit(action: .remove, box: 8),
        ReviewEdit(action: .remove, box: 99),
        ReviewEdit(action: .rename, box: 2, name: "bed"),                     // merged away already
    ]
    let o = RoomReview.apply(edits, to: room)
    for n in o.applied { print("  applied: " + n) }
    for n in o.rejected { print("  rejected: " + n) }
    let objs = o.room.objects ?? []
    let bed = objs.first { $0.id == 1 }.map { RoomCorrector.roomBounds($0, shell: shell) }
    check(bed != nil && abs(bed!.hi.z - 2.0) < 0.01 && !objs.contains { $0.id == 2 }, "the two bed halves are merged")
    check(objs.first { $0.id == 6 }?.label == "footstool" && objs.first { $0.id == 6 }?.source == "review", "a bad photo guess is renamed, the name cleaned")
    check(objs.first { $0.id == 5 }?.label == "chair" && objs.first { $0.id == 3 }?.label == "desk",
          "a confident photo name that fits its size is not renamed")
    check(objs.contains { $0.id == 4 } && o.rejected.contains { $0.contains("lamp stands on the desk") }, "a lamp on a desk is not merged into it")
    let wide = RoomModel(shell: shell, objects: [rbox(9, (0.5, 3.2), (0, 0.85), (2.1, 3.0), "chair")])
    check((RoomReview.apply([ReviewEdit(action: .rename, box: 9, name: "sofa")], to: wide).room.objects ?? []).first?.label == "sofa",
          "a photo name that can't fit the size (a chair 2.7 m wide) is renamed")
    check(objs.first { $0.id == 7 }?.label == "chair", "a box named by hand is not renamed")
    check(objs.contains { $0.id == 5 } && !objs.contains { $0.id == 8 } && o.room.removed?.first { $0.id == 8 }?.kind == "review",
          "a sure AI name is not removed, a tiny unnamed box is")
    check(o.rejected.contains { $0.hasPrefix("merge #3 and #6") && $0.contains("don't touch") }, "boxes apart are not merged")
    check(o.rejected.contains { $0.contains("#99") } && o.rejected.contains { $0.hasPrefix("rename #2") }, "unknown and merged-away boxes are turned down")

    // Too big for the name, swallowing a third box, structural rename.
    let o2 = RoomReview.apply([
        ReviewEdit(action: .rename, box: 2, name: "lamp"),
        ReviewEdit(action: .merge, box: 1, other: 7, name: "bed"),
        ReviewEdit(action: .rename, box: 6, name: "wall"),
    ], to: room)
    for n in o2.rejected { print("  rejected: " + n) }
    check(o2.rejected.contains { $0.hasPrefix("rename #2 to lamp") && $0.contains("too big") }, "a name that doesn't fit the size is turned down")
    check(o2.rejected.contains { $0.hasPrefix("merge #1 and #7") }, "a merge that doesn't touch or fit is turned down")
    check(o2.room.removed?.first { $0.id == 6 }?.kind == "wall", "renaming an unsure box to wall removes it as wall")
    let room3 = RoomModel(shell: shell, objects: [rbox(10, (0, 1.0), (0, 0.5), (0, 1.0), "table"),
                                                    rbox(11, (1.0, 2.0), (0, 0.5), (0, 1.0), "table"),
                                                    rbox(12, (0.9, 1.1), (0.1, 0.3), (0.4, 0.6))])
    check(RoomReview.apply([ReviewEdit(action: .merge, box: 10, other: 11, name: "table")], to: room3).rejected.first?.contains("swallow #12") == true,
          "a merge that would swallow a third box is turned down")
    let withRoomBox = RoomModel(shell: shell, objects: boxes + [rbox(50, (0, 4), (0, 2.4), (0, 3))])
    check(RoomCorrector.correct(withRoomBox).0.removed?.first { $0.id == 50 }?.kind == "room",
          "a box as big as the room is taken out by the clean up")
    let cleaned = RoomCorrector.correct(withRoomBox).0
    check(RoomCorrector.correct(cleaned).0.objects?.count == cleaned.objects?.count,
          "a second clean up keeps the boxes that were inside the room-sized box")
    // A room turned 30°: a removed wall piece doesn't keep out the sofa standing against it.
    do {
        let y: Float = 0.52, c = cos(y), s = sin(y)
        func turned(_ id: Int, _ u: Float, _ v: Float, _ size: [Float], label: String? = nil, kind: String? = nil) -> ObjectBox {
            ObjectBox(id: id, center: [u * c - v * s, size[1] / 2, u * s + v * c], size: size, yaw: y, points: 100,
                      label: label, source: label == nil ? nil : "model", kind: kind)
        }
        let sh = RoomShell(floorY: 0, ceilingY: 2.5, ceilingObserved: true, yaw: y, outline: [], walls: [], area: 20,
                           extent: [5, 4], notes: "")
        let wallPiece = turned(60, 1.5, 0.05, [3, 2.4, 0.1], label: "wall", kind: "wall")
        let sofa = turned(61, 1.5, 0.55, [2, 0.85, 0.9])
        let out = RoomCorrector.correct(RoomModel(shell: sh, objects: [sofa], removed: [wallPiece])).0
        check(out.objects?.contains { $0.id == 61 } == true, "a removed wall piece doesn't keep out the sofa against it in a turned room")
    }
    var ed = RoomEditor(room: room)
    ed.replace(with: o.room)
    check(ed.room == o.room && ed.canUndo, "the review is one undoable step")
    ed.undo()
    check(ed.room == room, "undo of the review puts everything back")
    check(RoomReview.cleanName("  Office Chair ") == "office chair" && RoomReview.cleanName("unknown") == nil && RoomReview.cleanName("") == nil,
          "names are cleaned")
} catch { check(false, "room review threw \(error)") }

// Saved rooms remember which box finder made them; old files have none.
do {
    let sh = RoomShell(floorY: 0, ceilingY: 2.5, ceilingObserved: true, yaw: 0, outline: [], walls: [], area: 20,
                       extent: [5, 4], notes: "")
    let old = try ScanBundle.decoder.decode(RoomModel.self, from: try ScanBundle.encoder.encode(RoomModel(shell: sh)))
    check(old.hasOldBoxes && !RoomModel(shell: sh, finder: ObjectFinder.version).hasOldBoxes,
          "rooms without a box finder version are marked as found by an older one")
    var flat = sh; flat.area = 90; flat.extent = [14, 37]
    check(flat.looksLikeSeveralRooms && !sh.looksLikeSeveralRooms, "a 14 × 37 m outline looks like several rooms, 5 × 4 m doesn't")
} catch { check(false, "room version threw \(error)") }

// Real rooms: ./test.sh path/to/room.json … prints what clean up would change.
for path in CommandLine.arguments.dropFirst() {
    guard let data = FileManager.default.contents(atPath: path),
          let room = try? ScanBundle.decoder.decode(RoomModel.self, from: data) else { print("can't read \(path)"); continue }
    let (fixed, notes) = RoomCorrector.correct(room)
    print("\n\(path): \(room.objects?.count ?? 0) boxes → \(fixed.objects?.count ?? 0), \(fixed.removed?.count ?? 0) removed")
    for n in notes { print("  " + n) }
    print(RoomReview.describe(fixed).split(separator: "\n").map { "  | " + $0 }.joined(separator: "\n"))
}

print(failures == 0 ? "\nAll checks passed" : "\n\(failures) check(s) failed")
exit(failures == 0 ? 0 : 1)
