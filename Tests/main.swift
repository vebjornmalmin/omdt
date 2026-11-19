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

    // room.json round trip.
    let dir = try ScanBundle.makeRecordingDirectory()
    let bundle = ScanBundle(url: dir, meta: ScanMeta(id: "x", version: 1, created: Date(), keyframeCount: 0,
                                                     triangleCount: 0, boundsMin: [], boundsMax: [], coverKeyframe: nil))
    try bundle.saveRoom(RoomModel(shell: room))
    check(bundle.loadRoom()?.shell == room, "room.json round trip")
} catch { check(false, "room fit threw \(error)") }

print(failures == 0 ? "\nAll checks passed" : "\n\(failures) check(s) failed")
exit(failures == 0 ? 0 : 1)
