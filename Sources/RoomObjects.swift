import Foundation
import simd

/// Builds the room as 3D objects (see `ROOM_MODEL_PLAN.md`): each box becomes a model fitted to
/// its scan points, or its cleaned piece of the scan; the floor and walls get the scan's colors.
/// Geometry only, so it runs on the Mac too; naming from photos is `RoomNamer`.
enum RoomObjects {
    struct Built {
        var room: RoomModel
        /// New scan pieces by file name, in their box's frame; the caller saves them.
        var pieces: [String: MeshOut]
        var notes: [String]
    }

    static func pieceName(_ id: Int) -> String { "object-\(id).ply" }

    /// True when a box needs its object built again: none yet, or its size or name changed since.
    static func isStale(_ b: ObjectBox) -> Bool {
        guard let m = b.model else { return true }
        let sameSize = m.boxSize.count == 3 && zip(m.boxSize, b.size).allSatisfy { abs($0 - $1) < 0.005 }
        return !sameSize || m.type != ObjectModeler.type(b.label)
    }

    /// Objects for every stale box (or just `only`), plus floor and wall colors.
    static func build(_ room: RoomModel, meshes: [MeshOut], samples given: [ColorSample]? = nil,
                      only: Set<Int>? = nil, force: Bool = false) -> Built {
        let samples = given ?? ColorSampler.sample(meshes)
        var out = room
        var pieces: [String: MeshOut] = [:]
        var notes: [String] = []
        var objects = room.objects ?? []
        for k in objects.indices {
            let b = objects[k]
            guard only?.contains(b.id) ?? true, force || isStale(b) else { continue }
            let pts = ObjectModeler.points(b, samples: samples, shell: room.shell)
            if var m = ObjectModeler.model(b, points: pts, shell: room.shell) {
                if m.pieceAbove != nil || m.pieceGap != nil,
                   let piece = ScanPiece.cut(meshes, box: b, shell: room.shell, above: m.pieceAbove, gap: m.pieceGap) {
                    m.piece = pieceName(b.id); pieces[pieceName(b.id)] = piece
                } else {
                    m.pieceAbove = nil; m.pieceGap = nil
                }
                objects[k].model = m
                notes.append(String(format: "box %d (%@, %.2f×%.2f×%.2f): %@ model, %d parts%@, fit %.0f%%", b.id, b.label ?? "unnamed",
                                    b.size[0], b.size[1], b.size[2], m.type, m.parts.count, m.piece != nil ? " + scan" : "", m.fit * 100))
            } else if let piece = ScanPiece.cut(meshes, box: b, shell: room.shell) {
                let name = pieceName(b.id)
                pieces[name] = piece
                objects[k].model = ObjectModel(source: .scanned, piece: name, fit: 1, type: ObjectModeler.type(b.label),
                                               boxSize: b.size)
                notes.append("box \(b.id) (\(b.label ?? "unnamed"), " + b.size.map { String(format: "%.2f", $0) }.joined(separator: "×") + "): scan piece, \(piece.idx.count / 3) triangles")
            } else {
                objects[k].model = nil
                notes.append("box \(b.id) (\(b.label ?? "unnamed")): nothing to build from")
            }
        }
        out.objects = objects
        let (floor, wall) = colors(samples, shell: room.shell)
        out.floorRGB = floor; out.wallRGB = wall
        return Built(room: out, pieces: pieces, notes: notes)
    }

    /// Median colors of the floor (facing up, within 3 cm of it) and the walls (facing like a
    /// wall, within 4 cm of it).
    static func colors(_ samples: [ColorSample], shell: RoomShell) -> (floor: [Float]?, wall: [Float]?) {
        let walls = shell.walls.map { (a: SIMD2($0.start[0], $0.start[1]), e: SIMD2($0.end[0], $0.end[1]),
                                       n: SIMD2($0.normal[0], $0.normal[1])) }
        var floor: [SIMD3<Float>] = [], wall: [SIMD3<Float>] = []
        for (i, s) in samples.enumerated() where i % 3 == 0 {
            if s.n.y > 0.9, abs(s.p.y - shell.floorY) < 0.03 { floor.append(s.rgb); continue }
            guard abs(s.n.y) < 0.3 else { continue }
            let q = SIMD2(s.p.x, s.p.z), nn = SIMD2(s.n.x, s.n.z)
            if walls.contains(where: { ObjectFinder.distance(q, $0.a, $0.e) < 0.04 && simd_dot(nn, $0.n) > 0.8 }) { wall.append(s.rgb) }
        }
        func med(_ v: [SIMD3<Float>]) -> [Float]? {
            guard v.count >= 50 else { return nil }
            let m = ObjectModeler.median(v.map { ObjectModeler.Pt(x: 0, h: 0, z: 0, n: .zero, rgb: $0) })
            return [m.x, m.y, m.z]
        }
        return (med(floor), med(wall))
    }
}
