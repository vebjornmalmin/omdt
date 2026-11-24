import Foundation
import simd

/// What kind of thing a box is, read from its name. Drives colors and export names.
enum BoxCategory: String, CaseIterable, Codable {
    case sofa, chair, table, desk, bed, cabinet, shelf, wardrobe, tv, lamp, plant, rug,
         appliance, sink, toilet, door, window, radiator, decor, other

    /// Names offered when naming a box by hand.
    static let quickNames = ["Sofa", "Chair", "Table", "Desk", "Bed", "Cabinet", "Shelf", "Wardrobe",
                             "TV", "Lamp", "Plant", "Rug", "Fridge", "Door", "Window", "Radiator"]

    private static let words: [String: BoxCategory] = {
        var m: [String: BoxCategory] = [:]
        func add(_ c: BoxCategory, _ ws: String) { for w in ws.split(separator: " ") { m[String(w)] = c } }
        add(.sofa, "sofa couch loveseat settee")
        add(.chair, "chair armchair stool bench seat ottoman pouf")
        add(.table, "table")
        add(.desk, "desk workstation")
        add(.bed, "bed mattress crib bunk")
        add(.cabinet, "cabinet cupboard drawer drawers dresser sideboard nightstand stand chest counter countertop bureau credenza")
        add(.shelf, "shelf shelves shelving bookcase bookcases bookshelf bookshelves rack")
        add(.wardrobe, "wardrobe closet armoire")
        add(.tv, "tv television monitor screen display")
        add(.lamp, "lamp light chandelier lantern")
        add(.plant, "plant houseplant flower flowers tree")
        add(.rug, "rug carpet mat")
        add(.appliance, "fridge refrigerator freezer oven stove microwave dishwasher washer dryer appliance")
        add(.sink, "sink basin washbasin")
        add(.toilet, "toilet")
        add(.door, "door doorway")
        add(.window, "window blinds curtain curtains")
        add(.radiator, "radiator heater")
        add(.decor, "picture painting frame mirror poster artwork art vase clock")
        return m
    }()

    /// The category of a name. English noun phrases end in their head word ("desk lamp" is a lamp),
    /// so words are tried from the last one back.
    static func of(_ label: String?) -> BoxCategory {
        guard let label else { return .other }
        let ws = label.lowercased().split { !$0.isLetter }.map(String.init)
        for w in ws.reversed() {
            if let c = words[w] { return c }
            if w.hasSuffix("s"), let c = words[String(w.dropLast())] { return c }
        }
        return .other
    }

    /// Display color, RGB 0-1.
    var rgb: SIMD3<Float> {
        switch self {
        case .sofa: return SIMD3(0.36, 0.55, 0.85)
        case .chair: return SIMD3(0.30, 0.72, 0.80)
        case .table: return SIMD3(0.85, 0.62, 0.30)
        case .desk: return SIMD3(0.80, 0.50, 0.25)
        case .bed: return SIMD3(0.62, 0.45, 0.85)
        case .cabinet: return SIMD3(0.65, 0.50, 0.38)
        case .shelf: return SIMD3(0.75, 0.58, 0.45)
        case .wardrobe: return SIMD3(0.55, 0.42, 0.33)
        case .tv: return SIMD3(0.25, 0.25, 0.30)
        case .lamp: return SIMD3(0.98, 0.85, 0.35)
        case .plant: return SIMD3(0.35, 0.75, 0.40)
        case .rug: return SIMD3(0.80, 0.40, 0.45)
        case .appliance: return SIMD3(0.80, 0.82, 0.85)
        case .sink, .toilet: return SIMD3(0.70, 0.85, 0.95)
        case .door: return SIMD3(0.60, 0.40, 0.25)
        case .window: return SIMD3(0.55, 0.80, 0.95)
        case .radiator: return SIMD3(0.90, 0.45, 0.35)
        case .decor: return SIMD3(0.90, 0.55, 0.75)
        case .other: return SIMD3(0.62, 0.62, 0.62)
        }
    }
}

/// Hand edits of a room's boxes: move, turn, rename, merge, delete, with undo.
/// Moves snap boxes against nearby walls.
struct RoomEditor {
    private(set) var room: RoomModel
    private var undoStack: [RoomModel] = []
    private var redoStack: [RoomModel] = []
    /// A box side this close to a wall (or poking through it) is put flush against it, in meters.
    var snapDistance: Float = 0.10

    init(room: RoomModel) { self.room = room }

    var objects: [ObjectBox] { room.objects ?? [] }
    var canUndo: Bool { !undoStack.isEmpty }
    var canRedo: Bool { !redoStack.isEmpty }

    func box(_ id: Int) -> ObjectBox? { objects.first { $0.id == id } }

    // MARK: undo

    private mutating func record() {
        undoStack.append(room)
        if undoStack.count > 100 { undoStack.removeFirst() }
        redoStack.removeAll()
    }

    mutating func undo() {
        guard let last = undoStack.popLast() else { return }
        redoStack.append(room)
        room = last
    }

    mutating func redo() {
        guard let next = redoStack.popLast() else { return }
        undoStack.append(room)
        room = next
    }

    private mutating func change(_ id: Int, _ f: (inout ObjectBox) -> Void) {
        guard var list = room.objects, let i = list.firstIndex(where: { $0.id == id }) else { return }
        f(&list[i])
        room.objects = list
    }

    // MARK: edits

    /// Call once when a drag starts, so the whole drag undoes in one step.
    mutating func beginMove() { record() }

    /// Moves a box's footprint center to `xz` (world), keeping its height, then snaps it to walls.
    mutating func move(_ id: Int, to xz: SIMD2<Float>) {
        guard let b = box(id) else { return }
        let c = snapped(b, xz)
        change(id) { $0.center = [c.x, $0.center[1], c.y] }
    }

    /// Turns a box a quarter turn about its center.
    mutating func rotate(_ id: Int) {
        guard let b = box(id) else { return }
        record()
        var yaw = b.yaw + .pi / 2
        if yaw > .pi { yaw -= 2 * .pi }
        change(id) { $0.yaw = yaw }
        if let nb = box(id) {
            let c = snapped(nb, SIMD2(nb.center[0], nb.center[2]))
            change(id) { $0.center = [c.x, $0.center[1], c.y] }
        }
    }

    /// Sets a box's name by hand. An empty name clears it.
    mutating func rename(_ id: Int, to name: String) {
        record()
        let n = name.trimmingCharacters(in: .whitespacesAndNewlines)
        change(id) { b in
            b.label = n.isEmpty ? nil : n
            b.source = n.isEmpty ? nil : "user"
            b.confidence = n.isEmpty ? nil : 1
        }
    }

    /// Sets the name the on-device model gave from the box's photos. A box you named stays yours.
    mutating func setModelName(_ id: Int, name: String, confidence: Float, kind: String?) {
        guard let b = box(id), b.source != "user" else { return }
        let n = name.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard !n.isEmpty else { return }
        record()
        change(id) { b in
            b.label = n; b.source = "model"; b.confidence = min(1, max(0, confidence)); b.kind = kind
        }
    }

    /// Takes the 3D objects built for `built`'s boxes, for boxes that still have the size they
    /// were built for. Not an undo step: objects follow from the boxes.
    mutating func setObjects(from built: RoomModel) {
        guard var list = room.objects else { return }
        for b in built.objects ?? [] {
            guard let m = b.model, let i = list.firstIndex(where: { $0.id == b.id }),
                  zip(m.boxSize, list[i].size).allSatisfy({ abs($0 - $1) < 0.005 }) else { continue }
            list[i].model = m
        }
        room.objects = list
        room.floorRGB = built.floorRGB ?? room.floorRGB
        room.wallRGB = built.wallRGB ?? room.wallRGB
    }

    /// Removes a box. It is kept in `removed` so it stays out when the room is fitted again.
    mutating func delete(_ id: Int) {
        guard var b = box(id) else { return }
        record()
        room.objects = objects.filter { $0.id != id }
        b.kind = "deleted"
        room.removed = (room.removed ?? []) + [b]
    }

    /// Runs the automatic clean-up (see `RoomCorrector`) as one undoable step. Returns what changed.
    @discardableResult
    mutating func cleanUp() -> [String] {
        let (fixed, notes) = RoomCorrector.correct(room)
        guard !notes.isEmpty else { return [] }
        record()
        room = fixed
        return notes
    }

    /// Swaps in a reviewed room as one undoable step.
    mutating func replace(with new: RoomModel) {
        guard new != room else { return }
        record()
        room = new
    }

    /// Joins two boxes into one box around both, kept in the bigger one's direction and id.
    /// Returns the id of the joined box.
    @discardableResult
    mutating func merge(_ a: Int, _ b: Int) -> Int? {
        guard a != b, let ba = box(a), let bb = box(b) else { return nil }
        record()
        let (keep, other) = ba.volume >= bb.volume ? (ba, bb) : (bb, ba)
        let c = cos(keep.yaw), s = sin(keep.yaw)
        var lo = SIMD3<Float>(repeating: .infinity), hi = SIMD3<Float>(repeating: -.infinity)
        for p in keep.corners + other.corners {
            let q = SIMD3(p.x * c + p.z * s, p.y, -p.x * s + p.z * c)   // into keep's frame
            lo = simd_min(lo, q); hi = simd_max(hi, q)
        }
        let mid = (lo + hi) / 2, ext = hi - lo
        var m = keep
        m.center = [mid.x * c - mid.z * s, mid.y, mid.x * s + mid.z * c]
        m.size = [ext.x, ext.y, ext.z]
        m.points = keep.points + other.points
        if m.label == nil {
            m.label = other.label; m.confidence = other.confidence; m.source = other.source; m.guesses = other.guesses
        }
        room.objects = objects.compactMap { $0.id == other.id ? nil : ($0.id == keep.id ? m : $0) }
        return keep.id
    }

    /// Cuts a box in two (see `BoxSplitter`), using the scan's surface points (world). The bigger
    /// part keeps the id and name; the other gets a new id and no name. Returns both ids, or nil
    /// when the points show no place to cut.
    @discardableResult
    mutating func split(_ id: Int, points: [SIMD3<Float>]) -> (Int, Int)? {
        guard let b = box(id), let (big, small) = BoxSplitter.split(b, points: points, floorY: room.shell.floorY) else { return nil }
        record()
        let newId = (objects.map(\.id) + (room.removed ?? []).map(\.id)).max().map { $0 + 1 } ?? 1
        var a = b; a.center = big.center; a.size = big.size; a.points = big.points
        var c = b; c.id = newId; c.center = small.center; c.size = small.size; c.points = small.points
        c.label = nil; c.confidence = nil; c.source = nil; c.guesses = nil; c.kind = nil
        room.objects = objects.flatMap { $0.id == id ? [a, c] : [$0] }
        return (id, newId)
    }

    // MARK: wall snapping

    /// Footprint center after snapping box `b`, placed at `xz`, against the walls.
    func snapped(_ b: ObjectBox, _ xz: SIMD2<Float>) -> SIMD2<Float> {
        let shell = room.shell
        let c = cos(shell.yaw), s = sin(shell.yaw)
        func toRoom(_ p: SIMD2<Float>) -> SIMD2<Float> { SIMD2(p.x * c + p.y * s, -p.x * s + p.y * c) }
        func toWorld(_ q: SIMD2<Float>) -> SIMD2<Float> { SIMD2(q.x * c - q.y * s, q.x * s + q.y * c) }

        // Half footprint along the room axes (exact for quarter turns, a safe bound otherwise).
        let d = b.yaw - shell.yaw
        let hu = abs(cos(d)) * b.size[0] / 2 + abs(sin(d)) * b.size[2] / 2
        let hv = abs(sin(d)) * b.size[0] / 2 + abs(cos(d)) * b.size[2] / 2
        var q = toRoom(xz)

        for w in shell.walls {
            let a = toRoom(SIMD2(w.start[0], w.start[1])), e = toRoom(SIMD2(w.end[0], w.end[1]))
            let n = toRoom(SIMD2(w.normal[0], w.normal[1]))
            if abs(n.x) > 0.9 {
                // Wall along v at u = a.x; the room is on the side the normal points to.
                let lo = min(a.y, e.y), hi = max(a.y, e.y)
                guard q.y + hv > lo + 0.01, q.y - hv < hi - 0.01 else { continue }
                if n.x > 0, q.x > a.x, q.x - hu - a.x < snapDistance { q.x = a.x + hu }
                if n.x < 0, q.x < a.x, a.x - (q.x + hu) < snapDistance { q.x = a.x - hu }
            } else if abs(n.y) > 0.9 {
                let lo = min(a.x, e.x), hi = max(a.x, e.x)
                guard q.x + hu > lo + 0.01, q.x - hu < hi - 0.01 else { continue }
                if n.y > 0, q.y > a.y, q.y - hv - a.y < snapDistance { q.y = a.y + hv }
                if n.y < 0, q.y < a.y, a.y - (q.y + hv) < snapDistance { q.y = a.y - hv }
            }
        }
        return toWorld(q)
    }
}

/// Cuts one box that holds two things (a sofa and the coffee table in front of it) in two, from
/// above: along each side, the line between 5 cm slices where the emptier slice has the fewest
/// occupied cells (helped by a step in mean height, then by being near the middle) is the cut,
/// leaving each part at least 30 cm and a quarter of the length. No cut when the box is about as full everywhere and has no step
/// (one thing, such as a table with chairs all round). Each part gets a
/// tight box around its points; parts that reached the floor still stand on it.
enum BoxSplitter {
    struct Part { var center: [Float]; var size: [Float]; var points: Int }

    static func split(_ b: ObjectBox, points world: [SIMD3<Float>], floorY: Float) -> (Part, Part)? {
        let c = cos(b.yaw), s = sin(b.yaw)
        let h = SIMD3(b.size[0], b.size[1], b.size[2]) / 2
        // Points inside the box, in its own frame (x along size[0], z along size[2]).
        var local: [SIMD3<Float>] = []
        for p in world {
            let d = p - b.centerVec
            let q = SIMD3(d.x * c + d.z * s, d.y, -d.x * s + d.z * c)
            if abs(q.x) <= h.x + 0.02, abs(q.y) <= h.y + 0.02, abs(q.z) <= h.z + 0.02,
               p.y > floorY + 0.04 { local.append(q) }
        }
        guard local.count >= 40 else { return nil }
        let cell: Float = 0.05
        var best: (axis: Int, at: Float, score: Float, clear: Bool)?
        for axis in [0, 2] {
            let other = axis == 0 ? 2 : 0
            let len = 2 * h[axis]
            // Each part keeps at least 30 cm and a quarter of the length.
            let margin = max(0.3, 0.25 * len)
            guard len >= 2 * margin + cell else { continue }
            let n = Int(len / cell) + 1, m = Int(2 * h[other] / cell) + 1
            var tops = [[Int: Float]](repeating: [:], count: n)
            for q in local {
                let i = min(n - 1, max(0, Int((q[axis] + h[axis]) / cell)))
                let j = min(m - 1, max(0, Int((q[other] + h[other]) / cell)))
                tops[i][j] = max(tops[i][j] ?? -.infinity, q.y)
            }
            let occ = tops.map(\.count)
            // Mean height of the occupied cells per slice (nil when empty).
            let mean: [Float?] = tops.map { $0.isEmpty ? nil : $0.values.reduce(0, +) / Float($0.count) }
            func avg(_ r: ClosedRange<Int>) -> Float? {
                let v = r.clamped(to: 0...(n - 1)).compactMap { mean[$0] }
                return v.isEmpty ? nil : v.reduce(0, +) / Float(v.count)
            }
            let typical = Float(occ.sorted()[n / 2]) / Float(m)
            // Cut lines between slice b-1 and b. A cut is clear when the emptier slice holds at most
            // 70 % of a typical slice's cells, or the mean height over 15 cm on each side differs
            // by 25 cm or more.
            let first = Int((margin / cell).rounded(.up)), last = Int(((len - margin) / cell).rounded(.down))
            guard first <= last else { continue }
            for b in first...last {
                let fill = Float(min(occ[b - 1], occ[b])) / Float(m)
                let l = avg((b - 3)...(b - 1)), r = avg(b...(b + 2))
                let step = (l != nil && r != nil) ? abs(l! - r!) : 0
                let mid = abs(Float(b) - Float(n) / 2) / Float(n)
                let score = fill - 0.3 * min(step, 1) + 0.05 * mid
                let clear = fill <= 0.7 * typical || step >= 0.25
                // Clear cuts first, then the lowest score.
                if best == nil || (clear && !best!.clear) || (clear == best!.clear && score < best!.score) {
                    best = (axis, -h[axis] + Float(b) * cell, score, clear)
                }
            }
        }
        guard let cut = best, cut.clear else { return nil }
        let sides = [local.filter { $0[cut.axis] < cut.at }, local.filter { $0[cut.axis] >= cut.at }]
        guard sides.allSatisfy({ $0.count >= 20 }) else { return nil }
        let bottom = -h.y
        let parts = sides.map { pts -> Part in
            // Inside the box that was cut.
            var lo = simd_max(pts.reduce(SIMD3<Float>(repeating: .infinity)) { simd_min($0, $1) }, -h)
            let hi = simd_min(pts.reduce(SIMD3<Float>(repeating: -.infinity)) { simd_max($0, $1) }, h)
            if lo.y - bottom < 0.15 { lo.y = bottom }        // stood on the floor before: still does
            let mid = (lo + hi) / 2, ext = simd_max(hi - lo, SIMD3(repeating: 0.02))
            let w = b.centerVec + SIMD3(mid.x * c - mid.z * s, mid.y, mid.x * s + mid.z * c)
            return Part(center: [w.x, w.y, w.z], size: [ext.x, ext.y, ext.z], points: pts.count)
        }
        // Both parts must hold something real across the cut, not a few stray points.
        guard parts.allSatisfy({ $0.size[cut.axis] >= 0.2 && min($0.size[0], $0.size[2]) >= 0.15 }) else { return nil }
        let vol = { (p: Part) in p.size[0] * p.size[1] * p.size[2] }
        return vol(parts[0]) >= vol(parts[1]) ? (parts[0], parts[1]) : (parts[1], parts[0])
    }
}

/// Triangulates a simple polygon (any winding) by ear clipping. Returns index triples.
enum PolygonTriangulator {
    static func triangulate(_ pts: [SIMD2<Float>]) -> [Int] {
        guard pts.count >= 3 else { return [] }
        var area: Float = 0
        for i in pts.indices { let a = pts[i], b = pts[(i + 1) % pts.count]; area += a.x * b.y - b.x * a.y }
        var idx = Array(pts.indices)
        if area < 0 { idx.reverse() }                       // work counter-clockwise
        func cross(_ o: SIMD2<Float>, _ a: SIMD2<Float>, _ b: SIMD2<Float>) -> Float {
            (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)
        }
        var out: [Int] = []
        var guardCount = 0
        while idx.count > 3 && guardCount < 10_000 {
            guardCount += 1
            var clipped = false
            for k in idx.indices {
                let i0 = idx[(k + idx.count - 1) % idx.count], i1 = idx[k], i2 = idx[(k + 1) % idx.count]
                let a = pts[i0], b = pts[i1], c = pts[i2]
                guard cross(a, b, c) > 1e-9 else { continue }  // reflex or flat corner
                let inside = idx.contains { j in
                    guard j != i0, j != i1, j != i2 else { return false }
                    let p = pts[j]
                    return cross(a, b, p) >= 0 && cross(b, c, p) >= 0 && cross(c, a, p) >= 0
                }
                if inside { continue }
                out += [i0, i1, i2]
                idx.remove(at: k)
                clipped = true
                break
            }
            if !clipped {                                    // only flat corners left: drop one
                idx.remove(at: 0)
            }
        }
        if idx.count == 3 { out += idx }
        return out
    }
}
