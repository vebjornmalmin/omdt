import Foundation
import simd

/// Fixes the usual mistakes in a fitted room's boxes, using what the on-device model said about
/// each box plus geometry:
/// 1. Boxes that were removed before (by hand, or as wall/door pieces) stay out after a re-fit.
/// 2. Pieces of wall, floor or ceiling, doors and windows move to `removed`: they are part of the
///    room, not things in it.
/// 3. Fragments of one thing are merged: boxes that touch, sit at about the same height, are not
///    stacked on each other, and are unnamed or named alike (the four flat pieces of a bed top).
/// 4. Furniture that stands on the floor but was only seen from above (a bed or table top) is
///    extended down to the floor.
/// Names given by hand are never overruled.
enum RoomCorrector {
    struct Options {
        /// Boxes this close (or overlapping) on the floor plan count as touching, m.
        var touchGap: Float = 0.08
        /// Height ranges this far apart (or overlapping) count as the same height, m.
        var heightGap: Float = 0.10
        /// A merge must fill its new box: union footprint ≤ this × the floor area the pieces cover.
        var fillRatio: Float = 1.8
        /// Furniture whose bottom is this far above the floor gets extended down to it, m.
        var standGap: Float = 0.12
        /// …unless its top is higher than this above the floor, m.
        var standMaxTop: Float = 1.5
    }

    static let structural: Set<String> = ["wall", "floor", "ceiling", "door", "window"]
    private static let structuralWords: Set<String> = ["wall", "walls", "floor", "ceiling", "door", "doorway", "window"]
    /// Kinds of furniture that stand on the floor.
    static let standing: Set<BoxCategory> = [.bed, .sofa, .table, .desk, .cabinet, .wardrobe, .chair, .appliance, .toilet]

    /// What a box is part of, if it is part of the room itself: from the model's kind, or else
    /// from the last word of its name.
    static func structureKind(_ b: ObjectBox) -> String? {
        guard b.source != "user" else { return nil }
        if let k = b.kind { return structural.contains(k) ? k : nil }
        guard let last = b.label?.lowercased().split(whereSeparator: { !$0.isLetter }).last.map(String.init),
              structuralWords.contains(last) else { return nil }
        return last == "walls" ? "wall" : last == "doorway" ? "door" : last
    }

    /// The cleaned room, and one line per change for the log.
    static func correct(_ room: RoomModel, options o: Options = Options()) -> (RoomModel, [String]) {
        var notes: [String] = []
        var objects = room.objects ?? []
        var removed = room.removed ?? []
        let shell = room.shell

        // 1. Keep out what was removed before.
        objects.removeAll { b in
            // A box removed for being the room itself keeps nothing out: it would be found and
            // removed again anyway, and everything in the room lies inside it.
            guard b.source != "user",
                  let r = removed.first(where: { $0.kind != "room" && covered(b, by: $0, shell: shell) }) else { return false }
            notes.append("kept box \(b.id) out (\(r.kind ?? "removed") \(r.id))")
            return true
        }

        // 2. Room structure out, and boxes that are the room itself (as big and tall as it).
        objects.removeAll { b in
            guard let k = structureKind(b) ?? (roomSized(b, shell: shell) ? "room" : nil) else { return false }
            var r = b; r.kind = k
            removed.append(r)
            notes.append("removed box \(b.id) (\(b.label ?? k), \(k))")
            return true
        }

        // 3. Merge fragments, best-filling pairs first, until nothing merges. `covers` is the floor
        // area the original pieces of each box cover, so a growing box can't swallow its neighbours.
        var covers: [Int: Float] = [:]
        for b in objects { covers[b.id] = footprint(b, shell: shell) }
        var merged = true
        while merged {
            merged = false
            var best: (i: Int, j: Int, fill: Float)?
            for i in objects.indices {
                for j in objects.indices where j > i {
                    let a = objects[i], b = objects[j]
                    guard let fill = mergeFill(a, b, covered: (covers[a.id] ?? 0) + (covers[b.id] ?? 0), shell: shell, o: o),
                          fits(merge(a, b, shell: shell), shell: shell) else { continue }
                    if best == nil || fill < best!.fill { best = (i, j, fill) }
                }
            }
            if let (i, j, _) = best {
                let m = merge(objects[i], objects[j], shell: shell)
                let other = m.id == objects[i].id ? objects[j] : objects[i]
                covers[m.id] = (covers[objects[i].id] ?? 0) + (covers[objects[j].id] ?? 0)
                notes.append("merged box \(other.id) into \(m.id)" + (m.label.map { " (\($0))" } ?? ""))
                objects[objects[i].id == m.id ? i : j] = m
                objects.remove(at: objects[i].id == m.id ? j : i)
                merged = true
            }
        }

        // 4. Standing furniture down to the floor.
        for k in objects.indices {
            let b = objects[k]
            let bottom = b.center[1] - b.size[1] / 2, top = b.center[1] + b.size[1] / 2
            guard standing.contains(BoxCategory.of(b.label)), bottom > shell.floorY + o.standGap,
                  top < shell.floorY + o.standMaxTop else { continue }
            objects[k].size[1] = top - shell.floorY
            objects[k].center[1] = (top + shell.floorY) / 2
            notes.append(String(format: "box %d (%@) now stands on the floor (was %.0f cm up)", b.id, b.label ?? "", (bottom - shell.floorY) * 100))
        }

        var out = room
        out.objects = objects.sorted { $0.volume > $1.volume }
        out.removed = removed.isEmpty ? nil : removed
        return (out, notes)
    }

    // MARK: geometry

    /// Room-frame bounds (u, y, v) of a box.
    static func roomBounds(_ b: ObjectBox, shell: RoomShell) -> (lo: SIMD3<Float>, hi: SIMD3<Float>) {
        let c = cos(shell.yaw), s = sin(shell.yaw)
        var lo = SIMD3<Float>(repeating: .infinity), hi = SIMD3<Float>(repeating: -.infinity)
        for p in b.corners {
            let q = SIMD3(p.x * c + p.z * s, p.y, -p.x * s + p.z * c)
            lo = simd_min(lo, q); hi = simd_max(hi, q)
        }
        return (lo, hi)
    }

    /// True when `b` is the removed box `r` found again: most of `b` lies inside `r` (grown by
    /// 3 cm, since re-fitted boxes wobble), and `r` is not much bigger, so a removed box the size
    /// of the room doesn't keep out everything in it. A wall piece re-fitted in up to 4 parts
    /// still counts.
    /// Measured along the room's axes, which boxes are turned to.
    static func covered(_ b: ObjectBox, by r: ObjectBox, shell: RoomShell) -> Bool {
        let (bl, bh) = roomBounds(b, shell: shell)
        let (rl0, rh0) = roomBounds(r, shell: shell)
        let rl = rl0 - 0.03, rh = rh0 + 0.03
        let e = simd_max(simd_min(bh, rh) - simd_max(bl, rl), .zero)
        let vol = max(b.volume, 1e-6)
        return e.x * e.y * e.z / vol > 0.5 && r.volume <= 4 * vol
    }

    /// True for an unnamed-by-hand box that covers most of the floor and most of the height:
    /// the room's own surfaces grouped into one box, not a thing in it.
    static func roomSized(_ b: ObjectBox, shell: RoomShell) -> Bool {
        b.source != "user" && footprint(b, shell: shell) > 0.6 * max(shell.area, 0.1) && b.size[1] > 0.5 * shell.height
    }

    /// Floor area of a box's footprint, m².
    static func footprint(_ b: ObjectBox, shell: RoomShell) -> Float {
        let (l, h) = roomBounds(b, shell: shell)
        return max(h.x - l.x, 0.01) * max(h.z - l.z, 0.01)
    }

    /// Largest believable footprint (long side, short side) and height, m. Unnamed boxes get a
    /// middle size so fragments can join up to a bed or a desk, but not into a row of desks.
    static func maxSize(_ c: BoxCategory?) -> (long: Float, short: Float, height: Float) {
        switch c {
        case .bed?: return (2.4, 2.2, 1.4)
        case .sofa?: return (3.4, 2.0, 1.2)
        case .table?: return (3.2, 1.6, 1.2)
        case .desk?: return (2.4, 1.4, 1.6)
        case .chair?: return (1.1, 1.1, 1.5)
        case .cabinet?, .shelf?, .wardrobe?: return (2.6, 0.9, 2.6)
        case .tv?: return (2.2, 0.6, 1.4)
        case .lamp?: return (0.9, 0.9, 2.2)
        case .plant?: return (1.4, 1.4, 2.6)
        case .appliance?: return (1.2, 1.0, 2.2)
        case .rug?: return (4.0, 3.0, 0.1)
        case nil: return (2.1, 1.3, 2.0)
        default: return (1.5, 1.0, 2.0)
        }
    }

    /// True when a merged box is still a believable size for what it is called.
    static func fits(_ m: ObjectBox, shell: RoomShell) -> Bool {
        let (l, h) = roomBounds(m, shell: shell)
        let du = h.x - l.x, dv = h.z - l.z
        let limit = maxSize(m.label.map { BoxCategory.of($0) })
        return max(du, dv) <= limit.long && min(du, dv) <= limit.short && h.y - l.y <= limit.height
    }

    /// How well two boxes would merge (union footprint / floor area the pieces cover, lower is
    /// better), or nil when they should not merge.
    static func mergeFill(_ a: ObjectBox, _ b: ObjectBox, covered: Float? = nil, shell: RoomShell, o: Options) -> Float? {
        // Names: two named boxes merge only when they name the same kind of thing.
        if let la = a.label, let lb = b.label {
            let ca = BoxCategory.of(la), cb = BoxCategory.of(lb)
            guard la.lowercased() == lb.lowercased() || (ca == cb && ca != .other) else { return nil }
        }
        let (al, ah) = roomBounds(a, shell: shell), (bl, bh) = roomBounds(b, shell: shell)
        // Touching on the floor plan.
        let gapU = max(bl.x - ah.x, al.x - bh.x), gapV = max(bl.z - ah.z, al.z - bh.z)
        guard gapU <= o.touchGap, gapV <= o.touchGap else { return nil }
        // About the same height.
        let gapY = max(bl.y - ah.y, al.y - bh.y)
        guard gapY <= o.heightGap else { return nil }
        // Not one standing on the other (a lamp on a desk stays its own box).
        if stacked(a, al, ah, on: bl, bh) || stacked(b, bl, bh, on: al, ah) { return nil }
        // The union has to be filled by the two, not an L with a big empty corner.
        let area = { (l: SIMD3<Float>, h: SIMD3<Float>) in max(h.x - l.x, 0.01) * max(h.z - l.z, 0.01) }
        let lo = simd_min(al, bl), hi = simd_max(ah, bh)
        let fill = area(lo, hi) / (covered ?? (area(al, ah) + area(bl, bh)))
        return fill <= o.fillRatio ? fill : nil
    }

    /// True when the box with bounds (tl, th) sits on top of (bl, bh): mostly over it, and its
    /// bottom at or above the other's top.
    private static func stacked(_ t: ObjectBox, _ tl: SIMD3<Float>, _ th: SIMD3<Float>,
                                on bl: SIMD3<Float>, _ bh: SIMD3<Float>) -> Bool {
        let ou = max(0, min(th.x, bh.x) - max(tl.x, bl.x)), ov = max(0, min(th.z, bh.z) - max(tl.z, bl.z))
        let over = ou * ov / max((th.x - tl.x) * (th.z - tl.z), 1e-6)
        return over > 0.6 && tl.y >= bh.y - 0.05
    }

    /// One box around both, in the room's directions. Keeps the id, name and kind of the box
    /// with the better name (yours, then the model's, then the classifier's), else the bigger one.
    static func merge(_ a: ObjectBox, _ b: ObjectBox, shell: RoomShell) -> ObjectBox {
        func rank(_ x: ObjectBox) -> (Int, Float) {
            let r = x.source == "user" ? 3 : x.source == "model" ? 2 : x.label != nil ? 1 : 0
            return (r, x.volume)
        }
        let (keep, other) = rank(a) >= rank(b) ? (a, b) : (b, a)
        let (al, ah) = roomBounds(a, shell: shell), (bl, bh) = roomBounds(b, shell: shell)
        let lo = simd_min(al, bl), hi = simd_max(ah, bh)
        let mid = (lo + hi) / 2, ext = hi - lo
        let c = cos(shell.yaw), s = sin(shell.yaw)
        var m = keep
        m.center = [mid.x * c - mid.z * s, mid.y, mid.x * s + mid.z * c]
        m.size = [ext.x, ext.y, ext.z]
        m.yaw = shell.yaw
        m.points = a.points + b.points
        if m.label == nil {
            m.label = other.label; m.confidence = other.confidence; m.source = other.source
            m.guesses = other.guesses; m.kind = other.kind
        }
        return m
    }
}
