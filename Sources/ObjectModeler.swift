import Foundation
import simd

/// One solid piece of a modelled object: a tabletop, a leg, a sofa back.
struct ModelPart: Codable, Equatable {
    enum Kind: String, Codable { case box, cylinder }
    var kind: Kind
    var name: String
    /// Center relative to the object's box center, in the box's turned frame (x along size[0],
    /// y up, z along size[2]), m.
    var center: [Float]
    var size: [Float]
    var rgb: [Float]

    var centerVec: SIMD3<Float> { SIMD3(center[0], center[1], center[2]) }
    var sizeVec: SIMD3<Float> { SIMD3(size[0], size[1], size[2]) }
}

/// What a box looks like as a 3D object: modelled from parts, or its piece of the scan.
struct ObjectModel: Codable, Equatable {
    enum Source: String, Codable { case modeled, scanned }
    var source: Source
    var parts: [ModelPart] = []
    /// Its scan piece's file in the scan's objects folder, in the box's frame: the whole object
    /// when scanned, or what stands on a modelled table (screens, lamps).
    var piece: String?
    /// For a modelled object with a piece: the piece starts this high above the floor, m; or,
    /// with `pieceGap`, it is everything except that height band (a tabletop with chairs around
    /// and under it).
    var pieceAbove: Float?
    var pieceGap: [Float]?
    /// Share of the object's scan points the model covers (within 4 cm).
    var fit: Float
    /// The model type it was built as ("table", "sofa", "block", "scan").
    var type: String
    /// The box size it was built for; when the box changes size the object is built again.
    var boxSize: [Float]
}

/// A surface point of the scan with its color.
struct ColorSample {
    var p: SIMD3<Float>
    var n: SIMD3<Float>
    var rgb: SIMD3<Float>
}

enum ColorSampler {
    /// The meshes' vertices averaged into `cell` cubes, with normals and colors (0–1).
    static func sample(_ meshes: [MeshOut], cell: Float = 0.02) -> [ColorSample] {
        var acc: [Int64: (p: SIMD3<Float>, n: SIMD3<Float>, c: SIMD3<Float>, k: Float)] = [:]
        acc.reserveCapacity(meshes.reduce(0) { $0 + $1.pos.count } / 8)
        let inv = 1 / cell
        for m in meshes {
            let hasColor = m.col.count == m.pos.count
            for i in 0..<min(m.pos.count, m.nrm.count) {
                let p = m.pos[i]
                guard p.x.isFinite, p.y.isFinite, p.z.isFinite else { continue }
                let q = SIMD3<Int64>(Int64((p.x * inv).rounded(.down)), Int64((p.y * inv).rounded(.down)),
                                     Int64((p.z * inv).rounded(.down))) &+ (1 << 20)
                let key = (q.x << 42) | (q.y << 21) | q.z
                let c = hasColor ? SIMD3(Float(m.col[i].x), Float(m.col[i].y), Float(m.col[i].z)) / 255 : SIMD3(repeating: 0.6)
                var v = acc[key, default: (.zero, .zero, .zero, 0)]
                v.p += p; v.n += m.nrm[i]; v.c += c; v.k += 1
                acc[key] = v
            }
        }
        var out: [ColorSample] = []
        out.reserveCapacity(acc.count)
        for v in acc.values {
            let nl = simd_length(v.n)
            guard nl > 0.5 * v.k else { continue }
            out.append(ColorSample(p: v.p / v.k, n: v.n / nl, rgb: v.c / v.k))
        }
        return out
    }
}

/// Builds a 3D object for a box from the scan points inside it: a model made of parts for the
/// usual furniture (see `ROOM_MODEL_PLAN.md`), or nil when no model fits well, so the caller uses
/// the scan piece instead.
enum ObjectModeler {
    /// A point inside a box: (x, z) in the box's turned frame, `h` height above the floor.
    struct Pt {
        var x: Float, h: Float, z: Float
        var n: SIMD3<Float>
        var rgb: SIMD3<Float>
    }

    /// Below this share of points within 4 cm of the parts, the model doesn't count.
    static let minFit: Float = 0.55

    /// The model type for a name: what kind of model is tried.
    static func type(_ label: String?) -> String {
        switch BoxCategory.of(label) {
        case .table, .desk: return "table"
        case .chair: return "chair"
        case .sofa: return "sofa"
        case .bed: return "bed"
        case .cabinet, .wardrobe, .appliance, .tv, .radiator, .sink, .toilet: return "block"
        default: return "scan"
        }
    }

    /// The scan points belonging to a box: inside it (grown by 3 cm), above the floor, and not on
    /// a wall.
    static func points(_ b: ObjectBox, samples: [ColorSample], shell: RoomShell) -> [Pt] {
        let c = cos(b.yaw), s = sin(b.yaw)
        let h = SIMD3(b.size[0], b.size[1], b.size[2]) / 2 + SIMD3(0.03, 0.03, 0.03)
        let walls = shell.walls.map { (a: SIMD2($0.start[0], $0.start[1]), e: SIMD2($0.end[0], $0.end[1]),
                                       n: SIMD2($0.normal[0], $0.normal[1])) }
        var out: [Pt] = []
        for sm in samples {
            let d = sm.p - b.centerVec
            guard abs(d.y) <= h.y else { continue }
            let x = d.x * c + d.z * s, z = -d.x * s + d.z * c
            guard abs(x) <= h.x, abs(z) <= h.z else { continue }
            let height = sm.p.y - shell.floorY
            guard height > 0.02 else { continue }
            // Wall surface: close to a wall and facing like it.
            let q = SIMD2(sm.p.x, sm.p.z), nn = SIMD2(sm.n.x, sm.n.z)
            if walls.contains(where: { ObjectFinder.distance(q, $0.a, $0.e) < 0.04 && simd_dot(nn, $0.n) > 0.8 }) { continue }
            let nl = SIMD3(sm.n.x * c + sm.n.z * s, sm.n.y, -sm.n.x * s + sm.n.z * c)
            out.append(Pt(x: x, h: height, z: z, n: nl, rgb: sm.rgb))
        }
        return out
    }

    /// The model for a box from its points, or nil when no model of its type fits. A box without
    /// a name is tried as a table, which only counts when its top is clear (a wrong guess looks
    /// worse than the scan).
    static func model(_ b: ObjectBox, points pts: [Pt], shell: RoomShell) -> ObjectModel? {
        let named = type(b.label)
        let t = b.label == nil ? "table" : named
        guard t != "scan", pts.count >= 30 else { return nil }
        // A box too big for what it is called holds more than that one thing (a sofa with the
        // coffee tables in front of it): modelling it as one would invent a giant sofa.
        if b.label != nil {
            let limit = RoomCorrector.maxSize(BoxCategory.of(b.label))
            let long = max(b.size[0], b.size[2]), short = min(b.size[0], b.size[2])
            // Height doesn't count for tables: what stands on or hangs over one is the scan.
            guard long <= limit.long * 1.15, short <= limit.short * 1.15,
                  t == "table" || b.size[1] <= limit.height * 1.15 else { return nil }
        }
        var parts: [ModelPart]?
        var above: Float?, gap: [Float]?
        switch t {
        case "table":
            if let r = table(pts) { parts = r.parts; above = r.above; gap = r.gap }
            // Not named: only a clear table with its own supports counts.
            if b.label == nil && gap != nil { return nil }
        case "chair": parts = chair(pts)
        case "sofa": parts = sofa(pts)
        case "bed": parts = bed(pts)
        default: parts = block(pts)
        }
        guard let ph = parts, !ph.isEmpty else { return nil }
        // What the scan piece shows doesn't count against the fit.
        var counted = above.map { a in pts.filter { $0.h < a } } ?? pts
        if let g = gap { counted = pts.filter { $0.h >= g[0] && $0.h <= g[1] } }
        let f = fit(counted, ph)
        guard f >= (b.label == nil ? 0.8 : minFit) else { return nil }
        // Parts were built with y as height above the floor; the box frame has y from its center.
        let dy = shell.floorY - b.center[1]
        let local = ph.map { p -> ModelPart in var p = p; p.center[1] += dy; return p }
        return ObjectModel(source: .modeled, parts: local, pieceAbove: gap == nil ? above : nil, pieceGap: gap,
                           fit: f, type: named, boxSize: b.size)
    }

    // MARK: types

    /// Tabletop at the most common up-facing height, legs or pedestals under it where the scan
    /// shows them (four corner legs when none were seen). `above`: what stands on it is left to
    /// the scan from this height (when anything does). `gap`: when what is under the top is too
    /// wide for legs or a drawer unit (chairs around and under it), only the top is modelled and
    /// everything outside this height band is the scan.
    static func table(_ pts: [Pt]) -> (parts: [ModelPart], above: Float?, gap: [Float]?)? {
        guard let level = mode(pts.filter { $0.n.y > 0.8 }.map(\.h), 0.35, 1.2) else { return nil }
        let top = pts.filter { $0.n.y > 0.8 && abs($0.h - level) < 0.03 }
        guard top.count >= 15, let r = rect(top) else { return nil }
        let thick: Float = 0.03
        var parts = [slab("top", r, from: level - thick, to: level, rgb: median(top))]
        let below = pts.filter { $0.h > 0.06 && $0.h < level - 0.12 }
        let (legs, rest) = supportsAndRest(below, under: r, height: level - thick, guessColor: median(top))
        let topArea = r.size.x * r.size.y
        // Chairs or other things around and under it: only the top is modelled, the scan shows the rest.
        if rest >= 30 && Float(rest) > 0.2 * Float(below.count) {
            return (parts, nil, [level - thick - 0.03, level + 0.01])
        }
        if let wide = legs.first(where: { $0.size[0] * $0.size[2] > 0.3 * topArea }) {
            // A closed front from floor to top along most of the length (cupboards under a
            // counter): a body under the top. Gaps (chairs and legs around and under a table):
            // the scan shows it.
            let inside = below.filter { abs($0.x - wide.center[0]) <= wide.size[0] / 2 && abs($0.z - wide.center[2]) <= wide.size[2] / 2 }
            if closedFront(flatFront(inside, along: r.size.x >= r.size.y ? 0 : 2), along: r.size.x >= r.size.y ? 0 : 2,
                           level: level) >= 0.75 {
                parts.append(ModelPart(kind: .box, name: "body", center: wide.center, size: wide.size, rgb: wide.rgb))
                let onTop = pts.filter { $0.h > level + 0.02 }.count
                return (parts, onTop >= 12 ? level + 0.01 : nil, nil)
            }
            return (parts, nil, [level - thick - 0.03, level + 0.01])
        }
        parts += legs
        let onTop = pts.filter { $0.h > level + 0.02 }.count
        return (parts, onTop >= 12 ? level + 0.01 : nil, nil)
    }

    /// Seat at the most common up-facing height (30–65 cm), a back on the side the high points
    /// are, legs under the seat.
    static func chair(_ pts: [Pt]) -> [ModelPart]? {
        guard let level = mode(pts.filter { $0.n.y > 0.75 }.map(\.h), 0.3, 0.65) else { return nil }
        let seat = pts.filter { $0.n.y > 0.75 && abs($0.h - level) < 0.04 }
        guard seat.count >= 8, let r = rect(seat) else { return nil }
        let color = median(seat)
        var parts = [slab("seat", r, from: level - 0.05, to: level, rgb: color)]
        let high = pts.filter { $0.h > level + 0.12 }
        if high.count >= 10, let back = backPanel(high, seat: r, from: level, minDepth: 0.03, maxDepth: 0.15) {
            parts.append(back)
        }
        parts += supports(pts.filter { $0.h > 0.06 && $0.h < level - 0.1 }, under: r, height: level - 0.05, guessColor: color)
        return parts
    }

    /// Base and seat to the seat height (25–60 cm), back on the side the high points are, arms at
    /// the ends.
    static func sofa(_ pts: [Pt]) -> [ModelPart]? {
        guard let level = mode(pts.filter { $0.n.y > 0.75 }.map(\.h), 0.25, 0.6),
              let all = rect(pts) else { return nil }
        let seatPts = pts.filter { $0.n.y > 0.75 && abs($0.h - level) < 0.05 }
        let color = seatPts.count >= 10 ? median(seatPts) : median(pts)
        var parts: [ModelPart] = []
        var seatRect = all
        let high = pts.filter { $0.h > level + 0.15 }
        if high.count >= 15, let back = backPanel(high, seat: all, from: 0, minDepth: 0.12, maxDepth: 0.4) {
            parts.append(back)
            // The seat stops at the back.
            let bc = back.centerVec, bs = back.sizeVec
            if bs.x < bs.z {          // back runs along z: it sits at one x end
                if bc.x < all.mid.x { seatRect.lo.x = bc.x + bs.x / 2 } else { seatRect.hi.x = bc.x - bs.x / 2 }
            } else {
                if bc.z < all.mid.y { seatRect.lo.y = bc.z + bs.z / 2 } else { seatRect.hi.y = bc.z - bs.z / 2 }
            }
            // Arms: high points at the two ends across the back's direction, outside its band.
            let alongX = bs.x >= bs.z
            let rest = high.filter { p in
                alongX ? abs(p.z - bc.z) > bs.z / 2 + 0.03 : abs(p.x - bc.x) > bs.x / 2 + 0.03
            }
            for end in [false, true] {
                let lo = alongX ? seatRect.lo.x : seatRect.lo.y, hi = alongX ? seatRect.hi.x : seatRect.hi.y
                let armPts = rest.filter { p in
                    let v = alongX ? p.x : p.z
                    return end ? v > hi - 0.3 : v < lo + 0.3
                }
                guard armPts.count >= 10 else { continue }
                let vs = armPts.map { alongX ? $0.x : $0.z }
                let width = min(max(end ? hi - percentile(vs, 0.1) : percentile(vs, 0.9) - lo, 0.1), 0.3)
                var ar = seatRect
                if alongX { if end { ar.lo.x = hi - width } else { ar.hi.x = lo + width } } else {
                    if end { ar.lo.y = hi - width } else { ar.hi.y = lo + width }
                }
                parts.append(slab("arm", ar, from: 0, to: percentile(armPts.map(\.h), 0.9), rgb: median(armPts)))
            }
        }
        parts.append(slab("base", seatRect, from: 0, to: level * 0.6, rgb: color * 0.85))
        parts.append(slab("seat", seatRect.inset(0.02), from: level * 0.6, to: level, rgb: color))
        return parts
    }

    /// Frame and mattress to the top height (25–80 cm), headboard on a side if seen.
    static func bed(_ pts: [Pt]) -> [ModelPart]? {
        guard let level = mode(pts.filter { $0.n.y > 0.8 }.map(\.h), 0.25, 0.8) else { return nil }
        let top = pts.filter { $0.n.y > 0.8 && abs($0.h - level) < 0.05 }
        guard top.count >= 20, let r = rect(top) else { return nil }
        let sides = pts.filter { abs($0.n.y) < 0.4 && $0.h < level - 0.12 }
        let mattress = median(top)
        let frame = sides.count >= 10 ? median(sides) : mattress * 0.7
        let split = max(level - 0.2, level * 0.5)
        var parts = [slab("frame", r, from: 0, to: split, rgb: frame),
                     slab("mattress", r.inset(0.02), from: split, to: level, rgb: mattress)]
        let high = pts.filter { $0.h > level + 0.15 }
        if high.count >= 15, let head = backPanel(high, seat: r, from: 0, minDepth: 0.04, maxDepth: 0.15) {
            var h = head; h.name = "headboard"; parts.append(h)
        }
        return parts
    }

    /// One block around the points: on the floor when it reaches near it.
    static func block(_ pts: [Pt]) -> [ModelPart]? {
        guard let r = rect(pts) else { return nil }
        let hs = pts.map(\.h)
        let bottom = percentile(hs, 0.02) < 0.15 ? 0 : percentile(hs, 0.02)
        let top = percentile(hs, 0.98)
        guard top - bottom > 0.02 else { return nil }
        return [slab("body", r, from: bottom, to: top, rgb: median(pts))]
    }

    // MARK: pieces

    /// A floor plan rectangle in the box frame: lo/hi as (x, z).
    struct Rect {
        var lo: SIMD2<Float>, hi: SIMD2<Float>
        var mid: SIMD2<Float> { (lo + hi) / 2 }
        var size: SIMD2<Float> { hi - lo }
        func inset(_ d: Float) -> Rect {
            let e = SIMD2(min(d, size.x / 4), min(d, size.y / 4))
            return Rect(lo: lo + e, hi: hi - e)
        }
    }

    /// The points' floor plan, from the `lo` to the `hi` percentile on each axis (drops strays).
    static func rect(_ pts: [Pt], lo: Float = 0.02, hi: Float = 0.98) -> Rect? {
        guard pts.count >= 3 else { return nil }
        let xs = pts.map(\.x), zs = pts.map(\.z)
        let r = Rect(lo: SIMD2(percentile(xs, lo), percentile(zs, lo)), hi: SIMD2(percentile(xs, hi), percentile(zs, hi)))
        return r.size.x > 0.02 && r.size.y > 0.02 ? r : nil
    }

    static func slab(_ name: String, _ r: Rect, from: Float, to: Float, rgb: SIMD3<Float>, kind: ModelPart.Kind = .box) -> ModelPart {
        let m = r.mid, s = r.size
        return ModelPart(kind: kind, name: name, center: [m.x, (from + to) / 2, m.y],
                         size: [max(s.x, 0.02), max(to - from, 0.01), max(s.y, 0.02)],
                         rgb: [rgbClamp(rgb.x), rgbClamp(rgb.y), rgbClamp(rgb.z)])
    }

    /// Legs and pedestals under a top: groups of points under it seen from above. Thin groups are
    /// legs, wide ones a pedestal or drawer unit. Four corner legs when nothing was seen.
    static func supports(_ below: [Pt], under r: Rect, height: Float, guessColor: SIMD3<Float>) -> [ModelPart] {
        supportsAndRest(below, under: r, height: height, guessColor: guessColor).parts
    }

    /// Legs and pedestals under a top, and how many points under or around it are something else
    /// (chairs): a support reaches up to the top and stands mostly under it.
    static func supportsAndRest(_ below: [Pt], under r: Rect, height: Float, guessColor: SIMD3<Float>) -> (parts: [ModelPart], rest: Int) {
        let grown = Rect(lo: r.lo - 0.05, hi: r.hi + 0.05)
        func within(_ p: Pt, _ q: Rect) -> Bool { p.x >= q.lo.x && p.x <= q.hi.x && p.z >= q.lo.y && p.z <= q.hi.y }
        var out: [ModelPart] = []
        var rest = below.filter { !within($0, grown) }.count
        for comp in clusters(below.filter { within($0, grown) }, cell: 0.04) where comp.count >= 4 {
            let reaches = percentile(comp.map(\.h), 0.95) >= height - 0.2
            let under = Float(comp.filter { within($0, r) }.count) / Float(comp.count) >= 0.8
            guard reaches && under else { rest += comp.count; continue }
            let cr = bounds(comp)
            let thin = max(cr.size.x, cr.size.y) <= 0.2
            var lr = cr
            if thin {
                // Legs at least 3 cm thick; keep their middle.
                let m = cr.mid, half = simd_max(cr.size, SIMD2(repeating: 0.03)) / 2
                lr = Rect(lo: m - half, hi: m + half)
            }
            out.append(slab(thin ? "leg" : "pedestal", lr, from: 0, to: height, rgb: median(comp),
                            kind: thin && abs(cr.size.x - cr.size.y) < 0.03 ? .cylinder : .box))
        }
        if out.isEmpty {
            let inset: Float = 0.04, leg: Float = 0.04
            for (x, z) in [(r.lo.x + inset, r.lo.y + inset), (r.hi.x - inset, r.lo.y + inset),
                           (r.hi.x - inset, r.hi.y - inset), (r.lo.x + inset, r.hi.y - inset)] {
                let m = SIMD2<Float>(x, z)
                out.append(slab("leg (not seen)", Rect(lo: m - leg / 2, hi: m + leg / 2), from: 0, to: height,
                                rgb: guessColor * 0.8))
            }
        }
        return (out, rest)
    }

    /// The points' full floor plan extent (may be a line).
    static func bounds(_ pts: [Pt]) -> Rect {
        var lo = SIMD2<Float>(repeating: .infinity), hi = SIMD2<Float>(repeating: -.infinity)
        for p in pts { lo = simd_min(lo, SIMD2(p.x, p.z)); hi = simd_max(hi, SIMD2(p.x, p.z)) }
        return Rect(lo: lo, hi: hi)
    }

    /// A back panel on the side of a seat where the high points are: which side from where their
    /// middle lies, how thick from how far they reach in.
    static func backPanel(_ high: [Pt], seat r: Rect, from: Float, minDepth: Float, maxDepth: Float) -> ModelPart? {
        let cx = high.map(\.x).reduce(0, +) / Float(high.count), cz = high.map(\.z).reduce(0, +) / Float(high.count)
        let ox = (cx - r.mid.x) / max(r.size.x / 2, 0.05), oz = (cz - r.mid.y) / max(r.size.y / 2, 0.05)
        guard max(abs(ox), abs(oz)) > 0.25 else { return nil }       // high points all over: no clear back
        let top = percentile(high.map(\.h), 0.95)
        var b = r
        if abs(ox) >= abs(oz) {
            let xs = high.map(\.x)
            if ox > 0 { let d = min(max(r.hi.x - percentile(xs, 0.1), minDepth), maxDepth); b.lo.x = r.hi.x - d }
            else { let d = min(max(percentile(xs, 0.9) - r.lo.x, minDepth), maxDepth); b.hi.x = r.lo.x + d }
        } else {
            let zs = high.map(\.z)
            if oz > 0 { let d = min(max(r.hi.y - percentile(zs, 0.1), minDepth), maxDepth); b.lo.y = r.hi.y - d }
            else { let d = min(max(percentile(zs, 0.9) - r.lo.y, minDepth), maxDepth); b.hi.y = r.lo.y + d }
        }
        return slab("back", b, from: from, to: top, rgb: median(high))
    }

    // MARK: helpers

    /// Most common height in [lo, hi] (2 cm bins, 3-bin window), if at least 8 points share it.
    static func mode(_ hs: [Float], _ lo: Float, _ hi: Float, bin: Float = 0.02) -> Float? {
        let n = Int((hi - lo) / bin) + 1
        var count = [Int](repeating: 0, count: n)
        for h in hs where h >= lo && h <= hi { count[min(n - 1, Int((h - lo) / bin))] += 1 }
        var best = (c: 0, i: 0)
        for i in 0..<n {
            let c = count[max(0, i - 1)...min(n - 1, i + 1)].reduce(0, +)
            if c > best.c { best = (c, i) }
        }
        guard best.c >= 8 else { return nil }
        // Weighted middle of the window.
        var sum: Float = 0, w: Float = 0
        for i in max(0, best.i - 1)...min(n - 1, best.i + 1) { sum += Float(count[i]) * (lo + (Float(i) + 0.5) * bin); w += Float(count[i]) }
        return sum / max(w, 1)
    }

    static func percentile(_ v: [Float], _ q: Float) -> Float {
        guard !v.isEmpty else { return 0 }
        let s = v.sorted()
        return s[min(s.count - 1, max(0, Int(q * Float(s.count - 1))))]
    }

    static func median(_ pts: [Pt]) -> SIMD3<Float> {
        guard !pts.isEmpty else { return SIMD3(repeating: 0.6) }
        return SIMD3(percentile(pts.map(\.rgb.x), 0.5), percentile(pts.map(\.rgb.y), 0.5), percentile(pts.map(\.rgb.z), 0.5))
    }

    private static func rgbClamp(_ v: Float) -> Float { min(1, max(0, v)) }

    /// The points on the flattest upright face running along an axis (0: x, 2: z): facing
    /// across it, within 4 cm of the face's middle. A cupboard front, not chair legs.
    static func flatFront(_ pts: [Pt], along axis: Int) -> [Pt] {
        func across(_ p: Pt) -> Float { axis == 0 ? p.z : p.x }
        func facing(_ p: Pt) -> Float { axis == 0 ? p.n.z : p.n.x }
        var best: [Pt] = []
        for sign: Float in [1, -1] {
            let face = pts.filter { abs($0.n.y) < 0.3 && facing($0) * sign > 0.8 }
            guard !face.isEmpty else { continue }
            let mid = percentile(face.map(across), 0.5)
            let on = face.filter { abs(across($0) - mid) < 0.04 }
            if on.count > best.count { best = on }
        }
        return best
    }

    /// Share of 5 cm slices along an axis (0: x, 2: z) whose points reach from near the floor
    /// (under 30 cm) to near the top (within 30 cm of `level`): a closed cupboard front.
    static func closedFront(_ pts: [Pt], along axis: Int, level: Float) -> Float {
        guard let r = rect(pts, lo: 0, hi: 1) else { return 0 }
        let lo = axis == 0 ? r.lo.x : r.lo.y, len = axis == 0 ? r.size.x : r.size.y
        let n = max(1, Int(len / 0.05))
        var low = [Bool](repeating: false, count: n), high = [Bool](repeating: false, count: n)
        for p in pts {
            let i = min(n - 1, max(0, Int(((axis == 0 ? p.x : p.z) - lo) / 0.05)))
            if p.h < 0.3 { low[i] = true }
            if p.h > level - 0.3 { high[i] = true }
        }
        return Float((0..<n).filter { low[$0] && high[$0] }.count) / Float(n)
    }

    /// Share of the `cell` grid over the points' floor plan that holds any of them.
    static func filled(_ pts: [Pt], cell: Float) -> Float {
        guard let r = rect(pts, lo: 0, hi: 1) else { return 0 }
        var cells = Set<SIMD2<Int32>>()
        for p in pts { cells.insert(SIMD2(Int32(((p.x - r.lo.x) / cell).rounded(.down)), Int32(((p.z - r.lo.y) / cell).rounded(.down)))) }
        let total = (Int(r.size.x / cell) + 1) * (Int(r.size.y / cell) + 1)
        return Float(cells.count) / Float(max(total, 1))
    }

    /// Groups of points that touch seen from above (`cell` grid, 8-neighbours).
    static func clusters(_ pts: [Pt], cell: Float) -> [[Pt]] {
        guard !pts.isEmpty else { return [] }
        var cells: [SIMD2<Int32>: [Int]] = [:]
        for (i, p) in pts.enumerated() {
            cells[SIMD2(Int32((p.x / cell).rounded(.down)), Int32((p.z / cell).rounded(.down))), default: []].append(i)
        }
        var seen = Set<SIMD2<Int32>>()
        var out: [[Pt]] = []
        for start in cells.keys where !seen.contains(start) {
            var stack = [start], comp: [Pt] = []
            seen.insert(start)
            while let k = stack.popLast() {
                comp += cells[k]!.map { pts[$0] }
                for dx: Int32 in -1...1 { for dz: Int32 in -1...1 {
                    let nk = SIMD2(k.x + dx, k.y + dz)
                    if cells[nk] != nil, !seen.contains(nk) { seen.insert(nk); stack.append(nk) }
                } }
            }
            out.append(comp)
        }
        return out
    }

    /// Share of points within 4 cm of a part (parts as boxes, in height-above-floor frame).
    static func fit(_ pts: [Pt], _ parts: [ModelPart]) -> Float {
        guard !pts.isEmpty else { return 0 }
        var near = 0
        for p in pts {
            let q = SIMD3(p.x, p.h, p.z)
            for part in parts {
                let d = simd_max(simd_abs(q - part.centerVec) - part.sizeVec / 2, .zero)
                if simd_length(d) <= 0.04 { near += 1; break }
            }
        }
        return Float(near) / Float(pts.count)
    }
}

/// The piece of the scan mesh that belongs to a box, cut out, cleaned and simplified: for objects
/// without a fitting model.
enum ScanPiece {
    /// Triangles with all corners inside the box (grown by 2 cm), above the floor and off the
    /// walls, merged onto a `cell` grid. In the box's frame, with the scan's colors. Nil when
    /// nothing is left.
    /// `above`: only what is higher than this above the floor (things on a table). `gap`: leave
    /// out this band of heights above the floor (a modelled tabletop).
    static func cut(_ meshes: [MeshOut], box b: ObjectBox, shell: RoomShell, above: Float? = nil, gap: [Float]? = nil,
                    cell: Float = 0.015) -> MeshOut? {
        let c = cos(b.yaw), s = sin(b.yaw)
        let h = SIMD3(b.size[0], b.size[1], b.size[2]) / 2 + 0.02
        let floorLocal = shell.floorY + (above ?? 0.015) - b.center[1]
        let walls = shell.walls.map { (a: SIMD2($0.start[0], $0.start[1]), e: SIMD2($0.end[0], $0.end[1]),
                                       n: SIMD2($0.normal[0], $0.normal[1])) }
        var out = MeshOut()
        var clusterOf: [Int64: UInt32] = [:]
        var sums: [(p: SIMD3<Float>, n: SIMD3<Float>, c: SIMD3<Float>, k: Float)] = []
        let inv = 1 / cell
        for m in meshes {
            let hasColor = m.col.count == m.pos.count
            // Local position of each vertex, or nil when it is outside or on floor or wall.
            var local = [Int32](repeating: -2, count: m.pos.count)    // -2 unknown, -1 out, else cluster
            func vertex(_ i: Int) -> Int32 {
                if local[i] != -2 { return local[i] }
                let p = m.pos[i], d = p - b.centerVec
                let q = SIMD3(d.x * c + d.z * s, d.y, -d.x * s + d.z * c)
                var keep = abs(q.x) <= h.x && abs(q.y) <= h.y && abs(q.z) <= h.z && q.y > floorLocal
                if let gap, keep {
                    let height = p.y - shell.floorY
                    if height >= gap[0], height <= gap[1] { keep = false }
                }
                if keep {
                    let xz = SIMD2(p.x, p.z), nn = SIMD2(m.nrm[i].x, m.nrm[i].z)
                    if walls.contains(where: { ObjectFinder.distance(xz, $0.a, $0.e) < 0.03 && simd_dot(nn, $0.n) > 0.8 }) { keep = false }
                }
                guard keep else { local[i] = -1; return -1 }
                let g = SIMD3<Int64>(Int64((q.x * inv).rounded(.down)), Int64((q.y * inv).rounded(.down)),
                                     Int64((q.z * inv).rounded(.down))) &+ (1 << 20)
                let key = (g.x << 42) | (g.y << 21) | g.z
                let nl = SIMD3(m.nrm[i].x * c + m.nrm[i].z * s, m.nrm[i].y, -m.nrm[i].x * s + m.nrm[i].z * c)
                let col = hasColor ? SIMD3(Float(m.col[i].x), Float(m.col[i].y), Float(m.col[i].z)) : SIMD3(repeating: 150)
                if let k = clusterOf[key] {
                    sums[Int(k)].p += q; sums[Int(k)].n += nl; sums[Int(k)].c += col; sums[Int(k)].k += 1
                    local[i] = Int32(k)
                } else {
                    let k = UInt32(sums.count)
                    clusterOf[key] = k
                    sums.append((q, nl, col, 1))
                    local[i] = Int32(k)
                }
                return local[i]
            }
            var t = 0
            while t + 2 < m.idx.count {
                let a = vertex(Int(m.idx[t])), bb = vertex(Int(m.idx[t + 1])), cc = vertex(Int(m.idx[t + 2]))
                t += 3
                guard a >= 0, bb >= 0, cc >= 0, a != bb, bb != cc, a != cc else { continue }
                out.idx += [UInt32(a), UInt32(bb), UInt32(cc)]
            }
        }
        guard !out.idx.isEmpty else { return nil }
        out.pos = sums.map { $0.p / $0.k }
        out.nrm = sums.map { simd_length($0.n) > 0 ? simd_normalize($0.n) : SIMD3(0, 1, 0) }
        out.col = sums.map { v in
            let c = v.c / v.k
            return SIMD4(UInt8(min(255, c.x)), UInt8(min(255, c.y)), UInt8(min(255, c.z)), 255)
        }
        return out
    }
}
