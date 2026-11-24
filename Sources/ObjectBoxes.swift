import Foundation
import simd

/// An upright box around one thing in the room, turned with the room's walls.
struct ObjectBox: Codable, Equatable, Identifiable {
    var id: Int
    /// World center [x, y, z].
    var center: [Float]
    /// Size along the room's first axis, height, and along the second axis, in meters.
    var size: [Float]
    /// Same as the room's yaw: radians from world +x toward +z.
    var yaw: Float
    /// Number of 3 cm surface samples inside.
    var points: Int
    var label: String?
    var confidence: Float?
    /// "vision" (image classifier) or "model" (on-device language model).
    var source: String?
    /// Best image-classifier guesses, for debugging.
    var guesses: [String]?
    /// What the on-device model says this is: "furniture", "object", "wall", "floor", "ceiling",
    /// "door", "window" or "unclear". On removed boxes, why it was removed ("deleted" by hand).
    var kind: String?

    var centerVec: SIMD3<Float> { SIMD3(center[0], center[1], center[2]) }

    /// The 8 corners in world coordinates: bottom four first, then the top four.
    var corners: [SIMD3<Float>] {
        let c = cos(yaw), s = sin(yaw)
        let u = SIMD3<Float>(c, 0, s), v = SIMD3<Float>(-s, 0, c), up = SIMD3<Float>(0, 1, 0)
        let h = SIMD3(size[0], size[1], size[2]) / 2
        var out: [SIMD3<Float>] = []
        for dy: Float in [-1, 1] {
            for (du, dv): (Float, Float) in [(-1, -1), (1, -1), (1, 1), (-1, 1)] {
                out.append(centerVec + u * (du * h.x) + up * (dy * h.y) + v * (dv * h.z))
            }
        }
        return out
    }

    /// World-axis bounds, used to match boxes between updates.
    var bounds: (lo: SIMD3<Float>, hi: SIMD3<Float>) {
        let cs = corners
        return (cs.reduce(cs[0]) { simd_min($0, $1) }, cs.reduce(cs[0]) { simd_max($0, $1) })
    }

    var volume: Float { size[0] * size[1] * size[2] }
}

/// Finds the things in a room: surface points that are not floor, ceiling or wall, grouped
/// into connected clusters, each wrapped in an upright box.
enum ObjectFinder {
    /// Raised when box finding changes enough that saved rooms are worth finding again.
    /// 2: wall bands, ceiling structures, local floor and ceiling, divider barriers.
    static let version = 2

    struct Options {
        var cell: Float = 0.05
        var minVoxels = 10
        var minPoints = 60
        /// Largest side must be at least this, in meters.
        var minSize: Float = 0.12
        var floorMargin: Float = 0.04
        var ceilingMargin: Float = 0.08
        var wallMargin: Float = 0.08
        /// Surfaces facing exactly like a wall this close to it count as wall too, m.
        var wallBand: Float = 0.25
        /// Clusters with a floor plan at least this big are split at tabletop height, m².
        var splitMinArea: Float = 0.8
        /// Height range of desk, table and counter tops above the floor, m.
        var tableLow: Float = 0.65
        var tableHigh: Float = 1.15
        /// Least area of a tabletop, m², and of the low part (seat, legs) of a thing standing next
        /// to one for it to be its own box.
        var minTableArea: Float = 0.15
        var minStandingArea: Float = 0.05
        /// Columns this close to a wall that reach this high are a tall thing along the wall
        /// (cabinets to the ceiling, curtains, a wardrobe), kept apart from what stands in front, m.
        var wallStripDepth: Float = 0.7
        var wallStripHeight: Float = 1.6
        /// Big things starting this high above the floor are ceiling: a lowered part, beams,
        /// ducts or light panels, m. "Big" is a floor plan of at least `ceilingStructureArea`, m².
        var ceilingStructureLow: Float = 2.0
        var ceilingStructureArea: Float = 0.8
    }

    /// A kept surface point in room coordinates (u, height, v).
    struct Pt {
        var u: Float, y: Float, v: Float
        var wallLike: Bool
        /// Faces up: a top surface.
        var up: Bool
    }

    static func find(_ samples: [RoomFitter.Sample], shell: RoomShell, options o: Options = Options()) -> [ObjectBox] {
        let c = cos(shell.yaw), s = sin(shell.yaw)
        func toRoom(_ x: Float, _ z: Float) -> SIMD2<Float> { SIMD2(x * c + z * s, -x * s + z * c) }
        let poly = shell.outline.map { toRoom($0[0], $0[1]) }
        let wallNormals = shell.walls.map { toRoom($0.normal[0], $0.normal[1]) }

        // Keep points inside the room that are not on the floor, ceiling or a wall.
        // `wallLike`: within 40 cm of a wall and facing the same way, for spotting wall pieces
        // that fall outside the margin (a slightly crooked wall).
        let floor = LocalLevel(samples, shell: shell, toRoom: toRoom, floor: true)
        let ceiling = LocalLevel(samples, shell: shell, toRoom: toRoom, floor: false)
        var pts: [Pt] = []
        pts.reserveCapacity(samples.count / 2)
        for sm in samples {
            let y = sm.p.y
            let q = toRoom(sm.p.x, sm.p.z)
            guard y > floor.at(q) + o.floorMargin, y < ceiling.at(q) - o.ceilingMargin else { continue }
            guard contains(poly, q) else { continue }
            let n = toRoom(sm.n.x, sm.n.z)
            var onWall = false, wallLike = false
            for i in 0..<poly.count {
                let d = distance(q, poly[i], poly[(i + 1) % poly.count])
                guard d < 0.4 else { continue }
                let facing = simd_dot(n, wallNormals[i])
                // Near a wall: drop it when it faces like the wall (paint, frames), keep it otherwise.
                if d < o.wallMargin && facing > 0.7 { onWall = true; break }
                // Further out, only surfaces facing squarely like the wall: an uneven or slightly
                // crooked wall, radiators, sills. Their thin band would chain everything along the
                // wall into one box.
                if d < o.wallBand && facing > 0.9 { onWall = true; break }
                if facing > 0.94 { wallLike = true }
            }
            if !onWall { pts.append(Pt(u: q.x, y: y, v: q.y, wallLike: wallLike, up: sm.n.y > 0.8)) }
        }

        // Voxelize and group touching voxels (26-neighbours).
        let inv = 1 / o.cell
        func key(_ i: Int, _ j: Int, _ k: Int) -> Int64 {
            (Int64(i + (1 << 20)) << 42) | (Int64(j + (1 << 20)) << 21) | Int64(k + (1 << 20))
        }
        var voxelOf = [Int64](repeating: 0, count: pts.count)
        var voxels: [Int64: SIMD3<Int32>] = [:]
        for (n, p) in pts.enumerated() {
            let i = Int((p.u * inv).rounded(.down)), j = Int((p.y * inv).rounded(.down)), k = Int((p.v * inv).rounded(.down))
            let kk = key(i, j, k)
            voxelOf[n] = kk
            voxels[kk] = SIMD3(Int32(i), Int32(j), Int32(k))
        }
        var label: [Int64: Int] = [:]
        var clusterSizes: [Int] = []
        for (start, _) in voxels where label[start] == nil {
            let id = clusterSizes.count
            var size = 0
            var stack = [start]
            label[start] = id
            while let kk = stack.popLast() {
                size += 1
                let v = voxels[kk]!
                for di in -1...1 { for dj in -1...1 { for dk in -1...1 {
                    let nk = key(Int(v.x) + di, Int(v.y) + dj, Int(v.z) + dk)
                    if voxels[nk] != nil, label[nk] == nil { label[nk] = id; stack.append(nk) }
                } } }
            }
            clusterSizes.append(size)
        }

        // Points of each big enough cluster; big ones are split at tabletop height.
        var members = [[Int]](repeating: [], count: clusterSizes.count)
        for n in pts.indices { members[label[voxelOf[n]]!].append(n) }
        var groups: [(idx: [Int], table: Bool)] = []
        for id in clusterSizes.indices where clusterSizes[id] >= o.minVoxels && members[id].count >= o.minPoints {
            for part in splitAtWalls(members[id], pts, poly: poly, floorY: shell.floorY, o: o) {
                groups += splitAtTables(part, pts, floorY: shell.floorY, o: o)
            }
        }

        // One box per group, from its points' extent.
        var boxes: [ObjectBox] = []
        var floatingTables: [(index: Int, bottom: Float)] = []
        for g in groups where g.idx.count >= o.minPoints {
            var l = SIMD3<Float>(repeating: .infinity), h = SIMD3<Float>(repeating: -.infinity)
            var wallLike = 0
            for n in g.idx {
                let p = pts[n], q = SIMD3(p.u, p.y, p.v)
                l = simd_min(l, q); h = simd_max(h, q)
                if p.wallLike { wallLike += 1 }
            }
            let count = g.idx.count
            // A flat piece nearly as tall as the room, facing like the wall beside it, is wall.
            if h.y - l.y > 0.8 * shell.height, Float(wallLike) > 0.6 * Float(count) { continue }
            // Big things high up are part of the ceiling (wall cabinets start lower).
            if l.y >= shell.floorY + o.ceilingStructureLow, (h.x - l.x) * (h.z - l.z) >= o.ceilingStructureArea { continue }
            // Small flat bits hugging the ceiling are lights, vents or noise, not furniture.
            if h.y > shell.ceilingY - 0.3, h.y - l.y < 0.2, count < 400 { continue }
            // Flat leftovers just above the floor are rugs or floor noise.
            if l.y < shell.floorY + 0.1, h.y - l.y < 0.05 { continue }
            // Things that reach down near the floor stand on it, and so do tables seen without legs
            // (unless something found separately stands under them: see below).
            let floating = l.y >= shell.floorY + 0.15, bottom = l.y
            if !floating || g.table { l.y = shell.floorY }
            if floating && g.table { floatingTables.append((boxes.count, bottom)) }
            var ext = h - l
            guard max(ext.x, ext.y, ext.z) >= o.minSize else { continue }
            ext.y = max(ext.y, 0.03)     // a shelf or tabletop seen only from above
            let mid = SIMD3((l.x + h.x) / 2, l.y + ext.y / 2, (l.z + h.z) / 2)
            let world = SIMD3(mid.x * c - mid.z * s, mid.y, mid.x * s + mid.z * c)
            boxes.append(ObjectBox(id: boxes.count, center: [world.x, world.y, world.z],
                                   size: [max(ext.x, 0.02), ext.y, max(ext.z, 0.02)], yaw: shell.yaw, points: count))
        }
        joinTablesToBases(&boxes, floatingTables, shell: shell)
        // A box that lies almost wholly inside a bigger one is a piece of it (a cabinet end panel
        // found apart from the front).
        boxes.sort { $0.volume > $1.volume }
        let rb = boxes.map { RoomCorrector.roomBounds($0, shell: shell) }
        var keep = [Bool](repeating: true, count: boxes.count)
        for j in boxes.indices {
            let v = rb[j].hi - rb[j].lo, vol = max(v.x * v.y * v.z, 1e-6)
            for i in 0..<j where keep[i] {
                let e = simd_max(simd_min(rb[i].hi, rb[j].hi) - simd_max(rb[i].lo, rb[j].lo), .zero)
                if e.x * e.y * e.z > 0.8 * vol { keep[j] = false; boxes[i].points += boxes[j].points; break }
            }
        }
        return boxes.indices.filter { keep[$0] }.map { boxes[$0] }
    }

    /// A tabletop found apart from what stands under it (a gap between the top and its drawers or
    /// frame) was extended to the floor; join it with the piece under it instead of overlapping it.
    static func joinTablesToBases(_ boxes: inout [ObjectBox], _ tables: [(index: Int, bottom: Float)], shell: RoomShell) {
        var gone = Set<Int>()
        for t in tables {
            let (tl, th) = RoomCorrector.roomBounds(boxes[t.index], shell: shell)
            for j in boxes.indices where j != t.index && !gone.contains(j) && !tables.contains(where: { $0.index == j }) {
                let (bl, bh) = RoomCorrector.roomBounds(boxes[j], shell: shell)
                guard bh.y <= t.bottom + 0.05, bh.y >= t.bottom - 0.15 else { continue }
                let ou = max(0, min(th.x, bh.x) - max(tl.x, bl.x)), ov = max(0, min(th.z, bh.z) - max(tl.z, bl.z))
                let under = ou * ov / max(min((th.x - tl.x) * (th.z - tl.z), (bh.x - bl.x) * (bh.z - bl.z)), 1e-6)
                guard under > 0.5 else { continue }
                let lo = simd_min(tl, bl), hi = simd_max(th, bh)
                let mid = (lo + hi) / 2, c = cos(shell.yaw), s = sin(shell.yaw)
                boxes[t.index].center = [mid.x * c - mid.z * s, mid.y, mid.x * s + mid.z * c]
                boxes[t.index].size = [hi.x - lo.x, hi.y - lo.y, hi.z - lo.z]
                boxes[t.index].points += boxes[j].points
                gone.insert(j)
            }
        }
        boxes = boxes.enumerated().filter { !gone.contains($0.offset) }.map(\.element)
    }

    /// Splits off tall things along the walls from a big cluster, from above: kitchen cabinets up
    /// to the ceiling, curtains or a wardrobe, which otherwise chain the sofa, table and chairs in
    /// front of them into one box.
    /// - Columns within `wallStripDepth` of a wall that reach above `wallStripHeight` are "along
    ///   the wall", grown over touching tall columns up to 30 cm further out.
    /// - Touching ones form one group; a group that runs along two walls (cabinets on one,
    ///   curtains on the next) is cut at the corner.
    /// - The rest is grouped by what still touches.
    /// - Small tall bits hugging a wall, spread thinly along it, are leftover wall surface (a
    ///   crooked patch, a frame) and are dropped. A cabinet front seen only from the front is as
    ///   thin but stands further out; a slim bookcase or a floor lamp by the wall is a compact
    ///   block and stays. Other tiny leftovers join the group they border most.
    static func splitAtWalls(_ idx: [Int], _ pts: [Pt], poly: [SIMD2<Float>], floorY: Float, o: Options) -> [[Int]] {
        let cell: Float = 0.05
        var lo = SIMD2<Float>(repeating: .infinity), hi = SIMD2<Float>(repeating: -.infinity)
        for n in idx { let q = SIMD2(pts[n].u, pts[n].v); lo = simd_min(lo, q); hi = simd_max(hi, q) }
        let W = Int((hi.x - lo.x) / cell) + 1, H = Int((hi.y - lo.y) / cell) + 1
        let cellArea = cell * cell
        guard poly.count >= 3, Float(W * H) * cellArea >= o.splitMinArea, W * H < 400_000 else { return [idx] }
        func cellOf(_ n: Int) -> Int {
            min(W - 1, Int((pts[n].u - lo.x) / cell)) + min(H - 1, Int((pts[n].v - lo.y) / cell)) * W
        }
        func center(_ k: Int) -> SIMD2<Float> { lo + SIMD2(Float(k % W) + 0.5, Float(k / W) + 0.5) * cell }
        func neighbours(_ k: Int, diagonal: Bool) -> [Int] {
            let x = k % W, y = k / W
            var out: [Int] = []
            for dy in -1...1 { for dx in -1...1 where (dx != 0 || dy != 0) && (diagonal || dx == 0 || dy == 0) {
                let nx = x + dx, ny = y + dy
                if nx >= 0, ny >= 0, nx < W, ny < H { out.append(ny * W + nx) }
            } }
            return out
        }
        func nearest(_ q: SIMD2<Float>, among edges: [Int]? = nil) -> (d: Float, edge: Int) {
            var best = (d: Float.infinity, edge: -1)
            for i in edges ?? Array(0..<poly.count) {
                let d = distance(q, poly[i], poly[(i + 1) % poly.count])
                if d < best.d { best = (d, i) }
            }
            return best
        }
        var top = [Float](repeating: -.infinity, count: W * H)
        for n in idx { let k = cellOf(n); top[k] = max(top[k], pts[n].y - floorY) }
        // Along the wall (1) or not (0); -1 for empty cells.
        var along = [Int8](repeating: -1, count: W * H)
        var wall = [Int](repeating: -1, count: W * H), wallDist = [Float](repeating: 0, count: W * H)
        for k in 0..<(W * H) where top[k] > -.infinity {
            along[k] = 0
            guard top[k] >= o.wallStripHeight else { continue }
            let nb = nearest(center(k))
            if nb.d < o.wallStripDepth { along[k] = 1; wall[k] = nb.edge; wallDist[k] = nb.d }
        }
        var front = (0..<(W * H)).filter { along[$0] == 1 }
        guard !front.isEmpty else { return [idx] }
        while !front.isEmpty {
            var next: [Int] = []
            for k in front {
                for nk in neighbours(k, diagonal: false) where along[nk] == 0 && top[nk] >= o.wallStripHeight {
                    let e = wall[k], d = distance(center(nk), poly[e], poly[(e + 1) % poly.count])
                    guard d < o.wallStripDepth + 0.3 else { continue }
                    along[nk] = 1; wall[nk] = e; wallDist[nk] = d; next.append(nk)
                }
            }
            front = next
        }
        // One ring more of anything above 1 m: a cell holding only a few points of a tall front
        // looks lower than it is. Sofas and beds stay below that.
        for k in (0..<(W * H)).filter({ along[$0] == 0 && top[$0] >= 1.0 }) {
            if let s = neighbours(k, diagonal: false).first(where: { along[$0] == 1 && wall[$0] >= 0 }) {
                let d = distance(center(k), poly[wall[s]], poly[(wall[s] + 1) % poly.count])
                guard d < o.wallStripDepth + 0.3 else { continue }
                along[k] = 2; wall[k] = wall[s]; wallDist[k] = d
            }
        }
        for k in 0..<(W * H) where along[k] == 2 { along[k] = 1 }
        let rest = along.filter { $0 == 0 }.count
        guard Float(rest) * cellArea >= 0.1 else { return [idx] }

        // Groups: touching cells of the same kind (8-connected: a thin end panel meets a thin
        // front only at a corner).
        var group = [Int](repeating: -1, count: W * H)
        var sizes: [Int] = []
        for start in 0..<(W * H) where along[start] >= 0 && group[start] < 0 {
            let g = sizes.count
            var stack = [start], comp: [Int] = []
            group[start] = g
            while let k = stack.popLast() {
                comp.append(k)
                for nk in neighbours(k, diagonal: true) where along[nk] == along[k] && group[nk] < 0 {
                    group[nk] = g; stack.append(nk)
                }
            }
            sizes.append(comp.count)
            guard along[start] == 1 else { continue }
            // Along two walls: cut at the corner, each cell to the nearer of the walls it mostly runs along.
            let minStrip = Int(0.2 / cellArea)
            var perWall: [Int: Int] = [:]
            for k in comp { perWall[wall[k], default: 0] += 1 }
            let major = perWall.filter { $0.value >= minStrip }.map(\.key).sorted()
            guard major.count > 1 else { continue }
            var sub: [Int: Int] = [major[0]: g]
            for e in major.dropFirst() { sub[e] = sizes.count; sizes.append(0) }
            sizes[g] = 0
            for k in comp {
                let e = nearest(center(k), among: major).edge
                group[k] = sub[e]!; sizes[sub[e]!] += 1
            }
        }
        guard sizes.count > 1 else { return [idx] }

        let tiny = Int(0.03 / cellArea), minStrip = Int(0.2 / cellArea)
        var target = Array(0..<sizes.count)
        var stripCells = [Int](repeating: 0, count: sizes.count), stripDist = [Float](repeating: 0, count: sizes.count)
        var gLo = [SIMD2<Int>](repeating: SIMD2(Int.max, Int.max), count: sizes.count)
        var gHi = [SIMD2<Int>](repeating: SIMD2(Int.min, Int.min), count: sizes.count)
        for k in 0..<(W * H) where along[k] == 1 {
            let g = group[k], c = SIMD2(k % W, k / W)
            stripCells[g] += 1; stripDist[g] += wallDist[k]
            gLo[g] = simd_min(gLo[g], c); gHi[g] = simd_max(gHi[g], c)
        }
        for g in sizes.indices where stripCells[g] > 0 && sizes[g] < minStrip
            && stripDist[g] / Float(stripCells[g]) < o.wallBand {
            let ext = gHi[g] &- gLo[g] &+ 1
            let compact = Float(stripCells[g]) / Float(ext.x * ext.y) >= 0.5 && Float(max(ext.x, ext.y)) * cell <= 1.2
            if !compact { target[g] = -1 }
        }
        for g in sizes.indices where sizes[g] > 0 && sizes[g] < tiny && target[g] >= 0 {
            var border: [Int: Int] = [:]
            for k in 0..<(W * H) where group[k] == g {
                for nk in neighbours(k, diagonal: true) {
                    let h = group[nk]
                    if h >= 0, h != g, sizes[h] >= tiny, target[h] >= 0 { border[h, default: 0] += 1 }
                }
            }
            if let b = border.max(by: { $0.value < $1.value }) { target[g] = b.key }
        }
        var out = [[Int]](repeating: [], count: sizes.count)
        for n in idx { let t = target[group[cellOf(n)]]; if t >= 0 { out[t].append(n) } }
        return out.filter { !$0.isEmpty }
    }

    /// Splits a cluster of touching things at tabletop height, from above:
    /// - the tabletop level is the height (0.65–1.15 m) with the most up-facing surface;
    /// - each separate patch of tabletop is one table or desk, with what stands on it (screens,
    ///   lamps, a divider) and whatever is under it;
    /// - next to the tabletops, anything with enough of its own low part (a chair seat, a bin)
    ///   is its own thing.
    /// Returns the cluster unchanged when it has no tabletop. `table` marks tabletop groups,
    /// which stand on the floor even when their legs weren't scanned.
    static func splitAtTables(_ idx: [Int], _ pts: [Pt], floorY: Float, o: Options) -> [(idx: [Int], table: Bool)] {
        let cell: Float = 0.05
        var lo = SIMD2<Float>(repeating: .infinity), hi = SIMD2<Float>(repeating: -.infinity)
        for n in idx { let q = SIMD2(pts[n].u, pts[n].v); lo = simd_min(lo, q); hi = simd_max(hi, q) }
        let W = Int((hi.x - lo.x) / cell) + 1, H = Int((hi.y - lo.y) / cell) + 1
        let cellArea = cell * cell
        let whole = [(idx: idx, table: false)]
        guard Float(W * H) * cellArea >= o.splitMinArea, W * H < 400_000 else { return whole }
        func cellOf(_ n: Int) -> Int {
            min(W - 1, Int((pts[n].u - lo.x) / cell)) + min(H - 1, Int((pts[n].v - lo.y) / cell)) * W
        }

        // Highest point per cell, and the up-facing heights in the tabletop range per 2 cm bin
        // (each cell counted once per bin).
        var top = [Float](repeating: -.infinity, count: W * H)
        var bottom = [Float](repeating: .infinity, count: W * H)
        let bin: Float = 0.02
        let bins = Int((o.tableHigh - o.tableLow) / bin) + 1
        var binCells = [Set<Int>](repeating: [], count: bins)
        for n in idx {
            let k = cellOf(n), y = pts[n].y - floorY
            top[k] = max(top[k], y); bottom[k] = min(bottom[k], y)
            if pts[n].up, y >= o.tableLow, y <= o.tableHigh { binCells[Int((y - o.tableLow) / bin)].insert(k) }
        }
        // Tabletop level: the 6 cm window with the most cells.
        var best = (count: 0, at: 0)
        for i in 0..<bins {
            let cells = binCells[max(0, i - 1)...min(bins - 1, i + 1)].reduce(into: Set<Int>()) { $0.formUnion($1) }
            if cells.count > best.count { best = (cells.count, i) }
        }
        guard Float(best.count) * cellArea >= 2 * o.minTableArea else { return whole }
        let level = o.tableLow + (Float(best.at) + 0.5) * bin
        var isTable = [Bool](repeating: false, count: W * H)
        for i in max(0, best.at - 3)...min(bins - 1, best.at + 3) { for k in binCells[i] { isTable[k] = true } }
        // Upright surfaces rising above the tabletop (a divider between desks back to back, a
        // screen) don't count when finding separate tabletops, so a divider keeps two desks apart
        // even with no gap between their tops. The tops grow back over them afterwards.
        var barrier = [Bool](repeating: false, count: W * H)
        for n in idx where !pts[n].up {
            let y = pts[n].y - floorY
            if y > level + 0.1, y < level + 0.8 { barrier[cellOf(n)] = true }
        }

        // Separate tabletops (4-connected); tiny patches don't count as tables.
        func components(_ member: (Int) -> Bool) -> [[Int]] {
            var seen = [Bool](repeating: false, count: W * H)
            var out: [[Int]] = []
            for start in 0..<(W * H) where member(start) && !seen[start] {
                var comp: [Int] = [], stack = [start]
                seen[start] = true
                while let k = stack.popLast() {
                    comp.append(k)
                    let x = k % W, y = k / W
                    for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
                        let nx = x + dx, ny = y + dy
                        guard nx >= 0, ny >= 0, nx < W, ny < H else { continue }
                        let nk = ny * W + nx
                        if member(nk), !seen[nk] { seen[nk] = true; stack.append(nk) }
                    }
                }
                out.append(comp)
            }
            return out
        }
        // Tabletops that only touch along a thin strip (desks back to back with gaps in the
        // divider) are cut apart: find them eroded by 10 cm, then grow them back.
        let r = 2
        var core = [Bool](repeating: false, count: W * H)
        for y in 0..<H { for x in 0..<W where isTable[y * W + x] {
            var all = true
            outer: for dy in -r...r { for dx in -r...r {
                let nx = x + dx, ny = y + dy
                if nx < 0 || ny < 0 || nx >= W || ny >= H || !isTable[ny * W + nx] || barrier[ny * W + nx] { all = false; break outer }
            } }
            core[y * W + x] = all
        } }
        let tables = components { core[$0] }.filter { Float($0.count) * cellArea >= o.minTableArea * 0.5 }
        guard !tables.isEmpty else { return whole }
        var group = [Int](repeating: -1, count: W * H)
        var front: [Int] = []
        for (g, comp) in tables.enumerated() { for k in comp { group[k] = g; front.append(k) } }
        while !front.isEmpty {
            var next: [Int] = []
            for k in front {
                let x = k % W, y = k / W
                for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
                    let nx = x + dx, ny = y + dy
                    guard nx >= 0, ny >= 0, nx < W, ny < H else { continue }
                    let nk = ny * W + nx
                    if isTable[nk], group[nk] < 0 { group[nk] = group[k]; next.append(nk) }
                }
            }
            front = next
        }

        // Everything else: its own thing if it has enough low surface, else part of the
        // tabletop it borders most.
        var nextGroup = tables.count
        let rest = components { group[$0] < 0 && top[$0] > -.infinity }
        var lateMerge: [(comp: [Int], border: [Int: Int])] = []
        for comp in rest {
            // Reaching well below the tabletop: stands on the floor beside it (a chair, a bin), not on it.
            let low = comp.filter { bottom[$0] < level - 0.15 }.count
            var border: [Int: Int] = [:]
            for k in comp {
                let x = k % W, y = k / W
                for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
                    let nx = x + dx, ny = y + dy
                    guard nx >= 0, ny >= 0, nx < W, ny < H else { continue }
                    let g = group[ny * W + nx]
                    if g >= 0, g < tables.count { border[g, default: 0] += 1 }
                }
            }
            // Chairs and bins are compact; long thin or L-shaped leftovers along a tabletop are its
            // legs, end panels or a divider down to the floor.
            let xs = comp.map { $0 % W }, ys = comp.map { $0 / W }
            let fill = Float(comp.count) / Float((xs.max()! - xs.min()! + 1) * (ys.max()! - ys.min()! + 1))
            if border.isEmpty || (Float(low) * cellArea >= o.minStandingArea && fill >= 0.35) {
                for k in comp { group[k] = nextGroup }
                nextGroup += 1
            } else {
                lateMerge.append((comp, border))
            }
        }
        for m in lateMerge {
            let g = m.border.max { $0.value < $1.value }!.key
            for k in m.comp { group[k] = g }
        }

        var out = [[Int]](repeating: [], count: nextGroup)
        for n in idx { out[group[cellOf(n)]].append(n) }
        return out.enumerated().filter { !$0.element.isEmpty }.map { ($0.element, $0.offset < tables.count) }
    }

    static func contains(_ poly: [SIMD2<Float>], _ p: SIMD2<Float>) -> Bool {
        var inside = false
        for i in 0..<poly.count {
            let a = poly[i], b = poly[(i + 1) % poly.count]
            if (a.y > p.y) != (b.y > p.y), p.x < (b.x - a.x) * (p.y - a.y) / (b.y - a.y) + a.x { inside.toggle() }
        }
        return inside
    }

    static func distance(_ p: SIMD2<Float>, _ a: SIMD2<Float>, _ b: SIMD2<Float>) -> Float {
        let d = b - a
        let t = max(0, min(1, simd_dot(p - a, d) / max(simd_dot(d, d), 1e-9)))
        return simd_distance(p, a + d * t)
    }
}

/// The floor (or ceiling) height per 1 m square of the plan. Over a long walk the tracking drifts,
/// so the floor of a far room can sit a few cm off the fitted level and would otherwise count as
/// things standing on it. Each square takes the most common height of the flat surfaces facing up
/// within 25 cm of the fitted floor (a mattress lying on the floor and filling a whole square can
/// be taken for it). For the ceiling: facing down, from 60 cm under the fitted ceiling (the
/// highest in the scan) to 25 cm over it, since a hallway or a far room can be lower. Squares
/// without enough of it keep the fitted level.
struct LocalLevel {
    let base: Float
    private let lo: SIMD2<Float>
    private let w: Int, h: Int
    private var level: [Float]
    private let cell: Float = 1

    init(_ samples: [RoomFitter.Sample], shell: RoomShell, toRoom: (Float, Float) -> SIMD2<Float>, floor: Bool) {
        base = floor ? shell.floorY : shell.ceilingY
        let below: Float = floor ? 0.25 : 0.6, above: Float = 0.25, bin: Float = 0.01
        let minCount: Int32 = floor ? 55 : 200
        var mn = SIMD2<Float>(repeating: .infinity), mx = SIMD2<Float>(repeating: -.infinity)
        var flat: [(q: SIMD2<Float>, y: Float)] = []
        for sm in samples where (floor ? sm.n.y > 0.9 : sm.n.y < -0.9) && sm.p.y > base - below && sm.p.y < base + above {
            let q = toRoom(sm.p.x, sm.p.z)
            flat.append((q, sm.p.y)); mn = simd_min(mn, q); mx = simd_max(mx, q)
        }
        guard !flat.isEmpty else { lo = .zero; w = 0; h = 0; level = []; return }
        lo = mn
        w = Int((mx.x - mn.x) / cell) + 1; h = Int((mx.y - mn.y) / cell) + 1
        let bins = Int((below + above) / bin) + 1
        var hist = [[Int32]](repeating: [], count: w * h)
        for f in flat {
            let k = min(w - 1, Int((f.q.x - mn.x) / cell)) + min(h - 1, Int((f.q.y - mn.y) / cell)) * w
            if hist[k].isEmpty { hist[k] = [Int32](repeating: 0, count: bins) }
            hist[k][min(bins - 1, max(0, Int((f.y - base + below) / bin)))] += 1
        }
        let b = base
        level = hist.map { hs in
            // The 3 cm window with the most points: at least 0.05 m² of 3 cm samples for a floor,
            // 0.18 m² for a ceiling.
            guard hs.count == bins else { return b }
            var best = (n: Int32(0), i: 0)
            for i in 1..<(bins - 1) { let n = hs[i - 1] + hs[i] + hs[i + 1]; if n > best.n { best = (n, i) } }
            return best.n >= minCount ? b - below + (Float(best.i) + 0.5) * bin : b
        }
    }

    /// Height at a room-frame point (u, v).
    func at(_ q: SIMD2<Float>) -> Float {
        guard w > 0 else { return base }
        let x = Int(((q.x - lo.x) / cell).rounded(.down)), y = Int(((q.y - lo.y) / cell).rounded(.down))
        guard x >= 0, y >= 0, x < w, y < h else { return base }
        return level[y * w + x]
    }
}

/// Keeps box ids (and labels) stable while boxes are re-found from a growing scan.
struct BoxTracker {
    private(set) var boxes: [ObjectBox] = []
    private var nextId = 1
    private var missed: [Int: Int] = [:]
    /// Updates a box may be missing before it is dropped.
    var keepMissing = 3

    init(seed: [ObjectBox] = []) {
        boxes = seed
        nextId = (seed.map(\.id).max() ?? 0) + 1
    }

    static func overlap(_ a: ObjectBox, _ b: ObjectBox) -> Float {
        let (al, ah) = a.bounds, (bl, bh) = b.bounds
        let e = simd_max(simd_min(ah, bh) - simd_max(al, bl), .zero)
        let inter = e.x * e.y * e.z
        let va = (ah - al), vb = (bh - bl)
        let union = va.x * va.y * va.z + vb.x * vb.y * vb.z - inter
        return union > 0 ? inter / union : 0
    }

    /// Takes freshly found boxes and returns them with ids carried over from matching boxes.
    @discardableResult
    mutating func update(_ found: [ObjectBox]) -> [ObjectBox] {
        var pairs: [(o: Int, n: Int, iou: Float)] = []
        for (i, old) in boxes.enumerated() {
            for (j, new) in found.enumerated() {
                let iou = Self.overlap(old, new)
                if iou > 0.1 { pairs.append((i, j, iou)) }
            }
        }
        var usedOld = Set<Int>(), usedNew = Set<Int>()
        var out = found
        for p in pairs.sorted(by: { $0.iou > $1.iou }) where !usedOld.contains(p.o) && !usedNew.contains(p.n) {
            usedOld.insert(p.o); usedNew.insert(p.n)
            let old = boxes[p.o]
            out[p.n].id = old.id
            out[p.n].label = old.label; out[p.n].confidence = old.confidence
            out[p.n].source = old.source; out[p.n].guesses = old.guesses; out[p.n].kind = old.kind
            missed[old.id] = nil
        }
        for j in out.indices where !usedNew.contains(j) {
            out[j].id = nextId; nextId += 1
        }
        // Old boxes that vanished stay a little while, unless something new now covers them.
        for (i, old) in boxes.enumerated() where !usedOld.contains(i) {
            if found.contains(where: { Self.overlap(old, $0) > 0.05 }) { missed[old.id] = nil; continue }
            let m = (missed[old.id] ?? 0) + 1
            if m <= keepMissing { missed[old.id] = m; out.append(old) } else { missed[old.id] = nil }
        }
        boxes = out
        return out
    }
}
