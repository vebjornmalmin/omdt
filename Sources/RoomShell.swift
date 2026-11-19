import Foundation
import simd

// "Snap to room": fits floor, ceiling and straight walls to a fused scan.
//
// 1. Sample the mesh into a 3 cm grid of points with normals.
// 2. Floor = lowest large up-facing plane, ceiling = highest large down-facing plane (or a guess).
// 3. Room direction = the dominant wall-normal angle, folded to 90° (rooms are mostly rectangular).
// 4. In that frame, wall planes are peaks in a histogram of wall-point positions along each axis.
// 5. The floor plan is cut along those wall planes into rectangles; a rectangle is inside the room
//    when most of it was seen. The outline of the inside rectangles is the floor polygon, so every
//    edge lies exactly on a wall plane (or on the edge of what was scanned, when a wall is missing).

/// Floor, ceiling and walls fitted to a scan. World coordinates are ARKit meters, y up.
struct RoomShell: Codable, Equatable {
    var floorY: Float
    var ceilingY: Float
    /// False when no ceiling was scanned and the height is a guess.
    var ceilingObserved: Bool
    /// Direction of the room's first axis on the floor, radians from world +x toward +z.
    var yaw: Float
    /// Floor polygon corners as [x, z], ordered with the room on the left of each edge.
    var outline: [[Float]]
    var walls: [Wall]
    /// Floor area in m².
    var area: Float
    /// Size along the room's two axes, in meters.
    var extent: [Float]
    /// How the fit went, for the debug readout.
    var notes: String

    struct Wall: Codable, Equatable, Identifiable {
        var id: Int
        var start: [Float]   // [x, z]
        var end: [Float]
        /// Unit normal [x, z] pointing into the room.
        var normal: [Float]
        var length: Float
        /// Share of the wall's length backed by scanned wall surface, 0-1.
        var coverage: Float
    }

    var height: Float { ceilingY - floorY }
}

/// What gets saved as room.json in a scan bundle. Object boxes join the shell in a later step.
struct RoomModel: Codable, Equatable {
    static let currentVersion = 1
    var version = RoomModel.currentVersion
    var shell: RoomShell
}

extension ScanBundle {
    static let roomFile = "room.json"

    /// The saved room, if it was made by the current fitter.
    func loadRoom() -> RoomModel? {
        guard let data = try? Data(contentsOf: url.appendingPathComponent(Self.roomFile)),
              let room = try? Self.decoder.decode(RoomModel.self, from: data),
              room.version == RoomModel.currentVersion else { return nil }
        return room
    }

    func saveRoom(_ room: RoomModel) throws {
        try Self.encoder.encode(room).write(to: url.appendingPathComponent(Self.roomFile), options: .atomic)
    }
}

enum RoomFitter {
    struct Options {
        var sampleCell: Float = 0.03
        var heightBin: Float = 0.02
        /// Smallest floor or ceiling patch that counts, m².
        var minPlaneArea: Float = 0.3
        /// Smallest floor patch accepted when furniture hides most of the floor, m².
        var minFloorPatch: Float = 0.04
        /// Smallest wall patch that counts, m², and its smallest height span.
        var minWallArea: Float = 0.25
        var minWallSpan: Float = 0.5
        var gridCell: Float = 0.05
        /// Cut lines closer than this are merged.
        var mergeDistance: Float = 0.12
        var defaultHeight: Float = 2.4
    }

    enum FitError: LocalizedError {
        case tooSmall, noFloor
        var errorDescription: String? {
            switch self {
            case .tooSmall: return "The scan is too small to find a room."
            case .noFloor: return "Couldn't find the floor. Scan some open floor and try again."
            }
        }
    }

    struct Sample { var p: SIMD3<Float>; var n: SIMD3<Float> }

    // MARK: entry point

    static func fit(_ meshes: [MeshOut], options o: Options = Options()) throws -> RoomShell {
        let t0 = Date()
        let samples = sample(meshes, cell: o.sampleCell)
        guard samples.count >= 500 else { throw FitError.tooSmall }
        let sampleArea = o.sampleCell * o.sampleCell

        // Floor: the lowest large up-facing plane. When the walls reach well below it, that was
        // furniture (a bed or table hiding most of the floor), so take the lowest small floor patch
        // near the bottom of the walls, or the wall bottom itself.
        let allVertical = samples.filter { abs($0.n.y) < 0.3 }.map(\.p.y).sorted()
        let wallBottom = allVertical.isEmpty ? nil : allVertical[allVertical.count / 100]
        var floorY = horizontalPlane(samples, up: true, above: -.infinity, minArea: o.minPlaneArea, relative: 0.2, o: o)
        if let wb = wallBottom, wb < (floorY ?? .infinity) - 0.25 {
            let patch = horizontalPlane(samples, up: true, above: wb - 0.35, minArea: o.minFloorPatch, relative: 0, o: o)
            if let p = patch, p < (floorY ?? .infinity) - 0.25 { floorY = p } else if floorY == nil { floorY = wb }
        }
        guard let floorY else { throw FitError.noFloor }
        let vertical = samples.filter { abs($0.n.y) < 0.3 && $0.p.y > floorY + 0.1 }
        var ceilingY: Float
        var ceilingObserved = false
        if let c = horizontalPlane(samples, up: false, above: floorY + 1.8, minArea: o.minPlaneArea, relative: 0.2, o: o) {
            ceilingY = c; ceilingObserved = true
        } else {
            let tops = vertical.map(\.p.y).sorted()
            let wallTop = tops.isEmpty ? floorY : tops[min(tops.count - 1, Int(Float(tops.count) * 0.99))]
            ceilingY = max(floorY + o.defaultHeight, wallTop)
        }
        let walls = vertical.filter { $0.p.y < ceilingY - 0.05 }

        // Room direction and frame: u = (cos yaw, sin yaw), v = (-sin yaw, cos yaw) in (x, z).
        let yaw = dominantYaw(walls)
        let c = cos(yaw), s = sin(yaw)
        func toRoom(_ x: Float, _ z: Float) -> SIMD2<Float> { SIMD2(x * c + z * s, -x * s + z * c) }
        func toWorld(_ q: SIMD2<Float>) -> [Float] { [q.x * c - q.y * s, q.x * s + q.y * c] }

        // Wall planes along each axis.
        var planes: [[Plane]] = [[], []]
        for axis in 0..<2 {
            planes[axis] = wallPlanes(walls, axis: axis, toRoom: toRoom, sampleArea: sampleArea, o: o)
        }

        // Occupancy of everything between floor and ceiling, seen from above.
        let inRoom = samples.filter { $0.p.y > floorY - 0.05 && $0.p.y < ceilingY + 0.05 }
        let grid = Occupancy(points: inRoom.map { toRoom($0.p.x, $0.p.z) }, cell: o.gridCell)

        // Cut the plan along wall planes (plus the scanned extent where a wall is missing).
        let cuts = (0..<2).map { axis -> [Cut] in
            let (lo, hi) = grid.filledRange(axis: axis)
            var all = planes[axis].map { Cut(at: $0.offset, plane: $0) }
            all.append(Cut(at: lo, plane: nil)); all.append(Cut(at: hi, plane: nil))
            return mergeCuts(all, distance: o.mergeDistance)
        }
        let inside = classifyCells(cuts: cuts, grid: grid)
        guard let loop = outlineLoop(inside), loop.count >= 4 else { throw FitError.tooSmall }

        // Corners and walls in world coordinates.
        let corners = loop.map { SIMD2(cuts[0][$0.x].at, cuts[1][$0.y].at) }
        var shellWalls: [RoomShell.Wall] = []
        var area: Float = 0
        for i in 0..<loop.count {
            let a = corners[i], b = corners[(i + 1) % corners.count]
            area += a.x * b.y - b.x * a.y
            let d = b - a, len = simd_length(d)
            let inward = SIMD2(-d.y, d.x) / len
            // The edge lies on a cut line; it counts as seen where that line's wall has points.
            let along = abs(d.x) > abs(d.y) ? 0 : 1                 // axis the edge runs along
            let cut = cuts[1 - along][along == 0 ? loop[i].y : loop[i].x]
            var coverage: Float = 0
            if let p = cut.plane, p.facing == (inward[1 - along] > 0 ? 1 : -1) {
                coverage = p.coverage(from: min(a[along], b[along]), to: max(a[along], b[along]))
            }
            let n = toWorld(inward)
            shellWalls.append(RoomShell.Wall(id: i, start: toWorld(a), end: toWorld(b), normal: n,
                                             length: len, coverage: coverage))
        }
        let lo = corners.reduce(SIMD2<Float>(repeating: .infinity)) { simd_min($0, $1) }
        let hi = corners.reduce(SIMD2<Float>(repeating: -.infinity)) { simd_max($0, $1) }

        let ms = Int(Date().timeIntervalSince(t0) * 1000)
        let notes = "\(samples.count) points, \(walls.count) on walls, \(planes[0].count)+\(planes[1].count) wall planes, "
            + "\(cuts[0].count - 1)×\(cuts[1].count - 1) cells, yaw \(String(format: "%.1f", yaw * 180 / .pi))°, \(ms) ms"
        return RoomShell(floorY: floorY, ceilingY: ceilingY, ceilingObserved: ceilingObserved, yaw: yaw,
                         outline: corners.map(toWorld), walls: shellWalls, area: abs(area) / 2,
                         extent: [hi.x - lo.x, hi.y - lo.y], notes: notes)
    }

    // MARK: sampling

    /// Averages mesh vertices into a grid; cells whose normals disagree are dropped.
    static func sample(_ meshes: [MeshOut], cell: Float) -> [Sample] {
        var acc: [Int64: (p: SIMD3<Float>, n: SIMD3<Float>, k: Float)] = [:]
        acc.reserveCapacity(meshes.reduce(0) { $0 + $1.pos.count } / 4)
        let inv = 1 / cell
        for m in meshes {
            for i in 0..<min(m.pos.count, m.nrm.count) {
                let p = m.pos[i]
                guard p.x.isFinite, p.y.isFinite, p.z.isFinite else { continue }
                let q = SIMD3<Int64>(Int64((p.x * inv).rounded(.down)), Int64((p.y * inv).rounded(.down)),
                                     Int64((p.z * inv).rounded(.down))) &+ (1 << 20)
                let key = (q.x << 42) | (q.y << 21) | q.z
                add(&acc[key, default: (.zero, .zero, 0)], p, m.nrm[i])
            }
        }
        func add(_ v: inout (p: SIMD3<Float>, n: SIMD3<Float>, k: Float), _ p: SIMD3<Float>, _ n: SIMD3<Float>) {
            v.p += p; v.n += n; v.k += 1
        }
        var out: [Sample] = []
        out.reserveCapacity(acc.count)
        for v in acc.values {
            let nl = simd_length(v.n)
            guard nl > 0.5 * v.k else { continue }
            out.append(Sample(p: v.p / v.k, n: v.n / nl))
        }
        return out
    }

    // MARK: floor and ceiling

    /// Height of the lowest (up) or highest (down) horizontal plane with enough area.
    static func horizontalPlane(_ samples: [Sample], up: Bool, above: Float, minArea: Float, relative: Float,
                                o: Options) -> Float? {
        let pts = samples.filter { (up ? $0.n.y > 0.85 : $0.n.y < -0.85) && $0.p.y > above }.map(\.p.y)
        guard let lo = pts.min(), let hi = pts.max() else { return nil }
        let bins = Int((hi - lo) / o.heightBin) + 1
        var h = [Float](repeating: 0, count: bins)
        for y in pts { h[Int((y - lo) / o.heightBin)] += 1 }
        let sm = smooth(h, radius: 1, circular: false)
        let minCount = max(minArea / (o.sampleCell * o.sampleCell), relative * (sm.max() ?? 0))
        let order = up ? Array(0..<bins) : Array((0..<bins).reversed())
        for i in order where sm[i] >= minCount
            && (i == 0 || sm[i] >= sm[i - 1]) && (i == bins - 1 || sm[i] >= sm[i + 1]) {
            let center = lo + (Float(i) + 0.5) * o.heightBin
            let near = pts.filter { abs($0 - center) < 0.04 }
            return near.reduce(0, +) / Float(near.count)
        }
        return nil
    }

    // MARK: direction

    /// The wall direction folded into 90°, in (-45°, 45°].
    static func dominantYaw(_ walls: [Sample]) -> Float {
        guard !walls.isEmpty else { return 0 }
        let quarter = Float.pi / 2
        var h = [Float](repeating: 0, count: 90)
        var angles: [(a: Float, w: Float)] = []
        angles.reserveCapacity(walls.count)
        for s in walls {
            let w = simd_length(SIMD2(s.n.x, s.n.z))
            var a = atan2(s.n.z, s.n.x).truncatingRemainder(dividingBy: quarter)
            if a < 0 { a += quarter }
            angles.append((a, w))
            h[min(89, Int(a / quarter * 90))] += w
        }
        let sm = smooth(h, radius: 2, circular: true)
        let peak = (Float(sm.indices.max { sm[$0] < sm[$1] }!) + 0.5) / 90 * quarter
        // Circular mean of 4×angle near the peak.
        var cs: Float = 0, sn: Float = 0
        for (a, w) in angles {
            var d = abs(a - peak); d = min(d, quarter - d)
            if d < 6 * .pi / 180 { cs += w * cos(4 * a); sn += w * sin(4 * a) }
        }
        return atan2(sn, cs) / 4
    }

    // MARK: wall planes

    struct Plane {
        var axis: Int
        /// Position along the axis, in the room frame.
        var offset: Float
        /// +1 when the wall faces +axis (room on the + side), else -1.
        var facing: Int
        var count: Float
        /// 10 cm bins along the other axis that have wall points.
        var seen: Set<Int>

        func coverage(from a: Float, to b: Float) -> Float {
            let i0 = Int((a / 0.1).rounded(.down)), i1 = max(i0, Int((b / 0.1).rounded(.up)) - 1)
            let hits = (i0...i1).filter { seen.contains($0) }.count
            return Float(hits) / Float(i1 - i0 + 1)
        }
    }

    static func wallPlanes(_ walls: [Sample], axis: Int, toRoom: (Float, Float) -> SIMD2<Float>,
                           sampleArea: Float, o: Options) -> [Plane] {
        let bin: Float = 0.02
        var found: [Plane] = []
        for facing in [1, -1] {
            // Wall points whose normal is within 15° of this axis direction.
            var pts: [(q: SIMD2<Float>, y: Float)] = []
            for s in walls {
                let n = toRoom(s.n.x, s.n.z)
                let nl = simd_length(n)
                guard nl > 0, n[axis] * Float(facing) / nl > 0.966 else { continue }
                pts.append((toRoom(s.p.x, s.p.z), s.p.y))
            }
            guard let lo = pts.map({ $0.q[axis] }).min(), let hi = pts.map({ $0.q[axis] }).max() else { continue }
            let bins = Int((hi - lo) / bin) + 1
            var h = [Float](repeating: 0, count: bins)
            for p in pts { h[Int((p.q[axis] - lo) / bin)] += 1 }
            let sm = smooth(h, radius: 1, circular: false)
            let minCount = o.minWallArea / sampleArea
            for i in 0..<bins where sm[i] >= minCount
                && (i == 0 || sm[i] >= sm[i - 1]) && (i == bins - 1 || sm[i] > sm[i + 1]) {
                let center = lo + (Float(i) + 0.5) * bin
                let near = pts.filter { abs($0.q[axis] - center) < 0.04 }
                let ys = near.map(\.y).sorted()
                let span = ys[Int(Float(ys.count) * 0.95)] - ys[Int(Float(ys.count) * 0.05)]
                guard span >= o.minWallSpan else { continue }
                var seen = Set<Int>()
                var perBin: [Int: Int] = [:]
                for p in near { perBin[Int((p.q[1 - axis] / 0.1).rounded(.down)), default: 0] += 1 }
                for (k, v) in perBin where v >= 3 { seen.insert(k) }
                let offset = near.reduce(0) { $0 + $1.q[axis] } / Float(near.count)
                found.append(Plane(axis: axis, offset: offset, facing: facing, count: Float(near.count), seen: seen))
            }
        }
        // Keep the strongest of planes closer than 10 cm (also both faces of a thin wall).
        var kept: [Plane] = []
        for p in found.sorted(by: { $0.count > $1.count })
            where !kept.contains(where: { abs($0.offset - p.offset) < 0.1 }) {
            kept.append(p)
        }
        return kept.sorted { $0.offset < $1.offset }
    }

    // MARK: floor plan

    /// What was seen between floor and ceiling, from above, with gaps closed and holes filled.
    struct Occupancy {
        let origin: SIMD2<Float>
        let cell: Float
        let w: Int, h: Int
        var filled: [Bool]

        init(points: [SIMD2<Float>], cell: Float) {
            self.cell = cell
            // Bounds from robust percentiles so stray points far away don't blow up the grid.
            func range(_ k: Int) -> (Float, Float) {
                let v = points.map { $0[k] }.sorted()
                guard !v.isEmpty else { return (0, 1) }
                return (v[Int(Float(v.count - 1) * 0.002)] - 0.2, v[Int(Float(v.count - 1) * 0.998)] + 0.2)
            }
            let (x0, x1) = range(0), (y0, y1) = range(1)
            origin = SIMD2(x0, y0)
            w = min(1200, Int((x1 - x0) / cell) + 1); h = min(1200, Int((y1 - y0) / cell) + 1)
            var g = [Bool](repeating: false, count: w * h)
            for p in points {
                let i = Int((p.x - x0) / cell), j = Int((p.y - y0) / cell)
                if i >= 0, j >= 0, i < w, j < h { g[j * w + i] = true }
            }
            // Close gaps of up to ~20 cm, then fill everything not reachable from the border.
            g = Self.erode(Self.dilate(g, w, h, 2), w, h, 2)
            filled = Self.fillHoles(g, w, h)
        }

        /// Range of filled cells along an axis, as cell edges.
        func filledRange(axis: Int) -> (Float, Float) {
            var lo = Int.max, hi = Int.min
            for j in 0..<h { for i in 0..<w where filled[j * w + i] {
                let k = axis == 0 ? i : j
                lo = min(lo, k); hi = max(hi, k)
            } }
            guard lo <= hi else { return (origin[axis], origin[axis] + cell) }
            return (origin[axis] + Float(lo) * cell, origin[axis] + Float(hi + 1) * cell)
        }

        /// Share of filled cells whose centers fall in a rectangle.
        func fraction(_ a: SIMD2<Float>, _ b: SIMD2<Float>) -> Float {
            let i0 = max(0, Int(((a.x - origin.x) / cell - 0.5).rounded(.up)))
            let i1 = min(w - 1, Int(((b.x - origin.x) / cell - 0.5).rounded(.down)))
            let j0 = max(0, Int(((a.y - origin.y) / cell - 0.5).rounded(.up)))
            let j1 = min(h - 1, Int(((b.y - origin.y) / cell - 0.5).rounded(.down)))
            guard i0 <= i1, j0 <= j1 else { return 0 }
            var n = 0
            for j in j0...j1 { for i in i0...i1 where filled[j * w + i] { n += 1 } }
            return Float(n) / Float((i1 - i0 + 1) * (j1 - j0 + 1))
        }

        static func dilate(_ g: [Bool], _ w: Int, _ h: Int, _ r: Int) -> [Bool] {
            var out = g
            for j in 0..<h { for i in 0..<w where g[j * w + i] {
                for dj in -r...r { for di in -r...r {
                    let x = i + di, y = j + dj
                    if x >= 0, y >= 0, x < w, y < h { out[y * w + x] = true }
                } }
            } }
            return out
        }

        static func erode(_ g: [Bool], _ w: Int, _ h: Int, _ r: Int) -> [Bool] {
            let inv = dilate(g.map { !$0 }, w, h, r)
            return inv.map { !$0 }
        }

        static func fillHoles(_ g: [Bool], _ w: Int, _ h: Int) -> [Bool] {
            var outside = [Bool](repeating: false, count: w * h)
            var stack: [Int] = []
            for i in 0..<w { stack.append(i); stack.append((h - 1) * w + i) }
            for j in 0..<h { stack.append(j * w); stack.append(j * w + w - 1) }
            while let k = stack.popLast() {
                guard !g[k], !outside[k] else { continue }
                outside[k] = true
                let i = k % w, j = k / w
                if i > 0 { stack.append(k - 1) }
                if i < w - 1 { stack.append(k + 1) }
                if j > 0 { stack.append(k - w) }
                if j < h - 1 { stack.append(k + w) }
            }
            return outside.map { !$0 }
        }
    }

    struct Cut { var at: Float; var plane: Plane? }

    /// Sorts cut lines and merges close ones; wall planes win over scan edges, stronger walls over weaker.
    static func mergeCuts(_ cuts: [Cut], distance: Float) -> [Cut] {
        var out: [Cut] = []
        for c in cuts.sorted(by: { $0.at < $1.at }) {
            if let last = out.last, c.at - last.at < distance {
                if (c.plane?.count ?? -1) > (last.plane?.count ?? -1) { out[out.count - 1] = c }
            } else {
                out.append(c)
            }
        }
        return out
    }

    /// Which rectangles between cut lines are inside: mostly seen, connected to the largest
    /// region, with enclosed gaps filled. Indexed [i][j] for cuts[0][i]..[i+1] × cuts[1][j]..[j+1].
    static func classifyCells(cuts: [[Cut]], grid: Occupancy) -> [[Bool]] {
        let nu = cuts[0].count - 1, nv = cuts[1].count - 1
        guard nu > 0, nv > 0 else { return [] }
        var inside = (0..<nu).map { i in (0..<nv).map { j in
            grid.fraction(SIMD2(cuts[0][i].at, cuts[1][j].at), SIMD2(cuts[0][i + 1].at, cuts[1][j + 1].at)) >= 0.5
        } }
        // Largest 4-connected region.
        var label = [[Int]](repeating: [Int](repeating: -1, count: nv), count: nu)
        var sizes: [Int] = []
        for i in 0..<nu { for j in 0..<nv where inside[i][j] && label[i][j] < 0 {
            let id = sizes.count; var size = 0
            var stack = [(i, j)]
            while let (a, b) = stack.popLast() {
                guard a >= 0, b >= 0, a < nu, b < nv, inside[a][b], label[a][b] < 0 else { continue }
                label[a][b] = id; size += 1
                stack += [(a - 1, b), (a + 1, b), (a, b - 1), (a, b + 1)]
            }
            sizes.append(size)
        } }
        guard let best = sizes.indices.max(by: { sizes[$0] < sizes[$1] }) else { return inside }
        for i in 0..<nu { for j in 0..<nv { inside[i][j] = label[i][j] == best } }
        // Fill enclosed holes (pillars or unscanned patches in the middle).
        var outside = [[Bool]](repeating: [Bool](repeating: false, count: nv), count: nu)
        var stack: [(Int, Int)] = []
        for i in 0..<nu { stack += [(i, 0), (i, nv - 1)] }
        for j in 0..<nv { stack += [(0, j), (nu - 1, j)] }
        while let (a, b) = stack.popLast() {
            guard a >= 0, b >= 0, a < nu, b < nv, !inside[a][b], !outside[a][b] else { continue }
            outside[a][b] = true
            stack += [(a - 1, b), (a + 1, b), (a, b - 1), (a, b + 1)]
        }
        for i in 0..<nu { for j in 0..<nv { inside[i][j] = !outside[i][j] } }
        return inside
    }

    /// Outer boundary of the inside cells as cut-grid corners, room on the left, collinear
    /// corners removed.
    static func outlineLoop(_ inside: [[Bool]]) -> [SIMD2<Int>]? {
        let nu = inside.count
        guard nu > 0 else { return nil }
        let nv = inside[0].count
        func isIn(_ i: Int, _ j: Int) -> Bool { i >= 0 && j >= 0 && i < nu && j < nv && inside[i][j] }
        var next: [SIMD2<Int>: [SIMD2<Int>]] = [:]
        func edge(_ a: SIMD2<Int>, _ b: SIMD2<Int>) { next[a, default: []].append(b) }
        for i in 0..<nu { for j in 0..<nv where inside[i][j] {
            if !isIn(i, j - 1) { edge(SIMD2(i, j), SIMD2(i + 1, j)) }
            if !isIn(i + 1, j) { edge(SIMD2(i + 1, j), SIMD2(i + 1, j + 1)) }
            if !isIn(i, j + 1) { edge(SIMD2(i + 1, j + 1), SIMD2(i, j + 1)) }
            if !isIn(i - 1, j) { edge(SIMD2(i, j + 1), SIMD2(i, j)) }
        } }
        // Walk loops; keep the one enclosing the most area.
        var best: [SIMD2<Int>]?, bestArea = 0
        while let start = next.keys.first {
            var loop: [SIMD2<Int>] = []
            var p = start
            while var outs = next[p], !outs.isEmpty {
                loop.append(p)
                let q = outs.removeLast()
                next[p] = outs.isEmpty ? nil : outs
                p = q
                if p == start { break }
            }
            next[start] = nil
            var area = 0
            for k in 0..<loop.count {
                let a = loop[k], b = loop[(k + 1) % loop.count]
                area += a.x * b.y - b.x * a.y
            }
            if abs(area) > bestArea { bestArea = abs(area); best = loop }
        }
        guard var loop = best else { return nil }
        // Drop corners where the outline goes straight on.
        var k = 0
        while k < loop.count && loop.count > 3 {
            let a = loop[(k + loop.count - 1) % loop.count], b = loop[k], c = loop[(k + 1) % loop.count]
            let d1 = b &- a, d2 = c &- b
            if d1.x * d2.y - d1.y * d2.x == 0 { loop.remove(at: k) } else { k += 1 }
        }
        return loop
    }

    // MARK: helpers

    static func smooth(_ h: [Float], radius r: Int, circular: Bool) -> [Float] {
        let n = h.count
        return (0..<n).map { i in
            var s: Float = 0
            for d in -r...r {
                let k = i + d
                if circular { s += h[((k % n) + n) % n] } else if k >= 0 && k < n { s += h[k] }
            }
            return s
        }
    }
}
