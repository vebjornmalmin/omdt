import Foundation
import simd

// Multi-resolution TSDF fusion. Voxel size depends on how far the camera was from the
// surface when it was observed: 2 mm up close, doubling per level out to 32 mm.
//   level 0: depth < 0.25 m  -> 2 mm      level 3: < 2.0 m -> 16 mm
//   level 1: < 0.50 m        -> 4 mm      level 4: beyond  -> 32 mm
// Moving closer therefore integrates into finer blocks and the mesh refines there;
// coarse geometry underneath is hidden where finer data exists.

struct Voxel {
    var sdf: Int16 = 0          // truncated signed distance, normalized to +-32767
    var w: UInt8 = 0            // observation weight (0 = never observed)
    var r: UInt8 = 0, g: UInt8 = 0, b: UInt8 = 0
    var cw: UInt8 = 0           // color weight
}

struct DepthFrame {
    var depth: UnsafePointer<Float32>
    var conf: UnsafePointer<UInt8>?
    var width: Int, height: Int
    var depthStride: Int        // elements per row
    var confStride: Int         // bytes per row
    var fx: Float, fy: Float, cx: Float, cy: Float   // at depth resolution
    var camToWorld: simd_float4x4
}

struct ColorFrame {
    var y: UnsafePointer<UInt8>
    var cbcr: UnsafePointer<UInt8>
    var yStride: Int, cStride: Int
    var width: Int, height: Int
    var fx: Float, fy: Float, cx: Float, cy: Float
}

struct MeshOut {
    var pos: [SIMD3<Float>] = []
    var nrm: [SIMD3<Float>] = []
    var col: [SIMD4<UInt8>] = []
    var idx: [UInt32] = []
    var level = 0
    var isEmpty: Bool { idx.isEmpty }
}

@inline(__always)
func xformPoint(_ m: simd_float4x4, _ p: SIMD3<Float>) -> SIMD3<Float> {
    let r = m.columns.0 * p.x + m.columns.1 * p.y + m.columns.2 * p.z + m.columns.3
    return SIMD3(r.x, r.y, r.z)
}

func parallelFor(_ n: Int, _ body: (Int) -> Void) {
    if n <= 0 { return }
    if n == 1 { body(0); return }
    DispatchQueue.concurrentPerform(iterations: n, execute: body)
}

final class TSDFVolume {
    static let B = 16
    static let shift = 4
    static let levels = 5
    static let maxDepth: Float = 4.5
    static let minDepth: Float = 0.12

    @inline(__always) static func voxelSize(_ l: Int) -> Float { 0.002 * Float(1 << l) }
    @inline(__always) static func mu(_ l: Int) -> Float { max(4 * voxelSize(l), 0.012) }
    @inline(__always) static func level(forDepth d: Float) -> Int {
        d < 0.25 ? 0 : d < 0.5 ? 1 : d < 1.0 ? 2 : d < 2.0 ? 3 : 4
    }

    private var map: [UInt64: Int32] = [:]
    private var ptrs: [UnsafeMutablePointer<Voxel>] = []
    private var levelOf: [UInt8] = []
    private var coordOf: [SIMD3<Int32>] = []
    private var dirtyFlag: [Bool] = []
    private var dirtyList: [Int32] = []

    var blockCount: Int { ptrs.count }
    var allBlocks: [Int32] { (0..<Int32(ptrs.count)).map { $0 } }

    deinit { reset() }

    func reset() {
        for p in ptrs { p.deallocate() }
        ptrs.removeAll(); map.removeAll(); levelOf.removeAll()
        coordOf.removeAll(); dirtyFlag.removeAll(); dirtyList.removeAll()
    }

    // MARK: keys

    @inline(__always)
    private static func key(_ l: Int, _ x: Int, _ y: Int, _ z: Int) -> UInt64 {
        let off = 1 << 19, m = (1 << 20) - 1
        return UInt64(l) << 60 | UInt64((x + off) & m) << 40 | UInt64((y + off) & m) << 20 | UInt64((z + off) & m)
    }

    private static func decode(_ k: UInt64) -> (Int, Int, Int, Int) {
        let off = 1 << 19, m: UInt64 = (1 << 20) - 1
        return (Int(k >> 60), Int((k >> 40) & m) - off, Int((k >> 20) & m) - off, Int(k & m) - off)
    }

    @inline(__always)
    private func voxel(_ l: Int, _ gx: Int, _ gy: Int, _ gz: Int) -> Voxel? {
        let s = Self.shift, mk = Self.B - 1
        guard let i = map[Self.key(l, gx >> s, gy >> s, gz >> s)] else { return nil }
        return ptrs[Int(i)][(((gz & mk) << s) + (gy & mk)) << s + (gx & mk)]
    }

    private func markDirty(_ i: Int) {
        if !dirtyFlag[i] { dirtyFlag[i] = true; dirtyList.append(Int32(i)) }
    }

    // MARK: integration

    func integrate(depth d: DepthFrame, color: ColorFrame?) {
        let B = Self.B
        let c2w = d.camToWorld
        let w2c = c2w.inverse

        // 1. Find blocks near the observed surface (sampled every 2nd pixel).
        let step = 2
        let rows = (d.height + step - 1) / step
        let chunkRows = 8
        let nChunks = (rows + chunkRows - 1) / chunkRows
        var chunkKeys = [[UInt64]](repeating: [], count: nChunks)
        chunkKeys.withUnsafeMutableBufferPointer { out in
            DispatchQueue.concurrentPerform(iterations: nChunks) { ci in
                var local: [UInt64] = []
                var last: UInt64 = .max
                for r in (ci * chunkRows)..<min(rows, (ci + 1) * chunkRows) {
                    let v = r * step
                    for u in stride(from: 0, to: d.width, by: step) {
                        let dep = d.depth[v * d.depthStride + u]
                        if !(dep > Self.minDepth && dep < Self.maxDepth) { continue }
                        if let c = d.conf, c[v * d.confStride + u] < 1 { continue }
                        let l = Self.level(forDepth: dep)
                        let mu = Self.mu(l), bs = Self.voxelSize(l) * Float(B)
                        let rx = (Float(u) - d.cx) / d.fx, ry = -(Float(v) - d.cy) / d.fy
                        for k in -1...1 {
                            let t = dep + Float(k) * mu
                            let pw = xformPoint(c2w, SIMD3(rx * t, ry * t, -t))
                            let key = Self.key(l, Int((pw.x / bs).rounded(.down)),
                                               Int((pw.y / bs).rounded(.down)),
                                               Int((pw.z / bs).rounded(.down)))
                            if key != last { local.append(key); last = key }
                        }
                    }
                }
                out[ci] = local
            }
        }
        var keySet = Set<UInt64>()
        for c in chunkKeys { keySet.formUnion(c) }

        var active: [Int32] = []
        active.reserveCapacity(keySet.count)
        for k in keySet {
            if let i = map[k] { active.append(i); continue }
            let (l, x, y, z) = Self.decode(k)
            let p = UnsafeMutablePointer<Voxel>.allocate(capacity: B * B * B)
            p.initialize(repeating: Voxel(), count: B * B * B)
            let i = Int32(ptrs.count)
            ptrs.append(p); levelOf.append(UInt8(l))
            coordOf.append(SIMD3(Int32(x), Int32(y), Int32(z)))
            dirtyFlag.append(false); map[k] = i
            active.append(i)
        }

        // 2. Integrate every active block in parallel (each voxel has one writer).
        var touched = [UInt8](repeating: 0, count: active.count)
        touched.withUnsafeMutableBufferPointer { tp in
            active.withUnsafeBufferPointer { ap in
                DispatchQueue.concurrentPerform(iterations: ap.count) { ai in
                    tp[ai] = self.integrateBlock(Int(ap[ai]), d, w2c, color) ? 1 : 0
                }
            }
        }

        // 3. Mark what needs re-meshing: the block, its face neighbours, and the
        //    coarser blocks that overlap it (their hidden region changed).
        for (ai, bi32) in active.enumerated() where touched[ai] != 0 {
            let bi = Int(bi32)
            markDirty(bi)
            let l = Int(levelOf[bi]), c = coordOf[bi]
            let cx = Int(c.x), cy = Int(c.y), cz = Int(c.z)
            for (dx, dy, dz) in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)] {
                if let n = map[Self.key(l, cx + dx, cy + dy, cz + dz)] { markDirty(Int(n)) }
            }
            if l + 1 < Self.levels {
                for lc in (l + 1)..<Self.levels {
                    let s = lc - l
                    if let n = map[Self.key(lc, cx >> s, cy >> s, cz >> s)] { markDirty(Int(n)) }
                }
            }
        }
    }

    private func integrateBlock(_ bi: Int, _ d: DepthFrame, _ w2c: simd_float4x4, _ color: ColorFrame?) -> Bool {
        let B = Self.B
        let l = Int(levelOf[bi]), bc = coordOf[bi]
        let vs = Self.voxelSize(l), mu = Self.mu(l), invMu = 1 / mu
        let org = SIMD3<Float>(Float(bc.x), Float(bc.y), Float(bc.z)) * Float(B)
        let vox = ptrs[bi]
        var any = false

        for z in 0..<B { for y in 0..<B { for x in 0..<B {
            let pw = (org + SIMD3(Float(x), Float(y), Float(z))) * vs
            let pc = xformPoint(w2c, pw)
            let zc = -pc.z
            if zc < Self.minDepth { continue }
            let iz = 1 / zc
            let u = d.fx * pc.x * iz + d.cx
            let v = d.cy - d.fy * pc.y * iz
            if u < 0 || v < 0 { continue }
            let iu = Int(u + 0.5), iv = Int(v + 0.5)
            if iu >= d.width || iv >= d.height { continue }
            let dep = d.depth[iv * d.depthStride + iu]
            if !(dep > Self.minDepth && dep < Self.maxDepth) { continue }
            if let c = d.conf, c[iv * d.confStride + iu] < 1 { continue }
            if Self.level(forDepth: dep) != l { continue }
            let sdf = dep - zc
            if sdf < -mu { continue }
            let t = min(1, sdf * invMu)

            let vp = vox + ((z * B + y) * B + x)
            var vx = vp.pointee
            let w = Float(vx.w)
            let ns = (Float(vx.sdf) * (1 / 32767) * w + t) / (w + 1)
            vx.sdf = Int16(ns * 32767)
            vx.w = UInt8(min(w + 1, 64))

            if let cf = color, abs(sdf) < mu * 0.5 {
                let cu = cf.fx * pc.x * iz + cf.cx
                let cv = cf.cy - cf.fy * pc.y * iz
                if cu >= 0, cv >= 0 {
                    let xi = Int(cu), yi = Int(cv)
                    if xi < cf.width, yi < cf.height {
                        let Y = Float(cf.y[yi * cf.yStride + xi])
                        let ci = (yi >> 1) * cf.cStride + (xi >> 1) * 2
                        let cb = Float(cf.cbcr[ci]) - 128, cr = Float(cf.cbcr[ci + 1]) - 128
                        let nr = max(0, min(255, Y + 1.402 * cr))
                        let ng = max(0, min(255, Y - 0.344136 * cb - 0.714136 * cr))
                        let nb = max(0, min(255, Y + 1.772 * cb))
                        let cw = Float(vx.cw)
                        vx.r = UInt8((Float(vx.r) * cw + nr) / (cw + 1))
                        vx.g = UInt8((Float(vx.g) * cw + ng) / (cw + 1))
                        vx.b = UInt8((Float(vx.b) * cw + nb) / (cw + 1))
                        vx.cw = UInt8(min(cw + 1, 8))
                    }
                }
            }
            vp.pointee = vx
            any = true
        } } }
        return any
    }

    // MARK: dirty queue

    /// Removes and returns up to `budget` dirty blocks, nearest to `camera` first.
    func takeDirty(budget: Int, camera: SIMD3<Float>) -> [Int32] {
        if dirtyList.isEmpty { return [] }
        let B = Float(Self.B)
        func dist(_ i: Int32) -> Float {
            let c = coordOf[Int(i)]
            let bs = Self.voxelSize(Int(levelOf[Int(i)])) * B
            let center = (SIMD3<Float>(Float(c.x), Float(c.y), Float(c.z)) + 0.5) * bs
            return simd_distance_squared(center, camera)
        }
        var list = dirtyList
        if list.count > budget {
            list.sort { dist($0) < dist($1) }
            dirtyList = Array(list[budget...])
            list = Array(list[..<budget])
        } else {
            dirtyList = []
        }
        for i in list { dirtyFlag[Int(i)] = false }
        return list
    }

    func requeue(_ blocks: [Int32]) { for b in blocks { markDirty(Int(b)) } }

    // MARK: meshing (naive surface nets, no lookup tables)

    func mesh(blocks: [Int32]) -> [MeshOut] {
        var out = [MeshOut](repeating: MeshOut(), count: blocks.count)
        out.withUnsafeMutableBufferPointer { op in
            blocks.withUnsafeBufferPointer { bp in
                DispatchQueue.concurrentPerform(iterations: bp.count) { i in
                    op[i] = self.mesh(block: Int(bp[i]))
                }
            }
        }
        return out
    }

    private static let edgePairs: [(Int, Int)] = [(0, 1), (2, 3), (4, 5), (6, 7),
                                                  (0, 2), (1, 3), (4, 6), (5, 7),
                                                  (0, 4), (1, 5), (2, 6), (3, 7)]
    private static let cornerOff: [SIMD3<Float>] = (0..<8).map {
        SIMD3(Float($0 & 1), Float(($0 >> 1) & 1), Float(($0 >> 2) & 1))
    }

    private func isCovered(_ l: Int, _ g: SIMD3<Int>) -> Bool {
        if l == 0 { return false }
        for fl in stride(from: l - 1, through: 0, by: -1) {
            let s = 1 << (l - fl)
            let f = g &* s &+ SIMD3(repeating: s >> 1)
            if let v = voxel(fl, f.x, f.y, f.z), v.w > 0 { return true }
        }
        return false
    }

    func mesh(block bi: Int) -> MeshOut {
        let B = Self.B, P = B + 2, C = B + 1
        let l = Int(levelOf[bi]), bc = coordOf[bi]
        let vs = Self.voxelSize(l)
        var res = MeshOut(); res.level = l

        var nb = [UnsafeMutablePointer<Voxel>?](repeating: nil, count: 27)
        for dz in 0..<3 { for dy in 0..<3 { for dx in 0..<3 {
            if let i = map[Self.key(l, Int(bc.x) + dx - 1, Int(bc.y) + dy - 1, Int(bc.z) + dz - 1)] {
                nb[(dz * 3 + dy) * 3 + dx] = ptrs[Int(i)]
            }
        } } }

        // Padded voxel copy covering local indices -1...B.
        let n3 = P * P * P
        var sdf = [Float](repeating: 0, count: n3)
        var valid = [Bool](repeating: false, count: n3)
        var rgb = [SIMD4<UInt8>](repeating: .zero, count: n3)
        for pz in 0..<P {
            let z = pz - 1, nz = z < 0 ? 0 : (z >= B ? 2 : 1), lz = (z + B) & (B - 1)
            for py in 0..<P {
                let y = py - 1, ny = y < 0 ? 0 : (y >= B ? 2 : 1), ly = (y + B) & (B - 1)
                for px in 0..<P {
                    let x = px - 1, nx = x < 0 ? 0 : (x >= B ? 2 : 1), lx = (x + B) & (B - 1)
                    guard let p = nb[(nz * 3 + ny) * 3 + nx] else { continue }
                    let v = p[(lz * B + ly) * B + lx]
                    if v.w == 0 { continue }
                    let i = (pz * P + py) * P + px
                    sdf[i] = Float(v.sdf) * (1 / 32767)
                    valid[i] = true
                    rgb[i] = SIMD4(v.r, v.g, v.b, v.cw > 0 ? 255 : 0)
                }
            }
        }

        @inline(__always) func pidx(_ x: Int, _ y: Int, _ z: Int) -> Int { ((z + 1) * P + (y + 1)) * P + (x + 1) }
        @inline(__always) func cidx(_ x: Int, _ y: Int, _ z: Int) -> Int { ((z + 1) * C + (y + 1)) * C + (x + 1) }

        // Cell vertices (cells -1..<B per axis).
        var cellVert = [Int32](repeating: -1, count: C * C * C)
        var cp: [SIMD3<Float>] = [], cn: [SIMD3<Float>] = [], cc: [SIMD4<UInt8>] = [], ccov: [Bool] = []
        let originVox = SIMD3<Int>(Int(bc.x), Int(bc.y), Int(bc.z)) &* B
        let originF = SIMD3<Float>(Float(bc.x), Float(bc.y), Float(bc.z)) * Float(B)

        for cz in -1..<B { for cy in -1..<B { for cx in -1..<B {
            var s = SIMD8<Float>(repeating: 0)
            var mask = 0
            var ok = true
            var ids = (0, 0, 0, 0, 0, 0, 0, 0)
            withUnsafeMutableBytes(of: &ids) { raw in
                let ip = raw.bindMemory(to: Int.self)
                for k in 0..<8 {
                    let i = pidx(cx + (k & 1), cy + ((k >> 1) & 1), cz + ((k >> 2) & 1))
                    ip[k] = i
                    if !valid[i] { ok = false; break }
                    s[k] = sdf[i]
                    if s[k] >= 0 { mask |= 1 << k }
                }
            }
            if !ok || mask == 0 || mask == 255 { continue }

            var acc = SIMD3<Float>(repeating: 0); var cnt: Float = 0
            for (a, b) in Self.edgePairs where ((mask >> a) & 1) != ((mask >> b) & 1) {
                let t = s[a] / (s[a] - s[b])
                acc += Self.cornerOff[a] + t * (Self.cornerOff[b] - Self.cornerOff[a])
                cnt += 1
            }
            let local = acc / cnt
            let pos = (originF + SIMD3(Float(cx), Float(cy), Float(cz)) + local) * vs

            let g = SIMD3<Float>(
                (s[1] + s[3] + s[5] + s[7]) - (s[0] + s[2] + s[4] + s[6]),
                (s[2] + s[3] + s[6] + s[7]) - (s[0] + s[1] + s[4] + s[5]),
                (s[4] + s[5] + s[6] + s[7]) - (s[0] + s[1] + s[2] + s[3]))
            let gl = simd_length(g)
            let nrm = gl > 1e-9 ? g / gl : SIMD3<Float>(0, 0, 1)

            var cr: Float = 0, cg: Float = 0, cbl: Float = 0, cwSum: Float = 0
            withUnsafeBytes(of: &ids) { raw in
                let ip = raw.bindMemory(to: Int.self)
                for k in 0..<8 {
                    let c = rgb[ip[k]]
                    if c.w == 0 { continue }
                    let w = 1 / (0.05 + abs(s[k]))
                    cr += Float(c.x) * w; cg += Float(c.y) * w; cbl += Float(c.z) * w; cwSum += w
                }
            }
            let color: SIMD4<UInt8> = cwSum > 0
                ? SIMD4(UInt8(cr / cwSum), UInt8(cg / cwSum), UInt8(cbl / cwSum), 255)
                : SIMD4(140, 140, 140, 255)

            cellVert[cidx(cx, cy, cz)] = Int32(cp.count)
            cp.append(pos); cn.append(nrm); cc.append(color)
            ccov.append(l > 0 && isCovered(l, originVox &+ SIMD3(cx, cy, cz)))
        } } }
        if cp.isEmpty { return res }

        var outIdx = [Int32](repeating: -1, count: cp.count)
        @inline(__always) func emit(_ ci: Int) -> UInt32 {
            if outIdx[ci] < 0 {
                outIdx[ci] = Int32(res.pos.count)
                res.pos.append(cp[ci]); res.nrm.append(cn[ci]); res.col.append(cc[ci])
            }
            return UInt32(outIdx[ci])
        }

        // Quads: one per sign-changing grid edge owned by this block's voxels.
        for vz in 0..<B { for vy in 0..<B { for vx in 0..<B {
            let i0 = pidx(vx, vy, vz)
            if !valid[i0] { continue }
            for axis in 0..<3 {
                let i1 = pidx(vx + (axis == 0 ? 1 : 0), vy + (axis == 1 ? 1 : 0), vz + (axis == 2 ? 1 : 0))
                if !valid[i1] || (sdf[i0] >= 0) == (sdf[i1] >= 0) { continue }

                let ua = (axis + 1) % 3, wa = (axis + 2) % 3
                var q = [Int32](repeating: -1, count: 4)
                var bad = false
                var allCovered = true
                for (k, (du, dw)) in [(0, 0), (-1, 0), (-1, -1), (0, -1)].enumerated() {
                    var c = [vx, vy, vz]
                    c[ua] += du; c[wa] += dw
                    let v = cellVert[cidx(c[0], c[1], c[2])]
                    if v < 0 { bad = true; break }
                    q[k] = v
                    if !ccov[Int(v)] { allCovered = false }
                }
                if bad || allCovered { continue }

                let a = emit(Int(q[0])), b = emit(Int(q[1])), c = emit(Int(q[2])), d = emit(Int(q[3]))
                let gn = simd_cross(res.pos[Int(b)] - res.pos[Int(a)], res.pos[Int(c)] - res.pos[Int(a)])
                let nsum = res.nrm[Int(a)] + res.nrm[Int(b)] + res.nrm[Int(c)] + res.nrm[Int(d)]
                if simd_dot(gn, nsum) >= 0 {
                    res.idx.append(contentsOf: [a, b, c, a, c, d])
                } else {
                    res.idx.append(contentsOf: [a, d, c, a, c, b])
                }
            }
        } } }
        return res
    }
}
