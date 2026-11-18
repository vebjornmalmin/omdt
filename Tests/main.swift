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
