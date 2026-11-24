import Foundation
import Metal
import simd

/// Runs TSDF voxel integration on the GPU. Voxel blocks live in shared-memory buffers
/// (`BlockPool`), so the CPU meshes and saves straight from the same memory the GPU writes.
/// Same math as `TSDFVolume.integrateBlock`, one GPU thread per voxel.
final class GPUFusion {
    struct Block {
        var x: Int32, y: Int32, z: Int32, level: Int32
        /// Block slot inside its pool chunk.
        var slot: UInt32
        /// Index into the touched flags.
        var index: UInt32
    }

    struct Params {
        var w2c: simd_float4x4
        var fx: Float, fy: Float, cx: Float, cy: Float
        var width: Int32, height: Int32, depthStride: Int32, confStride: Int32
        var hasConf: Int32, hasColor: Int32
        var cfx: Float, cfy: Float, ccx: Float, ccy: Float
        var cwidth: Int32, cheight: Int32, yStride: Int32, cStride: Int32
        var minDepth: Float, maxDepth: Float
    }

    let device: MTLDevice
    private let queue: MTLCommandQueue
    private let pipeline: MTLComputePipelineState
    private var buffers: [String: MTLBuffer] = [:]

    /// GPU time of the last integration, in ms.
    private(set) var lastGPUms: Double = 0

    init?() {
        guard let dev = MTLCreateSystemDefaultDevice(), let q = dev.makeCommandQueue() else { return nil }
        let opts = MTLCompileOptions()
        opts.mathMode = .safe          // match the CPU results closely
        do {
            let lib = try dev.makeLibrary(source: Self.source, options: opts)
            guard let fn = lib.makeFunction(name: "integrate") else { return nil }
            pipeline = try dev.makeComputePipelineState(function: fn)
        } catch {
            print("GPU fusion unavailable: \(error)")
            return nil
        }
        device = dev; queue = q
    }

    /// A shared buffer of at least `bytes`, reused between frames.
    private func buffer(_ name: String, _ bytes: Int) -> MTLBuffer {
        if let b = buffers[name], b.length >= bytes { return b }
        let b = device.makeBuffer(length: max(bytes, 256) * 3 / 2, options: .storageModeShared)!
        buffers[name] = b
        return b
    }

    /// Integrates one frame into `blocks`. `chunks[i]` holds the voxels of blocks whose slot is in
    /// chunk i. Returns, per block, whether any voxel changed.
    func integrate(blocks: [(chunk: Int, block: Block)], chunks: [MTLBuffer],
                   depth d: DepthFrame, color: ColorFrame?, w2c: simd_float4x4) -> [Bool] {
        guard !blocks.isEmpty else { return [] }

        // Frame data, copied whole (row strides are kept).
        let dBytes = d.depthStride * d.height * 4
        let depthBuf = buffer("depth", dBytes)
        memcpy(depthBuf.contents(), d.depth, dBytes)
        let confBuf = buffer("conf", max(1, d.confStride * d.height))
        if let c = d.conf { memcpy(confBuf.contents(), c, d.confStride * d.height) }
        let yBuf = buffer("y", max(1, (color?.yStride ?? 0) * (color?.height ?? 0)))
        let cBuf = buffer("cbcr", max(1, (color?.cStride ?? 0) * ((color?.height ?? 0) + 1) / 2))
        if let c = color {
            memcpy(yBuf.contents(), c.y, c.yStride * c.height)
            memcpy(cBuf.contents(), c.cbcr, c.cStride * ((c.height + 1) / 2))
        }

        var p = Params(w2c: w2c, fx: d.fx, fy: d.fy, cx: d.cx, cy: d.cy,
                       width: Int32(d.width), height: Int32(d.height), depthStride: Int32(d.depthStride),
                       confStride: Int32(d.confStride), hasConf: d.conf == nil ? 0 : 1, hasColor: color == nil ? 0 : 1,
                       cfx: color?.fx ?? 0, cfy: color?.fy ?? 0, ccx: color?.cx ?? 0, ccy: color?.cy ?? 0,
                       cwidth: Int32(color?.width ?? 0), cheight: Int32(color?.height ?? 0),
                       yStride: Int32(color?.yStride ?? 0), cStride: Int32(color?.cStride ?? 0),
                       minDepth: TSDFVolume.minDepth, maxDepth: TSDFVolume.maxDepth)

        // Blocks grouped by chunk, one dispatch per chunk.
        let sorted = blocks.sorted { $0.chunk < $1.chunk }
        let stride = MemoryLayout<Block>.stride
        let blockBuf = buffer("blocks", sorted.count * stride)
        let bp = blockBuf.contents().bindMemory(to: Block.self, capacity: sorted.count)
        for (i, b) in sorted.enumerated() { bp[i] = b.block }
        let touchedBuf = buffer("touched", blocks.count * 4)
        memset(touchedBuf.contents(), 0, blocks.count * 4)

        guard let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else { return [] }
        enc.setComputePipelineState(pipeline)
        enc.setBytes(&p, length: MemoryLayout<Params>.stride, index: 2)
        enc.setBuffer(depthBuf, offset: 0, index: 3)
        enc.setBuffer(confBuf, offset: 0, index: 4)
        enc.setBuffer(yBuf, offset: 0, index: 5)
        enc.setBuffer(cBuf, offset: 0, index: 6)
        enc.setBuffer(touchedBuf, offset: 0, index: 7)
        let n3 = TSDFVolume.B * TSDFVolume.B * TSDFVolume.B
        let tg = min(256, pipeline.maxTotalThreadsPerThreadgroup)
        var start = 0
        while start < sorted.count {
            let chunk = sorted[start].chunk
            var end = start
            while end < sorted.count && sorted[end].chunk == chunk { end += 1 }
            var count = UInt32(end - start)
            enc.setBuffer(chunks[chunk], offset: 0, index: 0)
            enc.setBuffer(blockBuf, offset: start * stride, index: 1)
            enc.setBytes(&count, length: 4, index: 8)
            enc.dispatchThreads(MTLSize(width: Int(count) * n3, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
            start = end
        }
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        lastGPUms = (cb.gpuEndTime - cb.gpuStartTime) * 1000

        let tp = touchedBuf.contents().bindMemory(to: UInt32.self, capacity: blocks.count)
        // Touched flags are indexed by the caller's block order (Block.index).
        return (0..<blocks.count).map { tp[$0] != 0 }
    }

    // Kept in sync with TSDFVolume.integrateBlock.
    private static let source = """
    #include <metal_stdlib>
    using namespace metal;

    struct Voxel { short sdf; uchar w, r, g, b, cw; };
    struct Block { int x, y, z, level; uint slot; uint index; };
    struct Params {
        float4x4 w2c;
        float fx, fy, cx, cy;
        int width, height, depthStride, confStride;
        int hasConf, hasColor;
        float cfx, cfy, ccx, ccy;
        int cwidth, cheight, yStride, cStride;
        float minDepth, maxDepth;
    };

    static float voxelSize(int l) { return 0.002f * float(1 << l); }
    static float muOf(int l) { return max(4.0f * voxelSize(l), 0.012f); }
    static int levelFor(float d) { return d < 0.25f ? 0 : d < 0.5f ? 1 : d < 1.0f ? 2 : d < 2.0f ? 3 : 4; }

    kernel void integrate(device Voxel *vox [[buffer(0)]],
                          device const Block *blocks [[buffer(1)]],
                          constant Params &p [[buffer(2)]],
                          device const float *depth [[buffer(3)]],
                          device const uchar *conf [[buffer(4)]],
                          device const uchar *ly [[buffer(5)]],
                          device const uchar *cbcr [[buffer(6)]],
                          device atomic_uint *touched [[buffer(7)]],
                          constant uint &count [[buffer(8)]],
                          uint gid [[thread_position_in_grid]])
    {
        uint bi = gid >> 12;
        if (bi >= count) return;
        uint li = gid & 4095u;
        int x = int(li & 15u), y = int((li >> 4) & 15u), z = int(li >> 8);
        Block b = blocks[bi];
        int l = b.level;
        float vs = voxelSize(l), mu = muOf(l), invMu = 1.0f / mu;
        float3 pw = (float3(float(b.x), float(b.y), float(b.z)) * 16.0f + float3(float(x), float(y), float(z))) * vs;
        float4 pc = p.w2c * float4(pw, 1.0f);
        float zc = -pc.z;
        if (zc < p.minDepth) return;
        float iz = 1.0f / zc;
        float u = p.fx * pc.x * iz + p.cx;
        float v = p.cy - p.fy * pc.y * iz;
        if (u < 0.0f || v < 0.0f) return;
        int iu = int(u + 0.5f), iv = int(v + 0.5f);
        if (iu >= p.width || iv >= p.height) return;
        float dep = depth[iv * p.depthStride + iu];
        if (!(dep > p.minDepth && dep < p.maxDepth)) return;
        if (p.hasConf != 0 && conf[iv * p.confStride + iu] < 1) return;
        if (levelFor(dep) != l) return;
        float sdf = dep - zc;
        if (sdf < -mu) return;
        float t = min(1.0f, sdf * invMu);

        device Voxel &vx = vox[b.slot * 4096u + uint((z * 16 + y) * 16 + x)];
        float w = float(vx.w);
        float ns = (float(vx.sdf) * (1.0f / 32767.0f) * w + t) / (w + 1.0f);
        vx.sdf = short(ns * 32767.0f);
        vx.w = uchar(min(w + 1.0f, 64.0f));

        if (p.hasColor != 0 && fabs(sdf) < mu * 0.5f) {
            float cu = p.cfx * pc.x * iz + p.ccx;
            float cv = p.ccy - p.cfy * pc.y * iz;
            if (cu >= 0.0f && cv >= 0.0f) {
                int xi = int(cu), yi = int(cv);
                if (xi < p.cwidth && yi < p.cheight) {
                    float Y = float(ly[yi * p.yStride + xi]);
                    int ci = (yi >> 1) * p.cStride + (xi >> 1) * 2;
                    float cb = float(cbcr[ci]) - 128.0f, cr = float(cbcr[ci + 1]) - 128.0f;
                    float nr = clamp(Y + 1.402f * cr, 0.0f, 255.0f);
                    float ng = clamp(Y - 0.344136f * cb - 0.714136f * cr, 0.0f, 255.0f);
                    float nb = clamp(Y + 1.772f * cb, 0.0f, 255.0f);
                    float cw = float(vx.cw);
                    vx.r = uchar((float(vx.r) * cw + nr) / (cw + 1.0f));
                    vx.g = uchar((float(vx.g) * cw + ng) / (cw + 1.0f));
                    vx.b = uchar((float(vx.b) * cw + nb) / (cw + 1.0f));
                    vx.cw = uchar(min(cw + 1.0f, 8.0f));
                }
            }
        }
        atomic_store_explicit(&touched[b.index], 1u, memory_order_relaxed);
    }
    """
}
