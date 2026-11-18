import Foundation
import simd

/// Binary little-endian PLY with per-vertex color and normals, written straight into a
/// preallocated buffer by all cores at once.
enum MeshExporter {
    static func exportPLY(_ meshes: [MeshOut]) throws -> URL {
        var vOff = [Int](), fOff = [Int]()
        var V = 0, F = 0
        for m in meshes { vOff.append(V); fOff.append(F); V += m.pos.count; F += m.idx.count / 3 }
        guard V > 0 else {
            throw NSError(domain: "Export", code: 1, userInfo: [NSLocalizedDescriptionKey: "Nothing scanned yet."])
        }

        let header = """
        ply
        format binary_little_endian 1.0
        element vertex \(V)
        property float x
        property float y
        property float z
        property float nx
        property float ny
        property float nz
        property uchar red
        property uchar green
        property uchar blue
        element face \(F)
        property list uchar int vertex_indices
        end_header

        """
        let head = Array(header.utf8)
        let vBytes = 27, fBytes = 13
        let vStart = head.count, fStart = vStart + V * vBytes
        var data = Data(count: fStart + F * fBytes)

        data.withUnsafeMutableBytes { raw in
            let base = raw.baseAddress!
            head.withUnsafeBytes { base.copyMemory(from: $0.baseAddress!, byteCount: head.count) }
            meshes.withUnsafeBufferPointer { mp in
                DispatchQueue.concurrentPerform(iterations: mp.count) { mi in
                    let m = mp[mi]
                    for i in 0..<m.pos.count {
                        let o = vStart + (vOff[mi] + i) * vBytes
                        let p = m.pos[i], n = m.nrm[i], c = m.col[i]
                        base.storeBytes(of: p.x, toByteOffset: o, as: Float.self)
                        base.storeBytes(of: p.y, toByteOffset: o + 4, as: Float.self)
                        base.storeBytes(of: p.z, toByteOffset: o + 8, as: Float.self)
                        base.storeBytes(of: n.x, toByteOffset: o + 12, as: Float.self)
                        base.storeBytes(of: n.y, toByteOffset: o + 16, as: Float.self)
                        base.storeBytes(of: n.z, toByteOffset: o + 20, as: Float.self)
                        base.storeBytes(of: c.x, toByteOffset: o + 24, as: UInt8.self)
                        base.storeBytes(of: c.y, toByteOffset: o + 25, as: UInt8.self)
                        base.storeBytes(of: c.z, toByteOffset: o + 26, as: UInt8.self)
                    }
                    for f in 0..<(m.idx.count / 3) {
                        let o = fStart + (fOff[mi] + f) * fBytes
                        base.storeBytes(of: 3, toByteOffset: o, as: UInt8.self)
                        for k in 0..<3 {
                            base.storeBytes(of: Int32(vOff[mi]) + Int32(m.idx[f * 3 + k]),
                                            toByteOffset: o + 1 + 4 * k, as: Int32.self)
                        }
                    }
                }
            }
        }

        let dir = try FileManager.default.url(for: .documentDirectory, in: .userDomainMask,
                                              appropriateFor: nil, create: true)
            .appendingPathComponent("Scans", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let fmt = DateFormatter(); fmt.dateFormat = "yyyyMMdd-HHmmss"
        let url = dir.appendingPathComponent("Scan-\(fmt.string(from: Date())).ply")
        try data.write(to: url, options: .atomic)
        return url
    }
}
