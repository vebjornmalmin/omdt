import Foundation
import simd

/// Binary little-endian PLY with per-vertex color and normals, written straight into a
/// preallocated buffer by all cores at once.
enum MeshExporter {
    static let vertexBytes = 27   // 6 floats + 3 bytes
    static let faceBytes = 13     // count byte + 3 int32

    static func plyData(_ meshes: [MeshOut]) throws -> Data {
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
        let vBytes = vertexBytes, fBytes = faceBytes
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
        return data
    }

    /// Reads a PLY written by `plyData` back into a single mesh.
    static func readPLY(_ url: URL) throws -> MeshOut {
        let data = try Data(contentsOf: url, options: .mappedIfSafe)
        func fail(_ why: String) -> NSError {
            NSError(domain: "PLY", code: 1, userInfo: [NSLocalizedDescriptionKey: "Can't read mesh: \(why)"])
        }
        let marker = Array("end_header\n".utf8)
        guard let end = data.prefix(4096).firstRange(of: marker) else { throw fail("no header") }
        guard let header = String(data: data[..<end.lowerBound], encoding: .ascii) else { throw fail("bad header") }

        let lines = header.split(separator: "\n").map(String.init)
        let expected = ["property float x", "property float y", "property float z",
                        "property float nx", "property float ny", "property float nz",
                        "property uchar red", "property uchar green", "property uchar blue",
                        "property list uchar int vertex_indices"]
        guard lines.first == "ply", lines.contains("format binary_little_endian 1.0"),
              lines.filter({ $0.hasPrefix("property") }) == expected else { throw fail("unsupported layout") }
        func count(_ element: String) -> Int? {
            lines.first { $0.hasPrefix("element \(element) ") }.flatMap { Int($0.split(separator: " ")[2]) }
        }
        guard let V = count("vertex"), let F = count("face") else { throw fail("missing counts") }

        let vStart = end.upperBound - data.startIndex, fStart = vStart + V * vertexBytes
        guard data.count >= fStart + F * faceBytes else { throw fail("file is truncated") }

        var m = MeshOut()
        m.pos = .init(repeating: .zero, count: V)
        m.nrm = .init(repeating: .zero, count: V)
        m.col = .init(repeating: .zero, count: V)
        m.idx = .init(repeating: 0, count: F * 3)
        try data.withUnsafeBytes { raw in
            let b = raw.baseAddress!
            func f(_ o: Int) -> Float { b.loadUnaligned(fromByteOffset: o, as: Float.self) }
            for i in 0..<V {
                let o = vStart + i * vertexBytes
                m.pos[i] = SIMD3(f(o), f(o + 4), f(o + 8))
                m.nrm[i] = SIMD3(f(o + 12), f(o + 16), f(o + 20))
                m.col[i] = SIMD4(raw[o + 24], raw[o + 25], raw[o + 26], 255)
            }
            for i in 0..<F {
                let o = fStart + i * faceBytes
                guard raw[o] == 3 else { throw fail("non-triangle face") }
                for k in 0..<3 {
                    let v = b.loadUnaligned(fromByteOffset: o + 1 + 4 * k, as: Int32.self)
                    guard v >= 0, Int(v) < V else { throw fail("bad vertex index") }
                    m.idx[i * 3 + k] = UInt32(v)
                }
            }
        }
        return m
    }
}
