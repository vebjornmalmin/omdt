import SceneKit
import simd

/// SceneKit geometry built from fused meshes and keyframe poses.
enum MeshGeometry {
    static func vertexColorMaterial() -> SCNMaterial {
        let m = SCNMaterial(); m.lightingModel = .lambert; m.isDoubleSided = true
        m.diffuse.contents = UIColor.white; return m
    }

    static func make(_ m: MeshOut, _ mat: SCNMaterial) -> SCNGeometry {
        let n = m.pos.count
        let vsrc = SCNGeometrySource(data: m.pos.withUnsafeBytes { Data($0) }, semantic: .vertex, vectorCount: n,
                                     usesFloatComponents: true, componentsPerVector: 3, bytesPerComponent: 4,
                                     dataOffset: 0, dataStride: MemoryLayout<SIMD3<Float>>.stride)
        let nsrc = SCNGeometrySource(data: m.nrm.withUnsafeBytes { Data($0) }, semantic: .normal, vectorCount: n,
                                     usesFloatComponents: true, componentsPerVector: 3, bytesPerComponent: 4,
                                     dataOffset: 0, dataStride: MemoryLayout<SIMD3<Float>>.stride)
        var colors = [SIMD4<Float>](repeating: .zero, count: n)
        for i in 0..<n {
            let c = m.col[i]
            colors[i] = SIMD4(Float(c.x), Float(c.y), Float(c.z), 255) * (1 / 255)
        }
        let csrc = SCNGeometrySource(data: colors.withUnsafeBytes { Data($0) }, semantic: .color, vectorCount: n,
                                     usesFloatComponents: true, componentsPerVector: 4, bytesPerComponent: 4,
                                     dataOffset: 0, dataStride: MemoryLayout<SIMD4<Float>>.stride)
        let element = SCNGeometryElement(data: m.idx.withUnsafeBytes { Data($0) }, primitiveType: .triangles,
                                         primitiveCount: m.idx.count / 3, bytesPerIndex: 4)
        let geo = SCNGeometry(sources: [vsrc, nsrc, csrc], elements: [element])
        geo.materials = [mat]
        return geo
    }

    /// Wireframe pyramids showing where each keyframe was taken and which way it looked.
    static func frustums(_ frames: [Keyframe], depth: Float, color: UIColor) -> SCNGeometry {
        var pts: [SIMD3<Float>] = []
        var idx: [UInt32] = []
        for k in frames {
            let t = k.transform
            let fx = k.intrinsics[0], fy = k.intrinsics[1], cx = k.intrinsics[2], cy = k.intrinsics[3]
            let w = Float(k.width), h = Float(k.height)
            let base = UInt32(pts.count)
            pts.append(xformPoint(t, .zero))
            for (u, v) in [(0, 0), (w, 0), (w, h), (0, h)] as [(Float, Float)] {
                let dir = SIMD3((u - cx) / fx, -(v - cy) / fy, -1) * depth
                pts.append(xformPoint(t, dir))
            }
            for c in 1...4 {
                idx += [base, base + UInt32(c)]                               // apex to corner
                idx += [base + UInt32(c), base + UInt32(c % 4 + 1)]           // image rectangle
            }
            // Short tick on the image's top edge (sensor "up" is the portrait right side).
            let top = (pts[Int(base) + 1] + pts[Int(base) + 2]) / 2
            pts.append(top + (top - pts[Int(base)]) * 0.15)
            idx += [base + 1, UInt32(pts.count - 1), UInt32(pts.count - 1), base + 2]
        }
        let src = SCNGeometrySource(vertices: pts.map { SCNVector3($0) })
        let el = SCNGeometryElement(indices: idx, primitiveType: .line)
        let geo = SCNGeometry(sources: [src], elements: [el])
        let mat = SCNMaterial(); mat.lightingModel = .constant; mat.diffuse.contents = color
        geo.materials = [mat]
        return geo
    }
}
