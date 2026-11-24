import CoreGraphics
import Foundation
import SceneKit

/// Plain solid geometry of a room: floor, walls and one box per thing. Used by the Room screen and
/// for USDZ export. Only SceneKit and CoreGraphics, so the Mac tests can export too.
enum RoomScene {
    static let wallThickness: Float = 0.06

    static func color(_ rgb: SIMD3<Float>, alpha: CGFloat = 1) -> CGColor {
        CGColor(red: CGFloat(rgb.x), green: CGFloat(rgb.y), blue: CGFloat(rgb.z), alpha: alpha)
    }

    static func material(_ rgb: SIMD3<Float>, transparency: CGFloat = 1) -> SCNMaterial {
        let m = SCNMaterial()
        m.lightingModel = .physicallyBased
        m.diffuse.contents = color(rgb)
        m.roughness.contents = 0.85
        m.metalness.contents = 0.0
        m.transparency = transparency
        m.isDoubleSided = true
        return m
    }

    /// The floor polygon as a flat surface facing up.
    static func floorGeometry(_ shell: RoomShell, rgb: SIMD3<Float> = SIMD3(0.78, 0.76, 0.72)) -> SCNGeometry {
        let pts = shell.outline.map { SIMD2($0[0], $0[1]) }
        // Ear clipping works counter-clockwise in (x, z); in y-up space that faces down, so each
        // triangle is flipped.
        let tris = PolygonTriangulator.triangulate(pts)
        var idx: [UInt32] = []
        for t in stride(from: 0, to: tris.count, by: 3) { idx += [UInt32(tris[t]), UInt32(tris[t + 2]), UInt32(tris[t + 1])] }
        let verts = pts.map { SCNVector3(SIMD3($0.x, shell.floorY, $0.y)) }
        let normals = [SCNVector3](repeating: SCNVector3(SIMD3<Float>(0, 1, 0)), count: verts.count)
        let g = SCNGeometry(sources: [SCNGeometrySource(vertices: verts), SCNGeometrySource(normals: normals)],
                            elements: [SCNGeometryElement(indices: idx, primitiveType: .triangles)])
        g.materials = [material(rgb)]
        return g
    }

    /// Floor and wall colors from the scan, or the plain defaults.
    static func floorRGB(_ room: RoomModel) -> SIMD3<Float> { vec(room.floorRGB) ?? SIMD3(0.78, 0.76, 0.72) }
    static func wallRGB(_ room: RoomModel) -> SIMD3<Float> { vec(room.wallRGB) ?? SIMD3(0.92, 0.91, 0.88) }
    private static func vec(_ a: [Float]?) -> SIMD3<Float>? { a.flatMap { $0.count == 3 ? SIMD3($0[0], $0[1], $0[2]) : nil } }

    /// A box's 3D object (its model's parts, or its scan piece), placed and turned like the box;
    /// nil when it has none yet (or its piece isn't loaded).
    static func objectNode(_ b: ObjectBox, piece: MeshOut?) -> SCNNode? {
        guard let m = b.model else { return nil }
        let node = SCNNode()
        place(node, b)
        node.name = exportName(b)
        switch m.source {
        case .modeled:
            if m.piece != nil, let piece { node.addChildNode(SCNNode(geometry: pieceGeometry(piece))) }
            for (i, p) in m.parts.enumerated() {
                let s = p.sizeVec
                let g: SCNGeometry = p.kind == .cylinder
                    ? SCNCylinder(radius: CGFloat(min(s.x, s.z) / 2), height: CGFloat(s.y))
                    : SCNBox(width: CGFloat(s.x), height: CGFloat(s.y), length: CGFloat(s.z),
                             chamferRadius: CGFloat(min(0.008, s.min() / 4)))
                g.materials = [material(SIMD3(p.rgb[0], p.rgb[1], p.rgb[2]))]
                let child = SCNNode(geometry: g)
                child.simdPosition = p.centerVec
                child.name = String(p.name.map { $0.isLetter || $0.isNumber ? $0 : "_" }) + "_\(i)"
                node.addChildNode(child)
            }
        case .scanned:
            guard let piece else { return nil }
            node.addChildNode(SCNNode(geometry: pieceGeometry(piece)))
        }
        return node
    }

    /// A scan piece with its own colors.
    static func pieceGeometry(_ m: MeshOut) -> SCNGeometry {
        let n = m.pos.count
        let vsrc = SCNGeometrySource(vertices: m.pos.map { SCNVector3($0) })
        let nsrc = SCNGeometrySource(normals: m.nrm.map { SCNVector3($0) })
        var colors = [SIMD4<Float>](repeating: SIMD4(0.6, 0.6, 0.6, 1), count: n)
        if m.col.count == n {
            for i in 0..<n { let c = m.col[i]; colors[i] = SIMD4(Float(c.x), Float(c.y), Float(c.z), 255) / 255 }
        }
        let csrc = SCNGeometrySource(data: colors.withUnsafeBytes { Data($0) }, semantic: .color, vectorCount: n,
                                     usesFloatComponents: true, componentsPerVector: 4, bytesPerComponent: 4,
                                     dataOffset: 0, dataStride: MemoryLayout<SIMD4<Float>>.stride)
        let element = SCNGeometryElement(indices: m.idx, primitiveType: .triangles)
        let g = SCNGeometry(sources: [vsrc, nsrc, csrc], elements: [element])
        let mat = SCNMaterial()
        mat.lightingModel = .physicallyBased
        mat.diffuse.contents = CGColor(red: 1, green: 1, blue: 1, alpha: 1)
        mat.roughness.contents = 0.9
        mat.isDoubleSided = true
        g.materials = [mat]
        return g
    }

    /// A wall as a thin slab standing just outside the outline, so its inner face is the fitted plane.
    static func wallNode(_ w: RoomShell.Wall, shell: RoomShell, rgb: SIMD3<Float> = SIMD3(0.92, 0.91, 0.88)) -> SCNNode {
        let a = SIMD2(w.start[0], w.start[1]), b = SIMD2(w.end[0], w.end[1])
        let n = SIMD2(w.normal[0], w.normal[1])
        let d = (b - a) / max(w.length, 1e-4)
        let box = SCNBox(width: CGFloat(w.length + wallThickness * 2), height: CGFloat(shell.height),
                         length: CGFloat(wallThickness), chamferRadius: 0)
        box.materials = [material(rgb)]
        let node = SCNNode(geometry: box)
        let mid = (a + b) / 2 - n * (wallThickness / 2)
        node.simdPosition = SIMD3(mid.x, shell.floorY + shell.height / 2, mid.y)
        node.simdEulerAngles = SIMD3(0, atan2(-d.y, d.x), 0)      // box x axis along the wall
        node.name = "Wall_\(w.id)"
        return node
    }

    /// A solid box placed and turned like `b`.
    static func boxNode(_ b: ObjectBox, transparency: CGFloat = 1) -> SCNNode {
        let g = SCNBox(width: CGFloat(b.size[0]), height: CGFloat(b.size[1]), length: CGFloat(b.size[2]),
                       chamferRadius: CGFloat(min(0.01, b.size.min()! / 4)))
        g.materials = [material(BoxCategory.of(b.label).rgb, transparency: transparency)]
        let node = SCNNode(geometry: g)
        place(node, b)
        node.name = exportName(b)
        return node
    }

    /// Puts a node at a box's center, turned with it (local x along the box's first axis).
    static func place(_ node: SCNNode, _ b: ObjectBox) {
        node.simdPosition = b.centerVec
        node.simdEulerAngles = SIMD3(0, -b.yaw, 0)
    }

    /// A name that is a valid USD prim name: "Grey_sofa_12".
    static func exportName(_ b: ObjectBox) -> String {
        let base = (b.label ?? "Box").capitalizedFirst
        let clean = String(base.map { $0.isLetter || $0.isNumber ? $0 : "_" })
        return (clean.first?.isNumber == true ? "_" : "") + clean + "_\(b.id)"
    }

    /// The room as a scene for export: Floor, Walls and Objects groups. Boxes with a 3D object
    /// (and its piece in `pieces`, for scanned ones) export as that object, others as boxes.
    static func exportScene(_ room: RoomModel, pieces: [String: MeshOut] = [:]) -> SCNScene {
        let scene = SCNScene()
        let floor = SCNNode(geometry: floorGeometry(room.shell, rgb: floorRGB(room))); floor.name = "Floor"
        let walls = SCNNode(); walls.name = "Walls"
        for w in room.shell.walls { walls.addChildNode(wallNode(w, shell: room.shell, rgb: wallRGB(room))) }
        let objects = SCNNode(); objects.name = "Objects"
        for b in room.objects ?? [] {
            objects.addChildNode(objectNode(b, piece: b.model?.piece.flatMap { pieces[$0] }) ?? boxNode(b))
        }
        for n in [floor, walls, objects] { scene.rootNode.addChildNode(n) }
        return scene
    }

    struct ExportError: LocalizedError { var errorDescription: String? { "Could not write the USDZ file" } }

    static func writeUSDZ(_ room: RoomModel, pieces: [String: MeshOut] = [:], to url: URL) throws {
        try? FileManager.default.removeItem(at: url)
        guard exportScene(room, pieces: pieces).write(to: url, options: nil, delegate: nil, progressHandler: nil) else {
            throw ExportError()
        }
    }
}

private extension String {
    var capitalizedFirst: String { prefix(1).uppercased() + dropFirst() }
}
