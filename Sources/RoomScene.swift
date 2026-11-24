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
    static func floorGeometry(_ shell: RoomShell) -> SCNGeometry {
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
        g.materials = [material(SIMD3(0.78, 0.76, 0.72))]
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

    /// The room as a scene for export: Floor, Walls and Objects groups.
    static func exportScene(_ room: RoomModel) -> SCNScene {
        let scene = SCNScene()
        let floor = SCNNode(geometry: floorGeometry(room.shell)); floor.name = "Floor"
        let walls = SCNNode(); walls.name = "Walls"
        for w in room.shell.walls { walls.addChildNode(wallNode(w, shell: room.shell)) }
        let objects = SCNNode(); objects.name = "Objects"
        for b in room.objects ?? [] { objects.addChildNode(boxNode(b)) }
        for n in [floor, walls, objects] { scene.rootNode.addChildNode(n) }
        return scene
    }

    struct ExportError: LocalizedError { var errorDescription: String? { "Could not write the USDZ file" } }

    static func writeUSDZ(_ room: RoomModel, to url: URL) throws {
        try? FileManager.default.removeItem(at: url)
        guard exportScene(room).write(to: url, options: nil, delegate: nil, progressHandler: nil) else {
            throw ExportError()
        }
    }
}

private extension String {
    var capitalizedFirst: String { prefix(1).uppercased() + dropFirst() }
}
