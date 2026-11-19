import SceneKit
import UIKit

/// SceneKit nodes for a fitted room: floor, see-through walls and outline edges.
enum RoomGeometry {
    static let wallThickness: Float = 0.06

    static func node(_ room: RoomShell) -> SCNNode {
        let root = SCNNode()
        let h = room.height

        // Floor: the outline as a thin slab. SCNShape draws in its x-y plane; turned flat,
        // local (x, y) lands on world (x, -y), so z is passed negated.
        let path = UIBezierPath()
        for (i, p) in room.outline.enumerated() {
            let pt = CGPoint(x: CGFloat(p[0]), y: CGFloat(-p[1]))
            if i == 0 { path.move(to: pt) } else { path.addLine(to: pt) }
        }
        path.close()
        let floor = SCNShape(path: path, extrusionDepth: 0.01)
        floor.materials = [material(UIColor(white: 0.55, alpha: 1), transparency: 0.85)]
        let floorNode = SCNNode(geometry: floor)
        floorNode.eulerAngles.x = -.pi / 2
        floorNode.simdPosition.y = room.floorY - 0.005
        root.addChildNode(floorNode)

        // Walls: thin boxes standing outside the outline, so the inner face is the fitted plane.
        // Teal where the wall was scanned, orange where it is mostly guessed.
        for w in room.walls {
            let a = SIMD2(w.start[0], w.start[1]), b = SIMD2(w.end[0], w.end[1])
            let n = SIMD2(w.normal[0], w.normal[1])
            let d = (b - a) / max(w.length, 1e-4)
            let box = SCNBox(width: CGFloat(w.length), height: CGFloat(h), length: CGFloat(wallThickness), chamferRadius: 0)
            let seen = w.coverage >= 0.5
            box.materials = [material(seen ? .systemTeal : .systemOrange, transparency: seen ? 0.35 : 0.25)]
            let node = SCNNode(geometry: box)
            let mid = (a + b) / 2 - n * (wallThickness / 2)
            node.simdPosition = SIMD3(mid.x, room.floorY + h / 2, mid.y)
            node.eulerAngles.y = atan2(-d.y, d.x)          // box x axis along the wall
            node.renderingOrder = 10
            root.addChildNode(node)
        }

        // Outline at floor and ceiling plus the vertical corners.
        var pts: [SCNVector3] = []
        var idx: [UInt32] = []
        let k = room.outline.count
        for (i, p) in room.outline.enumerated() {
            pts.append(SCNVector3(p[0], room.floorY, p[1]))
            pts.append(SCNVector3(p[0], room.ceilingY, p[1]))
            let j = UInt32(((i + 1) % k) * 2), me = UInt32(i * 2)
            idx += [me, j, me + 1, j + 1, me, me + 1]
        }
        let lines = SCNGeometry(sources: [SCNGeometrySource(vertices: pts)],
                                elements: [SCNGeometryElement(indices: idx, primitiveType: .line)])
        let lm = SCNMaterial(); lm.lightingModel = .constant; lm.diffuse.contents = UIColor.white
        lines.materials = [lm]
        let lineNode = SCNNode(geometry: lines)
        lineNode.renderingOrder = 20
        root.addChildNode(lineNode)
        return root
    }

    private static func material(_ color: UIColor, transparency: CGFloat) -> SCNMaterial {
        let m = SCNMaterial()
        m.lightingModel = .lambert
        m.diffuse.contents = color
        m.transparency = transparency
        m.isDoubleSided = true
        m.writesToDepthBuffer = transparency > 0.8
        return m
    }
}
