import SceneKit
import UIKit

/// Wireframe boxes with floating labels, plus the floor outline. Used over the live camera
/// and in the saved-scan viewer.
final class BoxOverlay {
    let root = SCNNode()
    private var nodes: [Int: (frame: SCNNode, tag: SCNNode, text: String, color: UIColor)] = [:]
    private let outline = SCNNode()
    private let labelHeight: Float

    init(labelHeight: Float = 0.08) {
        self.labelHeight = labelHeight
        root.addChildNode(outline)
    }

    func update(_ boxes: [ObjectBox]) {
        let ids = Set(boxes.map(\.id))
        for (id, n) in nodes where !ids.contains(id) {
            n.frame.removeFromParentNode(); n.tag.removeFromParentNode(); nodes[id] = nil
        }
        for b in boxes {
            let color = Self.color(b.source)
            let text = b.label.map { b.source == "user" ? $0 : "\($0) \(Int((b.confidence ?? 0) * 100))%" } ?? ""
            var entry = nodes[b.id] ?? {
                let f = SCNNode(), t = SCNNode()
                f.renderingOrder = 100; t.renderingOrder = 110
                let bc = SCNBillboardConstraint(); bc.freeAxes = .all
                t.constraints = [bc]
                root.addChildNode(f); root.addChildNode(t)
                return (f, t, "\u{0}", UIColor.clear)
            }()
            entry.frame.geometry = Self.wireframe(b.corners, color: color)
            let top = b.centerVec + SIMD3(0, b.size[1] / 2 + labelHeight * 0.8, 0)
            entry.tag.simdPosition = top
            if text != entry.text || color != entry.color {
                entry.tag.geometry = text.isEmpty ? nil : Self.labelPlane(text, color: color, height: labelHeight)
                entry.text = text; entry.color = color
            }
            nodes[b.id] = entry
        }
    }

    func updateShell(_ shell: RoomShell?) {
        guard let shell, shell.outline.count >= 3 else { outline.geometry = nil; return }
        let pts = shell.outline.map { SCNVector3($0[0], shell.floorY + 0.01, $0[1]) }
        var idx: [UInt32] = []
        for i in 0..<pts.count { idx += [UInt32(i), UInt32((i + 1) % pts.count)] }
        let g = SCNGeometry(sources: [SCNGeometrySource(vertices: pts)],
                            elements: [SCNGeometryElement(indices: idx, primitiveType: .line)])
        g.materials = [Self.lineMaterial(.systemTeal)]
        outline.geometry = g
    }

    static func color(_ source: String?) -> UIColor {
        switch source {
        case "user": return .systemBlue
        case "model", "review": return .systemGreen
        case "vision": return .systemYellow
        default: return UIColor(white: 1, alpha: 0.8)
        }
    }

    private static func wireframe(_ c: [SIMD3<Float>], color: UIColor) -> SCNGeometry {
        let idx: [UInt32] = [0, 1, 1, 2, 2, 3, 3, 0, 4, 5, 5, 6, 6, 7, 7, 4, 0, 4, 1, 5, 2, 6, 3, 7]
        let g = SCNGeometry(sources: [SCNGeometrySource(vertices: c.map { SCNVector3($0) })],
                            elements: [SCNGeometryElement(indices: idx, primitiveType: .line)])
        g.materials = [lineMaterial(color)]
        return g
    }

    private static func lineMaterial(_ color: UIColor) -> SCNMaterial {
        let m = SCNMaterial()
        m.lightingModel = .constant
        m.diffuse.contents = color
        m.readsFromDepthBuffer = false     // stay visible through the mesh
        return m
    }

    /// A rounded name tag with a colored dot, as a plane `height` meters tall.
    static func labelPlane(_ text: String, color: UIColor, height labelHeight: Float) -> SCNGeometry {
        let font = UIFont.systemFont(ofSize: 40, weight: .semibold)
        let size = (text as NSString).size(withAttributes: [.font: font])
        let pad: CGFloat = 18, dot: CGFloat = 18
        let canvas = CGSize(width: ceil(size.width + pad * 3 + dot), height: ceil(size.height + pad))
        let image = UIGraphicsImageRenderer(size: canvas).image { _ in
            UIColor(white: 0, alpha: 0.7).setFill()
            UIBezierPath(roundedRect: CGRect(origin: .zero, size: canvas), cornerRadius: canvas.height / 2).fill()
            color.setFill()
            UIBezierPath(ovalIn: CGRect(x: pad, y: (canvas.height - dot) / 2, width: dot, height: dot)).fill()
            (text as NSString).draw(at: CGPoint(x: pad * 2 + dot, y: pad / 2),
                                    withAttributes: [.font: font, .foregroundColor: UIColor.white])
        }
        let h = CGFloat(labelHeight)
        let plane = SCNPlane(width: h * canvas.width / canvas.height, height: h)
        let m = SCNMaterial()
        m.lightingModel = .constant
        m.diffuse.contents = image
        m.readsFromDepthBuffer = false
        m.isDoubleSided = true
        plane.materials = [m]
        return plane
    }
}
