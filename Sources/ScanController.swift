import ARKit
import SceneKit
import SwiftUI

struct ShareItem: Identifiable {
    let id = UUID()
    let urls: [URL]
}

final class ScanController: NSObject, ObservableObject, ARSessionDelegate {
    @Published var isScanning = false
    @Published var showColors = true
    @Published var triangleCount = 0
    @Published var detailText = "–"
    @Published var trackingText = "Tap record to begin"
    @Published var isExporting = false
    @Published var shareItem: ShareItem?
    @Published var errorMessage: String?

    let isSupported = ARWorldTrackingConfiguration.isSupported
        && ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth)

    let view: ARSCNView = {
        let v = ARSCNView(frame: .zero)
        v.scene = SCNScene()
        v.antialiasingMode = .none
        v.preferredFramesPerSecond = 60
        v.automaticallyUpdatesLighting = false
        v.autoenablesDefaultLighting = true
        v.rendersMotionBlur = false
        v.rendersCameraGrain = false
        return v
    }()

    private let vol = TSDFVolume()
    private let sessionQueue = DispatchQueue(label: "scan.session", qos: .userInteractive)
    private let workQueue = DispatchQueue(label: "scan.work", qos: .userInitiated)

    private let colorMat: SCNMaterial = {
        let m = SCNMaterial(); m.lightingModel = .lambert; m.isDoubleSided = true
        m.diffuse.contents = UIColor.white; return m
    }()
    private let wireMat: SCNMaterial = {
        let m = SCNMaterial(); m.lightingModel = .constant; m.isDoubleSided = true
        m.diffuse.contents = UIColor.cyan; m.fillMode = .lines; return m
    }()

    // Touched from several threads.
    private let lock = NSLock()
    private var busy = false
    private var colored = true
    private var hasStarted = false
    private var lastPos = SIMD3<Float>(repeating: .infinity)
    private var lastFwd = SIMD3<Float>(0, 0, -1)
    private var lastPassTime: TimeInterval = 0

    // Work queue only.
    private var triCounts: [Int32: Int] = [:]
    private var totalTris = 0

    // Main thread only.
    private var nodes: [Int32: SCNNode] = [:]

    override init() {
        super.init()
        view.session.delegate = self
        view.session.delegateQueue = sessionQueue
    }

    private var isColored: Bool { lock.lock(); defer { lock.unlock() }; return colored }

    // MARK: controls

    func toggleScanning() { isScanning ? pause() : start() }

    private func start() {
        guard isSupported else { return }
        let config = ARWorldTrackingConfiguration()
        config.sceneReconstruction = []          // our own fusion replaces ARKit's mesh
        config.frameSemantics = .sceneDepth
        config.environmentTexturing = .none
        config.isLightEstimationEnabled = false
        config.planeDetection = []
        lock.lock(); let first = !hasStarted; hasStarted = true; lock.unlock()
        view.session.run(config, options: first ? [.resetTracking, .removeExistingAnchors] : [])
        isScanning = true
    }

    private func pause() {
        view.session.pause()
        isScanning = false
        trackingText = "Paused"
    }

    func reset() {
        let was = isScanning
        view.session.pause()
        isScanning = false
        lock.lock(); hasStarted = false; lastPos = SIMD3(repeating: .infinity); lock.unlock()
        for n in nodes.values { n.removeFromParentNode() }
        nodes.removeAll()
        triangleCount = 0; detailText = "–"
        trackingText = "Tap record to begin"
        workQueue.async { [self] in
            vol.reset(); triCounts.removeAll(); totalTris = 0
        }
        if was { start() }
    }

    func setShowColors(_ on: Bool) {
        showColors = on
        lock.lock(); colored = on; lock.unlock()
        let mat = on ? colorMat : wireMat
        for n in nodes.values { n.geometry?.materials = [mat] }
    }

    func export() {
        guard !isExporting else { return }
        if isScanning { pause() }
        isExporting = true
        workQueue.async { [self] in
            do {
                let meshes = vol.mesh(blocks: vol.allBlocks)
                let url = try MeshExporter.exportPLY(meshes)
                DispatchQueue.main.async { self.isExporting = false; self.shareItem = ShareItem(urls: [url]) }
            } catch {
                DispatchQueue.main.async { self.isExporting = false; self.errorMessage = error.localizedDescription }
            }
        }
    }

    // MARK: ARSessionDelegate (session queue)

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        guard frame.sceneDepth != nil else { return }
        let T = frame.camera.transform
        let pos = SIMD3(T.columns.3.x, T.columns.3.y, T.columns.3.z)
        let fwd = SIMD3(T.columns.2.x, T.columns.2.y, T.columns.2.z)

        lock.lock()
        let moved = simd_distance(pos, lastPos) > 0.006 || 1 - simd_dot(fwd, lastFwd) > 0.0003
            || frame.timestamp - lastPassTime > 0.5
        if busy || !moved { lock.unlock(); return }
        busy = true; lastPos = pos; lastFwd = fwd; lastPassTime = frame.timestamp
        lock.unlock()

        workQueue.async { [self] in
            runPass(frame, cameraPos: pos)
            lock.lock(); busy = false; lock.unlock()
        }
    }

    func session(_ session: ARSession, cameraDidChangeTrackingState camera: ARCamera) {
        let text: String
        switch camera.trackingState {
        case .normal: text = "Tracking"
        case .notAvailable: text = "Tracking unavailable"
        case .limited(let r):
            switch r {
            case .excessiveMotion: text = "Move slower"
            case .insufficientFeatures: text = "Need more detail / light"
            case .initializing: text = "Initializing…"
            case .relocalizing: text = "Relocalizing…"
            @unknown default: text = "Limited"
            }
        }
        DispatchQueue.main.async { self.trackingText = text }
    }

    func session(_ session: ARSession, didFailWithError error: Error) {
        DispatchQueue.main.async { self.errorMessage = error.localizedDescription; self.isScanning = false }
    }

    // MARK: fusion pass (work queue)

    private func runPass(_ frame: ARFrame, cameraPos: SIMD3<Float>) {
        guard let sd = frame.sceneDepth else { return }
        let dm = sd.depthMap
        let cm = sd.confidenceMap
        let img = frame.capturedImage
        guard CVPixelBufferGetPlaneCount(img) >= 2 else { return }

        CVPixelBufferLockBaseAddress(dm, .readOnly)
        if let cm { CVPixelBufferLockBaseAddress(cm, .readOnly) }
        CVPixelBufferLockBaseAddress(img, .readOnly)
        defer {
            CVPixelBufferUnlockBaseAddress(dm, .readOnly)
            if let cm { CVPixelBufferUnlockBaseAddress(cm, .readOnly) }
            CVPixelBufferUnlockBaseAddress(img, .readOnly)
        }
        guard let dBase = CVPixelBufferGetBaseAddress(dm),
              let yBase = CVPixelBufferGetBaseAddressOfPlane(img, 0),
              let cBase = CVPixelBufferGetBaseAddressOfPlane(img, 1) else { return }

        let dw = CVPixelBufferGetWidth(dm), dh = CVPixelBufferGetHeight(dm)
        let iw = CVPixelBufferGetWidthOfPlane(img, 0), ih = CVPixelBufferGetHeightOfPlane(img, 0)
        let K = frame.camera.intrinsics
        let s = Float(dw) / Float(iw)

        let depthPtr = UnsafePointer(dBase.assumingMemoryBound(to: Float32.self))
        let df = DepthFrame(
            depth: depthPtr,
            conf: cm.flatMap { CVPixelBufferGetBaseAddress($0) }.map { UnsafePointer($0.assumingMemoryBound(to: UInt8.self)) },
            width: dw, height: dh,
            depthStride: CVPixelBufferGetBytesPerRow(dm) / MemoryLayout<Float32>.size,
            confStride: cm.map { CVPixelBufferGetBytesPerRow($0) } ?? 0,
            fx: K[0][0] * s, fy: K[1][1] * s, cx: K[2][0] * s, cy: K[2][1] * s,
            camToWorld: frame.camera.transform)
        let cf = ColorFrame(
            y: UnsafePointer(yBase.assumingMemoryBound(to: UInt8.self)),
            cbcr: UnsafePointer(cBase.assumingMemoryBound(to: UInt8.self)),
            yStride: CVPixelBufferGetBytesPerRowOfPlane(img, 0),
            cStride: CVPixelBufferGetBytesPerRowOfPlane(img, 1),
            width: iw, height: ih, fx: K[0][0], fy: K[1][1], cx: K[2][0], cy: K[2][1])

        vol.integrate(depth: df, color: isColored ? cf : nil)

        // Detail readout from the depth at the screen center.
        let cd = depthPtr[(dh / 2) * df.depthStride + dw / 2]
        let detail = (cd > TSDFVolume.minDepth && cd < TSDFVolume.maxDepth)
            ? String(format: "%.0f mm", TSDFVolume.voxelSize(TSDFVolume.level(forDepth: cd)) * 1000) : "–"

        // Re-mesh the nearest dirty blocks within a per-pass budget.
        let dirty = vol.takeDirty(budget: 400, camera: cameraPos)
        let meshes = vol.mesh(blocks: dirty)
        let mat = isColored ? colorMat : wireMat

        var geos = [SCNGeometry?](repeating: nil, count: dirty.count)
        geos.withUnsafeMutableBufferPointer { out in
            meshes.withUnsafeBufferPointer { mp in
                DispatchQueue.concurrentPerform(iterations: mp.count) { i in
                    if !mp[i].isEmpty { out[i] = Self.makeGeometry(mp[i], mat) }
                }
            }
        }
        for (b, m) in zip(dirty, meshes) {
            totalTris += m.idx.count / 3 - (triCounts[b] ?? 0)
            triCounts[b] = m.idx.count / 3
        }
        let tris = totalTris

        DispatchQueue.main.async { [self] in
            let root = view.scene.rootNode
            for (b, g) in zip(dirty, geos) {
                if let g {
                    let node = nodes[b] ?? { let n = SCNNode(); root.addChildNode(n); nodes[b] = n; return n }()
                    node.geometry = g
                } else if let n = nodes.removeValue(forKey: b) {
                    n.removeFromParentNode()
                }
            }
            triangleCount = tris
            detailText = detail
        }
    }

    private static func makeGeometry(_ m: MeshOut, _ mat: SCNMaterial) -> SCNGeometry {
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
}
