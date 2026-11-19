import ARKit
import SceneKit
import SwiftUI

final class ScanController: NSObject, ObservableObject, ARSessionDelegate {
    @Published var isScanning = false
    @Published var showColors = true
    @Published var triangleCount = 0
    @Published var detailText = "–"
    @Published var trackingText = "Starting camera…"
    @Published var keyframeCount = 0
    @Published var isSaving = false
    /// Set after a successful save; the view opens it.
    @Published var savedScan: ScanBundle?
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
    private let recorder = KeyframeRecorder()
    private let sessionQueue = DispatchQueue(label: "scan.session", qos: .userInteractive)
    private let workQueue = DispatchQueue(label: "scan.work", qos: .userInitiated)

    private let colorMat = MeshGeometry.vertexColorMaterial()
    private let wireMat: SCNMaterial = {
        let m = SCNMaterial(); m.lightingModel = .constant; m.isDoubleSided = true
        m.diffuse.contents = UIColor.cyan; m.fillMode = .lines; return m
    }()

    // Touched from several threads.
    private let lock = NSLock()
    private var busy = false
    private var colored = true
    private var hasStarted = false
    /// Fusion and keyframes run only while recording; the camera preview runs all the time.
    private var recording = false
    private var trackingNormal = false
    private var lastPos = SIMD3<Float>(repeating: .infinity)
    private var lastFwd = SIMD3<Float>(0, 0, -1)
    private var lastPassTime: TimeInterval = 0

    // Work queue only.
    private var triCounts: [Int32: Int] = [:]
    private var totalTris = 0

    // Main thread only.
    private var nodes: [Int32: SCNNode] = [:]
    private var summaryTimer: Timer?
    private var sessionRunning = false
    /// Another screen covers the scanner; the camera stays off until it closes.
    private var covered = false
    private var needsTrackingReset = false
    private let debug = Debug.shared

    override init() {
        super.init()
        view.session.delegate = self
        view.session.delegateQueue = sessionQueue
        recorder.onCount = { [weak self] n in self?.keyframeCount = n }
        workQueue.async { ScanBundle.removeStaleRecordings() }
        debug.log("scanner ready, supported \(isSupported)")
    }

    /// Starts the live camera when the scanner first appears.
    func startCamera() {
        guard !sessionRunning, !covered else { return }
        runSession(reset: true)
    }

    private var isColored: Bool { lock.lock(); defer { lock.unlock() }; return colored }

    // MARK: controls

    func toggleScanning() { isScanning ? pause() : start() }

    /// Runs the camera and tracking. Called at launch, after reset and when the scans sheet closes,
    /// so the live picture is always there and tracking is ready before recording starts.
    private func runSession(reset: Bool) {
        guard isSupported else { return }
        let config = ARWorldTrackingConfiguration()
        config.sceneReconstruction = []          // our own fusion replaces ARKit's mesh
        config.frameSemantics = .sceneDepth
        config.environmentTexturing = .none
        config.isLightEstimationEnabled = false
        config.planeDetection = []
        let vf = config.videoFormat
        debug.update { $0.videoFormat = "\(Int(vf.imageResolution.width))×\(Int(vf.imageResolution.height)) @\(vf.framesPerSecond)" }
        debug.log("session.run (\(reset ? "reset" : "resume")), video \(debug.snapshot().videoFormat), "
                  + "app state \(UIApplication.shared.applicationState.rawValue) (0=active)")
        view.session.run(config, options: reset ? [.resetTracking, .removeExistingAnchors] : [])
        sessionRunning = true
        summaryTimer?.invalidate()
        summaryTimer = Timer.scheduledTimer(withTimeInterval: 3, repeats: true) { [weak self] _ in
            guard let self else { return }
            self.debug.log(self.debug.summaryLine())
        }
    }

    private func start() {
        guard isSupported else { return }
        if !sessionRunning || needsTrackingReset { runSession(reset: needsTrackingReset); needsTrackingReset = false }
        lock.lock(); let first = !hasStarted; hasStarted = true; recording = true; let normal = trackingNormal; lock.unlock()
        if first {
            do { try recorder.begin() } catch { errorMessage = "Photos won't be saved: \(error.localizedDescription)" }
        }
        debug.log("recording \(first ? "started" : "resumed")")
        isScanning = true
        if normal { trackingText = "Tracking" }
    }

    func pauseIfScanning() { if isScanning { pause() } }

    /// Stops recording; the camera keeps running.
    private func pause() {
        debug.log("recording paused")
        lock.lock(); recording = false; lock.unlock()
        isScanning = false
        trackingText = "Paused – tap record"
    }

    /// Stops the camera while another screen covers the scanner. Recording pauses too.
    func suspendCamera() {
        covered = true
        pauseIfScanning()
        guard sessionRunning else { return }
        debug.log("session.pause (covered)")
        summaryTimer?.invalidate(); summaryTimer = nil
        view.session.pause()
        sessionRunning = false
    }

    func resumeCamera() {
        covered = false
        guard !sessionRunning || needsTrackingReset else { return }
        runSession(reset: needsTrackingReset)
        needsTrackingReset = false
    }

    /// Clears the scan and restarts tracking with a fresh origin. The camera keeps running.
    func reset() {
        let was = isScanning
        debug.log("reset (was scanning \(was))")
        lock.lock(); hasStarted = false; recording = false; trackingNormal = false
        lastPos = SIMD3(repeating: .infinity); lock.unlock()
        isScanning = false
        recorder.cancel()
        for n in nodes.values { n.removeFromParentNode() }
        nodes.removeAll()
        triangleCount = 0; detailText = "–"
        trackingText = "Starting camera…"
        workQueue.async { [self] in
            vol.reset(); triCounts.removeAll(); totalTris = 0
        }
        // Queued after the volume reset, so no stale pass lands in the new scan.
        workQueue.async { DispatchQueue.main.async { [self] in
            if covered { needsTrackingReset = true; return }
            runSession(reset: true)
            if was { start() }
        } }
    }

    func setShowColors(_ on: Bool) {
        showColors = on
        lock.lock(); colored = on; lock.unlock()
        let mat = on ? colorMat : wireMat
        for n in nodes.values { n.geometry?.materials = [mat] }
    }

    /// Saves mesh and keyframes as a scan bundle, then clears the scanner for the next room.
    func save() {
        guard !isSaving else { return }
        if isScanning { pause() }
        isSaving = true
        debug.log("save started")
        let t0 = Date()
        workQueue.async { [self] in
            // Recording is off, so no new fusion passes or keyframes start; queued ones finish first.
            let meshes = vol.mesh(blocks: vol.allBlocks)
            guard meshes.contains(where: { !$0.isEmpty }) else {
                DispatchQueue.main.async { self.isSaving = false; self.errorMessage = "Nothing scanned yet." }
                return
            }
            // Without a recording (photos failed to start) the mesh is still saved.
            var rec: (dir: URL, keyframes: [Keyframe])?
            do {
                rec = try recorder.finish() ?? (ScanBundle.makeRecordingDirectory(), [])
                let bundle = try ScanBundle.finalize(recording: rec!.dir, meshes: meshes, keyframes: rec!.keyframes)
                debug.log(String(format: "saved %@: %d keyframes, %d tris in %.1f s", bundle.meta.id,
                                 bundle.meta.keyframeCount, bundle.meta.triangleCount, Date().timeIntervalSince(t0)))
                DispatchQueue.main.async {
                    self.isSaving = false
                    self.reset()
                    self.savedScan = bundle
                }
            } catch {
                // The recording has been handed over, so this scan can't continue recording photos.
                if let d = rec?.dir { try? FileManager.default.removeItem(at: d) }
                debug.log("save FAILED: \(error)")
                DispatchQueue.main.async {
                    self.isSaving = false
                    self.errorMessage = "Couldn't save the scan: \(error.localizedDescription)"
                    self.reset()
                }
            }
        }
    }

    // MARK: ARSessionDelegate (session queue)

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        debug.frame(frame)
        lock.lock(); let rec = recording; lock.unlock()
        guard rec, frame.sceneDepth != nil else { return }
        recorder.consider(frame)
        let T = frame.camera.transform
        let pos = SIMD3(T.columns.3.x, T.columns.3.y, T.columns.3.z)
        let fwd = SIMD3(T.columns.2.x, T.columns.2.y, T.columns.2.z)

        lock.lock()
        let moved = simd_distance(pos, lastPos) > 0.006 || 1 - simd_dot(fwd, lastFwd) > 0.0003
            || frame.timestamp - lastPassTime > 0.5
        if busy || !moved {
            lock.unlock()
            if busy { debug.update { $0.skippedBusy += 1 } }
            return
        }
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
        case .normal: text = "" // filled in below
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
        let state = Debug.describe(camera.trackingState)
        debug.update { $0.tracking = state; $0.trackingSince = Date() }
        debug.log("tracking -> \(state)")
        var normal = false
        if case .normal = camera.trackingState { normal = true }
        lock.lock(); trackingNormal = normal; let rec = recording; lock.unlock()
        let shown = normal ? (rec ? "Tracking" : "Ready – tap record") : text
        DispatchQueue.main.async { self.trackingText = shown }
    }

    func session(_ session: ARSession, didFailWithError error: Error) {
        let e = error as NSError
        debug.update { $0.errors += 1 }
        debug.log("session FAILED: \(e.domain) \(e.code) \(e.localizedDescription) \(e.userInfo)")
        lock.lock(); recording = false; lock.unlock()
        DispatchQueue.main.async { self.errorMessage = error.localizedDescription; self.isScanning = false }
    }

    func sessionWasInterrupted(_ session: ARSession) {
        debug.update { $0.interruptions += 1 }
        debug.log("session INTERRUPTED")
    }

    func sessionInterruptionEnded(_ session: ARSession) {
        debug.log("session interruption ended")
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

        let tPass = Date()
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
                    if !mp[i].isEmpty { out[i] = MeshGeometry.make(mp[i], mat) }
                }
            }
        }
        for (b, m) in zip(dirty, meshes) {
            totalTris += m.idx.count / 3 - (triCounts[b] ?? 0)
            triCounts[b] = m.idx.count / 3
        }
        let tris = totalTris
        let ms = Date().timeIntervalSince(tPass) * 1000, blocks = vol.blockCount
        debug.update { s in
            s.passes += 1; s.passMs = ms; s.blocks = blocks
            s.passMsAvg = s.passMsAvg == 0 ? ms : s.passMsAvg * 0.9 + ms * 0.1
        }

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
}
