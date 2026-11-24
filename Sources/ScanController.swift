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
    /// What the save is doing now, for the button.
    @Published var saveStep = "Saving…"
    /// Set after a successful save; the view opens it.
    @Published var savedScan: ScanBundle?
    @Published var errorMessage: String?
    @Published var showObjects = true { didSet { overlay.root.isHidden = !showObjects } }
    @Published var objectCount = 0
    /// The saved scan being added to, when continuing one.
    @Published private(set) var continuing: ScanBundle?
    /// True until tracking has lined up with a continued scan's world map.
    @Published private(set) var isLiningUp = false

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
    /// Work queue only: when dirty blocks were last re-meshed.
    private var lastMeshTime: TimeInterval = 0

    // Work queue only.
    private var triCounts: [Int32: Int] = [:]
    private var totalTris = 0
    /// 3 cm surface samples per block, for the live room fit and boxes.
    private var blockSamples: [Int32: [RoomFitter.Sample]] = [:]
    private var samplesVersion = 0
    private var analyzedVersionOnWork = -1

    // Live room and boxes (main thread).
    private let analysisQueue = DispatchQueue(label: "scan.analysis", qos: .utility)
    private var analysisBusy = false
    private var liveTimer: Timer?
    private var lookTimer: Timer?
    private var tracker = BoxTracker()
    private let tagger = LiveTagger()
    private let overlay = BoxOverlay()
    private(set) var liveShell: RoomShell?

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
        view.scene.rootNode.addChildNode(overlay.root)
        tagger.onChange = { [weak self] in self?.redrawBoxes() }
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
    private func runSession(reset: Bool, worldMap: ARWorldMap? = nil) {
        guard isSupported else { return }
        let config = ARWorldTrackingConfiguration()
        config.sceneReconstruction = []          // our own fusion replaces ARKit's mesh
        config.frameSemantics = .sceneDepth
        config.environmentTexturing = .none
        config.isLightEstimationEnabled = false
        config.planeDetection = []
        config.initialWorldMap = worldMap
        let vf = config.videoFormat
        debug.update { $0.videoFormat = "\(Int(vf.imageResolution.width))×\(Int(vf.imageResolution.height)) @\(vf.framesPerSecond)" }
        debug.log("session.run (\(worldMap != nil ? "world map" : reset ? "reset" : "resume")), video \(debug.snapshot().videoFormat), "
                  + "app state \(UIApplication.shared.applicationState.rawValue) (0=active)")
        view.session.run(config, options: reset ? [.resetTracking, .removeExistingAnchors] : [])
        sessionRunning = true
        summaryTimer?.invalidate()
        summaryTimer = Timer.scheduledTimer(withTimeInterval: 3, repeats: true) { [weak self] _ in
            guard let self else { return }
            self.debug.log(self.debug.summaryLine())
        }
        startLiveTimers()
    }

    // MARK: live room and boxes

    /// Refits room and boxes every 1.5 s while the scan grows, and looks at one box in the
    /// camera every 0.4 s, also while paused, so labels keep improving.
    private func startLiveTimers() {
        liveTimer?.invalidate(); lookTimer?.invalidate()
        liveTimer = Timer.scheduledTimer(withTimeInterval: 1.5, repeats: true) { [weak self] _ in
            MainActor.assumeIsolated { self?.analyze() }
        }
        lookTimer = Timer.scheduledTimer(withTimeInterval: 0.4, repeats: true) { [weak self] _ in
            MainActor.assumeIsolated {
                guard let self, self.showObjects, !self.tracker.boxes.isEmpty,
                      let frame = self.view.session.currentFrame else { return }
                self.tagger.look(frame: frame, boxes: self.tracker.boxes)
            }
        }
    }

    private func stopLiveTimers() {
        liveTimer?.invalidate(); liveTimer = nil
        lookTimer?.invalidate(); lookTimer = nil
    }

    private func analyze() {
        guard !analysisBusy else { return }
        analysisBusy = true
        workQueue.async { [self] in
            let version = samplesVersion
            let samples = version == analyzedVersionOnWork ? nil : blockSamples.values.flatMap { $0 }
            analyzedVersionOnWork = version
            guard let samples, samples.count >= 500 else {
                DispatchQueue.main.async { self.analysisBusy = false }
                return
            }
            analysisQueue.async { [self] in
                let t0 = Date()
                let shell = try? RoomFitter.fit(samples: samples)
                let boxes = shell.map { ObjectFinder.find(samples, shell: $0) } ?? []
                let ms = Int(Date().timeIntervalSince(t0) * 1000)
                DispatchQueue.main.async { [self] in
                    analysisBusy = false
                    lock.lock(); let current = hasStarted; lock.unlock()
                    guard current else { return }       // reset in the meantime
                    liveShell = shell
                    let tracked = tracker.update(boxes)
                    tagger.forget(except: Set(tracked.map(\.id)))
                    overlay.updateShell(shell)
                    redrawBoxes()
                    let roomText = shell.map {
                        String(format: "%.1f×%.1f m, %.2f m high, %d walls · %d pts · %d ms",
                               $0.extent.max() ?? 0, $0.extent.min() ?? 0, $0.height, $0.walls.count, samples.count, ms)
                    } ?? "no floor yet · \(samples.count) pts"
                    debug.update { $0.room = roomText }
                }
            }
        }
    }

    private func redrawBoxes() {
        // Pieces the model says are wall, floor, door… are not drawn as things.
        let boxes = tagger.apply(to: tracker.boxes).filter { RoomCorrector.structureKind($0) == nil }
        overlay.update(boxes)
        objectCount = boxes.count
        let summary = "\(boxes.count) boxes, " + tagger.summary
        debug.update { $0.objects = summary }
    }

    /// Current boxes with their labels, for saving.
    var labeledBoxes: [ObjectBox] { tagger.apply(to: tracker.boxes) }

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
        stopLiveTimers()
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
        tracker = BoxTracker(); tagger.reset(); liveShell = nil
        overlay.update([]); overlay.updateShell(nil); objectCount = 0
        continuing = nil; isLiningUp = false
        debug.update { $0.room = "–"; $0.objects = "–" }
        workQueue.async { [self] in
            vol.reset(); triCounts.removeAll(); totalTris = 0
            blockSamples.removeAll(); samplesVersion += 1
        }
        // Queued after the volume reset, so no stale pass lands in the new scan.
        workQueue.async { DispatchQueue.main.async { [self] in
            if covered { needsTrackingReset = true; return }
            runSession(reset: true)
            if was { start() }
        } }
    }

    /// Reopens a saved scan to add to it: loads its volume and boxes and lines tracking up with
    /// its world map. Fusion and photos wait until ARKit has recognized the room.
    func continueScan(_ scan: ScanBundle) {
        guard isSupported else { return }
        let map: ARWorldMap
        do {
            let data = try Data(contentsOf: scan.worldMapURL)
            guard let m = try NSKeyedUnarchiver.unarchivedObject(ofClass: ARWorldMap.self, from: data) else {
                throw CocoaError(.fileReadCorruptFile)
            }
            map = m
        } catch {
            errorMessage = "This scan can't be continued: \(error.localizedDescription)"
            return
        }
        debug.log("continue \(scan.meta.id): world map \(map.rawFeaturePoints.points.count) points")
        reset()
        do { try recorder.begin(continuing: scan) } catch {
            errorMessage = "Photos won't be saved: \(error.localizedDescription)"
        }
        lock.lock(); hasStarted = true; lock.unlock()
        continuing = scan
        isLiningUp = true
        trackingText = "Point at the scanned room to line up…"
        let room = scan.loadRoom()
        tracker = BoxTracker(seed: room?.objects ?? [])
        liveShell = room?.shell
        overlay.updateShell(room?.shell)
        redrawBoxes()

        // reset() queued a fresh session; this runs after it, with the world map.
        workQueue.async { [self] in
            do { try vol.read(from: scan.volumeURL) } catch {
                debug.log("continue: volume FAILED \(error)")
                DispatchQueue.main.async { self.errorMessage = error.localizedDescription }
            }
            let blocks = vol.takeDirty(budget: Int.max, camera: .zero)
            let meshes = vol.mesh(blocks: blocks)
            let mat = isColored ? colorMat : wireMat
            var geos = [SCNGeometry?](repeating: nil, count: blocks.count)
            geos.withUnsafeMutableBufferPointer { out in
                meshes.withUnsafeBufferPointer { mp in
                    DispatchQueue.concurrentPerform(iterations: mp.count) { i in
                        if !mp[i].isEmpty { out[i] = MeshGeometry.make(mp[i], mat) }
                    }
                }
            }
            for (b, m) in zip(blocks, meshes) {
                totalTris += m.idx.count / 3 - (triCounts[b] ?? 0)
                triCounts[b] = m.idx.count / 3
                blockSamples[b] = m.isEmpty ? nil : RoomFitter.sample([m], cell: 0.03)
            }
            samplesVersion += 1
            let tris = totalTris
            DispatchQueue.main.async { [self] in
                let root = view.scene.rootNode
                for (b, g) in zip(blocks, geos) {
                    guard let g else { continue }
                    let n = SCNNode(geometry: g); root.addChildNode(n); nodes[b] = n
                }
                triangleCount = tris
                covered = false; needsTrackingReset = false
                runSession(reset: true, worldMap: map)
                debug.log("continue: loaded \(blocks.count) blocks, \(tris) tris")
            }
        }
    }

    func setShowColors(_ on: Bool) {
        showColors = on
        lock.lock(); colored = on; lock.unlock()
        let mat = on ? colorMat : wireMat
        for n in nodes.values { n.geometry?.materials = [mat] }
    }

    /// Saves mesh and keyframes as a scan bundle (or into the scan being continued), with the
    /// world map, volume and location needed to continue it later. Then clears the scanner.
    func save(location: GeoPoint?) {
        guard !isSaving else { return }
        if isScanning { pause() }
        isSaving = true
        saveStep = "Saving…"
        // Keep saving when the app goes to the background (a big scan takes a while).
        var bg = UIBackgroundTaskIdentifier.invalid
        bg = UIApplication.shared.beginBackgroundTask(withName: "Save scan") { [self] in
            debug.log("save: background time ran out")
            UIApplication.shared.endBackgroundTask(bg)
            if saveTask == bg { saveTask = .invalid }
        }
        saveTask = bg
        debug.log("save started; live boxes: \(tracker.boxes.count), " + tagger.summary)
        let liveBoxes = labeledBoxes
        let existing = continuing
        // The world map has to come from the running session, before anything is reset.
        view.session.getCurrentWorldMap { [self] map, error in
            var mapData: Data?
            if let map {
                mapData = try? NSKeyedArchiver.archivedData(withRootObject: map, requiringSecureCoding: true)
                debug.log("world map: \(map.anchors.count) anchors, \(map.rawFeaturePoints.points.count) points, "
                          + "\((mapData?.count ?? 0) / 1024) KB")
            } else {
                debug.log("world map not available: \(error?.localizedDescription ?? "?")")
            }
            DispatchQueue.main.async { self.writeScan(liveBoxes: liveBoxes, existing: existing, mapData: mapData, location: location) }
        }
    }

    private var saveTask = UIBackgroundTaskIdentifier.invalid

    private func endSave() {
        isSaving = false
        if saveTask != .invalid { UIApplication.shared.endBackgroundTask(saveTask); saveTask = .invalid }
    }

    private func writeScan(liveBoxes: [ObjectBox], existing: ScanBundle?, mapData: Data?, location: GeoPoint?) {
        let t0 = Date()
        workQueue.async { [self] in
            var tStep = Date()
            func step(_ done: String, next: String?) {
                debug.log(String(format: "save: %@ in %.1f s", done, Date().timeIntervalSince(tStep)))
                tStep = Date()
                if let next { DispatchQueue.main.async { self.saveStep = next } }
            }
            DispatchQueue.main.async { self.saveStep = "Building the mesh…" }
            // Recording is off, so no new fusion passes or keyframes start; queued ones finish first.
            let meshes = vol.mesh(blocks: vol.allBlocks)
            guard meshes.contains(where: { !$0.isEmpty }) else {
                DispatchQueue.main.async { self.endSave(); self.errorMessage = "Nothing scanned yet." }
                return
            }
            step("mesh", next: "Writing files…")
            // Without a recording (photos failed to start) the mesh is still saved.
            var rec: (dir: URL, keyframes: [Keyframe])?
            do {
                let finished = recorder.finish()
                if let existing {
                    rec = (existing.url, finished?.keyframes ?? (try? existing.keyframes()) ?? [])
                } else {
                    rec = try finished ?? (ScanBundle.makeRecordingDirectory(), [])
                }
                let bundle = try ScanBundle.finalize(recording: rec!.dir, meshes: meshes, keyframes: rec!.keyframes,
                                                     location: location, existing: existing)
                // What continuing later needs.
                step("mesh and photos written", next: nil)
                if let mapData { try? mapData.write(to: bundle.worldMapURL, options: .atomic) }
                do {
                    try vol.write(to: bundle.volumeURL)
                } catch {
                    debug.log("volume not saved: \(error)")
                }
                step("world map and volume written", next: "Finding walls and furniture…")
                // Final room and boxes from the whole mesh, keeping the labels found while scanning.
                if let shell = try? RoomFitter.fit(meshes) {
                    let found = ObjectFinder.find(RoomFitter.sample(meshes, cell: 0.03), shell: shell)
                    var t = BoxTracker(seed: liveBoxes)
                    let objects = Array(t.update(found).prefix(found.count))
                    let before = existing?.loadRoom()
                    let (room, notes) = RoomCorrector.correct(RoomModel(shell: shell, objects: objects, removed: before?.removed,
                                                                         roomType: before?.roomType, finder: ObjectFinder.version))
                    for n in notes { debug.log("clean up: " + n) }
                    try? bundle.saveRoom(room)
                    let kept = room.objects ?? []
                    debug.log("saved room: \(kept.count) boxes, \(kept.filter { $0.label != nil }.count) named, "
                              + "\(room.removed?.count ?? 0) removed")
                }
                step("room", next: nil)
                debug.log(String(format: "saved %@: %d keyframes, %d tris in %.1f s", bundle.meta.id,
                                 bundle.meta.keyframeCount, bundle.meta.triangleCount, Date().timeIntervalSince(t0)))
                DispatchQueue.main.async {
                    self.endSave()
                    self.reset()
                    self.savedScan = bundle
                }
            } catch {
                // The recording has been handed over, so this scan can't continue recording photos.
                // A continued scan's folder is the saved scan itself and stays.
                if existing == nil, let d = rec?.dir { try? FileManager.default.removeItem(at: d) }
                debug.log("save FAILED: \(error)")
                DispatchQueue.main.async {
                    self.endSave()
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
        // Poses are unreliable while tracking is limited (and wrong until a continued scan lines up).
        guard case .normal = frame.camera.trackingState else { return }
        let T = frame.camera.transform
        let pos = SIMD3(T.columns.3.x, T.columns.3.y, T.columns.3.z)
        let fwd = SIMD3(T.columns.2.x, T.columns.2.y, T.columns.2.z)

        // Every depth frame is fused (the GPU keeps up); only skip while the last one is still running.
        lock.lock()
        if busy {
            lock.unlock()
            debug.update { $0.skippedBusy += 1 }
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
        DispatchQueue.main.async {
            if normal, self.isLiningUp {
                self.isLiningUp = false
                self.debug.log("lined up with the saved scan")
            }
            self.trackingText = self.isLiningUp ? "Point at the scanned room to line up…" : shown
        }
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
        let fuseMs = Date().timeIntervalSince(tPass) * 1000, gpuMs = vol.lastGPUms

        // Detail readout from the depth at the screen center.
        let cd = depthPtr[(dh / 2) * df.depthStride + dw / 2]
        let detail = (cd > TSDFVolume.minDepth && cd < TSDFVolume.maxDepth)
            ? String(format: "%.0f mm", TSDFVolume.voxelSize(TSDFVolume.level(forDepth: cd)) * 1000) : "–"

        let now = frame.timestamp
        let blocksNow = vol.blockCount
        debug.update { s in
            s.passes += 1; s.passMs = fuseMs; s.gpuMs = gpuMs; s.blocks = blocksNow
            s.passMsAvg = s.passMsAvg == 0 ? fuseMs : s.passMsAvg * 0.9 + fuseMs * 0.1
            if now - s.fuseSecondStart >= 1 { s.fuseRate = s.fusedThisSecond; s.fusedThisSecond = 0; s.fuseSecondStart = now }
            s.fusedThisSecond += 1
        }

        // Re-mesh the nearest dirty blocks about 15 times a second, so fusing keeps the frame rate.
        guard now - lastMeshTime >= 1.0 / 15 else { return }
        lastMeshTime = now
        let tMesh = Date()
        let dirty = vol.takeDirty(budget: 300, camera: cameraPos)
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
            blockSamples[b] = m.isEmpty ? nil : RoomFitter.sample([m], cell: 0.03)
        }
        if !dirty.isEmpty { samplesVersion += 1 }
        let tris = totalTris
        let meshMs = Date().timeIntervalSince(tMesh) * 1000
        debug.update { $0.meshMs = meshMs }

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
