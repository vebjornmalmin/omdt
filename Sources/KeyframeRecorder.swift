import ARKit
import CoreImage

/// Stores keyframes (photo, depth, pose) into a recording folder while scanning.
/// `consider` runs on the AR session queue and only does pose math; encoding and file
/// writes happen one frame at a time on a background queue, so ARKit never waits on disk.
final class KeyframeRecorder {
    static let imageWidth = 1024

    /// Called on the main thread with the number of stored keyframes.
    var onCount: ((Int) -> Void)?

    private let queue = DispatchQueue(label: "scan.keyframes", qos: .utility)
    private let ciContext = CIContext(options: [.cacheIntermediates: false])
    private let srgb = CGColorSpace(name: CGColorSpace.sRGB)!

    // Guarded by lock.
    private let lock = NSLock()
    private var selector = KeyframeSelector()
    private var dir: URL?
    private var busy = false
    private var generation = 0
    private var frames: [Keyframe] = []
    /// Index of the first keyframe of this recording; above 0 when continuing a saved scan.
    private var startIndex = 0
    /// False when recording into a saved scan's folder, which must survive a cancel.
    private var ownsDir = true

    var recordingDirectory: URL? { lock.lock(); defer { lock.unlock() }; return dir }

    /// Starts a new recording in a fresh folder, dropping any previous one.
    func begin() throws {
        let newDir = try ScanBundle.makeRecordingDirectory()
        lock.lock()
        let old = dir, oldOwned = ownsDir, oldStart = startIndex
        dir = newDir; generation += 1; selector.reset(); frames.removeAll(); busy = false
        startIndex = 0; ownsDir = true
        lock.unlock()
        if let old { discard(old, owned: oldOwned, from: oldStart) }
        report(0)
    }

    /// Records more keyframes into a saved scan, numbered after its existing ones.
    func begin(continuing scan: ScanBundle) throws {
        let existing = try scan.keyframes()
        lock.lock()
        let old = dir, oldOwned = ownsDir, oldStart = startIndex
        dir = scan.url; generation += 1; selector.reset(); busy = false
        frames = existing
        startIndex = (existing.map(\.id).max() ?? -1) + 1
        ownsDir = false
        lock.unlock()
        if let old { discard(old, owned: oldOwned, from: oldStart) }
        report(existing.count)
    }

    /// Stops recording and deletes what was recorded.
    func cancel() {
        lock.lock()
        let old = dir, oldOwned = ownsDir, oldStart = startIndex
        dir = nil; generation += 1; selector.reset(); frames.removeAll(); busy = false
        startIndex = 0; ownsDir = true
        lock.unlock()
        if let old { discard(old, owned: oldOwned, from: oldStart) }
        report(0)
    }

    /// Deletes a recording folder, or only this session's new files in a saved scan's folder.
    private func discard(_ url: URL, owned: Bool, from start: Int) {
        if owned { removeLater(url); return }
        queue.async {
            let fm = FileManager.default
            for sub in ["frames", "depth"] {
                let d = url.appendingPathComponent(sub)
                for f in (try? fm.contentsOfDirectory(atPath: d.path)) ?? [] {
                    if let n = Int(f.prefix { $0.isNumber }), n >= start { try? fm.removeItem(at: d.appendingPathComponent(f)) }
                }
            }
        }
    }

    /// Waits for pending writes, then hands over the recording. Recording stops.
    func finish() -> (dir: URL, keyframes: [Keyframe])? {
        queue.sync {}
        lock.lock(); defer { lock.unlock() }
        guard let d = dir else { return nil }
        let result = (d, frames.sorted { $0.id < $1.id })
        dir = nil; generation += 1; selector.reset(); frames.removeAll(); startIndex = 0; ownsDir = true
        return result
    }

    // MARK: session queue

    func consider(_ frame: ARFrame) {
        guard case .normal = frame.camera.trackingState, frame.sceneDepth != nil else {
            Debug.shared.update { $0.keyframeReason = "waiting for tracking" }
            return
        }
        let t = frame.camera.transform

        lock.lock()
        let canTake = dir != nil && !busy
        guard selector.consider(t, time: frame.timestamp, canTake: canTake), let d = dir else {
            let why = dir == nil ? "not recording" : busy ? "writing previous" : selector.lastReason
            lock.unlock()
            Debug.shared.update { $0.keyframeReason = why }
            return
        }
        let index = startIndex + selector.count
        selector.didCapture(t, time: frame.timestamp)
        busy = true
        let gen = generation
        lock.unlock()

        queue.async { [self] in
            let t0 = Date()
            var kf: Keyframe?
            do { kf = try write(frame, index: index, to: d) } catch { Debug.shared.log("keyframe \(index) FAILED: \(error)") }
            let ms = Date().timeIntervalSince(t0) * 1000
            Debug.shared.update { $0.keyframes += 1; $0.keyframeMs = ms; $0.keyframeReason = "saved #\(index)" }
            lock.lock()
            let current = gen == generation
            if current { busy = false; if let kf { frames.append(kf) } }
            let n = frames.count
            lock.unlock()
            if current { report(n) }
        }
    }

    // MARK: writing (keyframe queue)

    private func write(_ frame: ARFrame, index i: Int, to dir: URL) throws -> Keyframe? {
        guard let sd = frame.sceneDepth else { return nil }
        let img = frame.capturedImage
        let iw = CVPixelBufferGetWidth(img), ih = CVPixelBufferGetHeight(img)
        let s = Double(Self.imageWidth) / Double(iw)
        let ow = Self.imageWidth, oh = Int((Double(ih) * s).rounded())

        // Photo.
        let scaled = CIImage(cvPixelBuffer: img)
            .applyingFilter("CILanczosScaleTransform", parameters: [kCIInputScaleKey: s, kCIInputAspectRatioKey: 1])
            .cropped(to: CGRect(x: 0, y: 0, width: ow, height: oh))
        let q = CIImageRepresentationOption(rawValue: kCGImageDestinationLossyCompressionQuality as String)
        guard let jpeg = ciContext.jpegRepresentation(of: scaled, colorSpace: srgb, options: [q: 0.85]) else { return nil }
        try jpeg.write(to: dir.appendingPathComponent(ScanBundle.imagePath(i)))

        // Depth and confidence.
        let dm = sd.depthMap
        CVPixelBufferLockBaseAddress(dm, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(dm, .readOnly) }
        let dw = CVPixelBufferGetWidth(dm), dh = CVPixelBufferGetHeight(dm)
        guard let dBase = CVPixelBufferGetBaseAddress(dm) else { return nil }
        try ScanBundle.writeDepth(UnsafePointer(dBase.assumingMemoryBound(to: Float32.self)), width: dw, height: dh,
                                  stride: CVPixelBufferGetBytesPerRow(dm) / MemoryLayout<Float32>.size,
                                  to: dir.appendingPathComponent(ScanBundle.depthPath(i)))

        var confPath: String?
        if let cm = sd.confidenceMap {
            CVPixelBufferLockBaseAddress(cm, .readOnly)
            defer { CVPixelBufferUnlockBaseAddress(cm, .readOnly) }
            if let cBase = CVPixelBufferGetBaseAddress(cm) {
                let stride = CVPixelBufferGetBytesPerRow(cm)
                let src = cBase.assumingMemoryBound(to: UInt8.self)
                var out = Data(count: dw * dh)
                out.withUnsafeMutableBytes { o in
                    for y in 0..<dh { for x in 0..<dw { o[y * dw + x] = src[y * stride + x] } }
                }
                try out.write(to: dir.appendingPathComponent(ScanBundle.confidencePath(i)))
                confPath = ScanBundle.confidencePath(i)
            }
        }

        let K = frame.camera.intrinsics
        let fs = Float(s)
        return Keyframe(id: i, timestamp: frame.timestamp,
                        image: ScanBundle.imagePath(i), depth: ScanBundle.depthPath(i), confidence: confPath,
                        width: ow, height: oh, depthWidth: dw, depthHeight: dh,
                        intrinsics: [K[0][0] * fs, K[1][1] * fs, K[2][0] * fs, K[2][1] * fs],
                        camToWorld: Keyframe.flatten(frame.camera.transform),
                        sharpness: Self.sharpness(img))
    }

    /// Variance of a 4-neighbour Laplacian over the luma plane, sampled every 4th pixel.
    private static func sharpness(_ img: CVPixelBuffer) -> Float {
        CVPixelBufferLockBaseAddress(img, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(img, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddressOfPlane(img, 0) else { return 0 }
        let y = base.assumingMemoryBound(to: UInt8.self)
        let w = CVPixelBufferGetWidthOfPlane(img, 0), h = CVPixelBufferGetHeightOfPlane(img, 0)
        let st = CVPixelBufferGetBytesPerRowOfPlane(img, 0)
        var sum: Double = 0, sumSq: Double = 0, n: Double = 0
        for r in stride(from: 1, to: h - 1, by: 4) {
            for c in stride(from: 1, to: w - 1, by: 4) {
                let o = r * st + c
                let lap = 4 * Int(y[o]) - Int(y[o - 1]) - Int(y[o + 1]) - Int(y[o - st]) - Int(y[o + st])
                let v = Double(lap)
                sum += v; sumSq += v * v; n += 1
            }
        }
        guard n > 0 else { return 0 }
        let mean = sum / n
        return Float(sumSq / n - mean * mean)
    }

    private func report(_ n: Int) {
        DispatchQueue.main.async { [weak self] in self?.onCount?(n) }
    }

    private func removeLater(_ url: URL) {
        queue.async { try? FileManager.default.removeItem(at: url) }
    }
}
