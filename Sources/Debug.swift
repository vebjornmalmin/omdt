import ARKit
import Foundation
import os

/// Development diagnostics: a log (on screen, in Documents/debug.log, and on stdout so
/// `devicectl device process launch --console` shows it) plus live counters for the panel.
final class Debug {
    static let shared = Debug()

    let launch = Date()
    let logURL: URL
    let previousLogURL: URL

    private let lock = NSLock()
    private var lines: [String] = []
    private let maxLines = 400
    private var file: FileHandle?
    private let oslog = Logger(subsystem: "com.vebjorn.LidarScan", category: "debug")

    private(set) var stats = Stats()

    struct Stats {
        var frames = 0
        var framesWithDepth = 0
        var framesThisSecond = 0
        var fps = 0
        var secondStart: TimeInterval = 0
        var lastFrameAt: Date?
        var imageSize = "–"
        var depthSize = "–"
        var tracking = "–"
        var trackingSince = Date()
        var mapping = "–"
        var passes = 0
        var passMs: Double = 0
        var passMsAvg: Double = 0
        var gpuMs: Double = 0
        var meshMs: Double = 0
        var fuseRate = 0
        var fusedThisSecond = 0
        var fuseSecondStart: TimeInterval = 0
        var skippedBusy = 0
        var blocks = 0
        var keyframes = 0
        var keyframeReason = "–"
        var keyframeMs: Double = 0
        var interruptions = 0
        var errors = 0
        var videoFormat = "–"
        /// Mean luma of the camera image (0-255); near 0 means covered lens or a dark room.
        var brightness = -1
        var exposure = "–"
        /// Live room fit, boxes and tagging.
        var room = "–"
        var objects = "–"
        var model = "–"
    }

    private init() {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        logURL = docs.appendingPathComponent("debug.log")
        previousLogURL = docs.appendingPathComponent("debug-previous.log")
        let fm = FileManager.default
        try? fm.removeItem(at: previousLogURL)
        try? fm.moveItem(at: logURL, to: previousLogURL)
        fm.createFile(atPath: logURL.path, contents: nil)
        file = try? FileHandle(forWritingTo: logURL)
        log(Self.environmentSummary())
    }

    // MARK: log

    func log(_ message: String) {
        let t = Date().timeIntervalSince(launch)
        let line = String(format: "%8.3f  ", t) + message
        lock.lock()
        lines.append(line)
        if lines.count > maxLines { lines.removeFirst(lines.count - maxLines) }
        file?.write(Data((line + "\n").utf8))
        lock.unlock()
        print(line)
        oslog.log("\(message, privacy: .public)")
    }

    func recentLines(_ n: Int) -> [String] {
        lock.lock(); defer { lock.unlock() }
        return Array(lines.suffix(n))
    }

    func allLines() -> String {
        lock.lock(); defer { lock.unlock() }
        return lines.joined(separator: "\n")
    }

    // MARK: stats

    func update(_ f: (inout Stats) -> Void) {
        lock.lock(); f(&stats); lock.unlock()
    }

    func snapshot() -> Stats {
        lock.lock(); defer { lock.unlock() }
        return stats
    }

    /// Called for every ARFrame on the session queue.
    func frame(_ frame: ARFrame) {
        let hasDepth = frame.sceneDepth != nil
        let sampleLuma = stats.frames % 30 == 0
        let luma = sampleLuma ? Self.meanLuma(frame.capturedImage) : -1
        let exposure = sampleLuma
            ? String(format: "%.0f ms, offset %.1f", frame.camera.exposureDuration * 1000, frame.camera.exposureOffset) : ""
        var first = false, firstDepth = false
        var imageSize = "", depthSize = ""
        defer {
            if first { log("first frame: image \(imageSize), depth \(hasDepth ? "yes" : "no"), tracking \(Self.describe(frame.camera.trackingState))") }
            if firstDepth { log("first depth frame: \(depthSize)") }
        }
        update { s in
            first = s.frames == 0
            firstDepth = hasDepth && s.framesWithDepth == 0
            s.frames += 1
            if hasDepth { s.framesWithDepth += 1 }
            s.lastFrameAt = Date()
            if frame.timestamp - s.secondStart >= 1 {
                s.fps = s.framesThisSecond; s.framesThisSecond = 0; s.secondStart = frame.timestamp
            }
            s.framesThisSecond += 1
            s.mapping = Self.describe(frame.worldMappingStatus)
            if first {
                s.imageSize = "\(CVPixelBufferGetWidth(frame.capturedImage))×\(CVPixelBufferGetHeight(frame.capturedImage))"
            }
            if firstDepth, let d = frame.sceneDepth?.depthMap {
                s.depthSize = "\(CVPixelBufferGetWidth(d))×\(CVPixelBufferGetHeight(d))"
            }
            if luma >= 0 { s.brightness = luma; s.exposure = exposure }
            imageSize = s.imageSize; depthSize = s.depthSize
        }
    }

    static func meanLuma(_ img: CVPixelBuffer) -> Int {
        CVPixelBufferLockBaseAddress(img, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(img, .readOnly) }
        guard CVPixelBufferGetPlaneCount(img) > 0, let base = CVPixelBufferGetBaseAddressOfPlane(img, 0) else { return -1 }
        let y = base.assumingMemoryBound(to: UInt8.self)
        let w = CVPixelBufferGetWidthOfPlane(img, 0), h = CVPixelBufferGetHeightOfPlane(img, 0)
        let st = CVPixelBufferGetBytesPerRowOfPlane(img, 0)
        var sum = 0, n = 0
        for r in stride(from: 0, to: h, by: h / 48) { for c in stride(from: 0, to: w, by: w / 64) { sum += Int(y[r * st + c]); n += 1 } }
        return n > 0 ? sum / n : -1
    }

    /// One line with everything that matters, for the console every few seconds.
    func summaryLine() -> String {
        let s = snapshot()
        return "fps \(s.fps) frames \(s.frames) depth \(s.framesWithDepth) light \(s.brightness) exp \(s.exposure) | \(s.tracking) | map \(s.mapping) | "
            + "fused \(s.fuseRate)/s (\(String(format: "%.1f", s.passMs)) ms, gpu \(String(format: "%.1f", s.gpuMs)), mesh \(Int(s.meshMs)) ms, skips \(s.skippedBusy)) blocks \(s.blocks) | "
            + "keyframes \(s.keyframes) (\(s.keyframeReason)) | mem \(Int(Self.memoryMB())) MB | \(Self.thermal())"
    }

    // MARK: descriptions

    static func describe(_ t: ARCamera.TrackingState) -> String {
        switch t {
        case .normal: return "normal"
        case .notAvailable: return "not available"
        case .limited(let r):
            switch r {
            case .initializing: return "limited: initializing"
            case .excessiveMotion: return "limited: excessive motion"
            case .insufficientFeatures: return "limited: insufficient features"
            case .relocalizing: return "limited: relocalizing"
            @unknown default: return "limited: unknown"
            }
        }
    }

    static func describe(_ m: ARFrame.WorldMappingStatus) -> String {
        switch m {
        case .notAvailable: return "none"
        case .limited: return "limited"
        case .extending: return "extending"
        case .mapped: return "mapped"
        @unknown default: return "?"
        }
    }

    static func thermal() -> String {
        switch ProcessInfo.processInfo.thermalState {
        case .nominal: return "cool"
        case .fair: return "warm"
        case .serious: return "HOT"
        case .critical: return "CRITICAL"
        @unknown default: return "?"
        }
    }

    static func memoryMB() -> Double {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        return kr == KERN_SUCCESS ? Double(info.phys_footprint) / 1_048_576 : 0
    }

    static var buildDate: String {
        guard let url = Bundle.main.executableURL,
              let d = (try? FileManager.default.attributesOfItem(atPath: url.path))?[.modificationDate] as? Date
        else { return "?" }
        let f = DateFormatter(); f.dateFormat = "dd.MM HH:mm"
        return f.string(from: d)
    }

    static func environmentSummary() -> String {
        let os = ProcessInfo.processInfo.operatingSystemVersionString
        let ver = Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "?"
        let auth = AVCaptureDevice.authorizationStatus(for: .video)
        let camera: String
        switch auth {
        case .authorized: camera = "authorized"
        case .denied: camera = "DENIED"
        case .restricted: camera = "restricted"
        case .notDetermined: camera = "not asked yet"
        @unknown default: camera = "?"
        }
        return "launch: app \(ver) built \(buildDate), iOS \(os), camera \(camera), "
            + "worldTracking \(ARWorldTrackingConfiguration.isSupported), "
            + "sceneDepth \(ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth))"
    }
}
