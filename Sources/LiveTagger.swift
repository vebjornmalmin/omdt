import ARKit
import CoreImage
import FoundationModels
import Vision

/// Names boxes from the live camera. Each look crops one box from the current frame and runs
/// Vision's image classifier (fast); votes pile up over views. The best crop of each box goes
/// to the on-device language model one at a time (slow), and its answer wins once it's in.
/// A box is looked at again when a better view of it comes up, so labels keep improving.
/// Main thread only; its async work hops back to the main actor.
final class LiveTagger {
    struct State {
        var votes: [String: Float] = [:]
        var views = 0
        var lastLook = Date.distantPast
        var bestScore: Float = 0
        var bestCrop: CGImage?
        var modelName: String?
        var modelKind: String?
        var modelConfidence: Float = 0
        /// View score of the crop the model saw last.
        var modelScore: Float = 0
        var modelAt = Date.distantPast
    }

    /// Called when a label changes.
    var onChange: (() -> Void)?

    private(set) var states: [Int: State] = [:]
    private var visionBusy = false
    private var modelBusy = false
    private var modelBackoff = Date.distantPast
    /// Last known availability of the on-device model; rechecked every few seconds because it
    /// can be "not ready" for a while after launch or while it downloads.
    private var modelAvailable = false
    private var modelCheckedAt = Date.distantPast
    private var modelReason = ""
    /// How often each classifier label has come up over all looks. Broad parent labels
    /// ("portal" for every door and window) come up more often than the specific ones.
    private var labelSeen: [String: Int] = [:]
    private let ci = CIContext(options: [.cacheIntermediates: false])
    private let debug = Debug.shared

    init() { checkModel() }

    /// Updates `modelAvailable`, logging whenever it changes.
    private func checkModel() {
        modelCheckedAt = Date()
        let reason: String
        switch SystemLanguageModel.default.availability {
        case .available: reason = ""
        case .unavailable(let r): reason = "\(r)"
        }
        let available = reason.isEmpty
        if available != modelAvailable || reason != modelReason {
            let locale = Locale.current.identifier
            let supported = SystemLanguageModel.default.supportsLocale()
            debug.log((available ? "AI model available" : "AI model unavailable: \(reason)")
                      + ", language \(locale) \(supported ? "supported" : "NOT supported")")
        }
        if !modelBusy { setModelStatus(available ? "ready" : "off (\(reason))") }
        modelAvailable = available; modelReason = reason
    }

    func reset() { states.removeAll(); onChange?() }

    func forget(except ids: Set<Int>) {
        for id in states.keys where !ids.contains(id) { states[id] = nil }
    }

    /// Writes the current labels into the boxes.
    func apply(to boxes: [ObjectBox]) -> [ObjectBox] {
        boxes.map { b in
            var b = b
            if b.source == "user" { return b }          // a name given by hand wins
            if let l = label(b.id) {
                b.label = l.name; b.confidence = l.confidence; b.source = l.source
                b.guesses = topVotes(b.id, 3)
            }
            if let k = states[b.id]?.modelKind { b.kind = k }
            return b
        }
    }

    func label(_ id: Int) -> (name: String, confidence: Float, source: String)? {
        guard let s = states[id] else { return nil }
        if let m = s.modelName, s.modelConfidence >= 0.3 { return (m, s.modelConfidence, "model") }
        let total = s.votes.values.reduce(0, +)
        guard let top = s.votes.max(by: { $0.value < $1.value }), total > 0 else { return nil }
        // Parent and child labels often score the same ("conveyance", "portal", "window"); among
        // near-ties take the one seen least often overall, which is the most specific.
        let pick = s.votes.filter { $0.value >= top.value * 0.9 }
            .min { (labelSeen[$0.key] ?? 0, -$0.value) < (labelSeen[$1.key] ?? 0, -$1.value) }!
        return (Self.pretty(pick.key), pick.value / total, "vision")
    }

    private func topVotes(_ id: Int, _ n: Int) -> [String] {
        (states[id]?.votes ?? [:]).sorted { $0.value > $1.value }.prefix(n)
            .map { String(format: "%@ %.2f", Self.pretty($0.key), $0.value) }
    }

    var summary: String {
        let named = states.values.filter { !$0.votes.isEmpty || $0.modelName != nil }.count
        let byModel = states.values.filter { $0.modelName != nil }.count
        let views = states.values.reduce(0) { $0 + $1.views }
        return "\(named) named (\(byModel) by AI), \(views) looks"
    }

    // MARK: looking

    /// Picks the box with the best fresh view in this frame and classifies it.
    func look(frame: ARFrame, boxes: [ObjectBox]) {
        guard !visionBusy, case .normal = frame.camera.trackingState else { return }
        let img = frame.capturedImage
        let W = CGFloat(CVPixelBufferGetWidth(img)), H = CGFloat(CVPixelBufferGetHeight(img))
        let toCam = frame.camera.transform.inverse
        let K = frame.camera.intrinsics
        let now = Date()

        var best: (box: ObjectBox, rect: CGRect, score: Float, priority: Float)?
        for b in boxes {
            let corners = b.corners
            let cam = corners.map { xformPoint(toCam, $0) }
            guard cam.allSatisfy({ $0.z < -0.15 }) else { continue }
            var lo = CGPoint(x: CGFloat.infinity, y: .infinity), hi = CGPoint(x: -CGFloat.infinity, y: -.infinity)
            for c in cam {
                // Pinhole projection into the sensor image (camera looks down -z, image y down).
                let p = CGPoint(x: CGFloat(K[0][0] * c.x / -c.z + K[2][0]), y: CGFloat(K[1][1] * -c.y / -c.z + K[2][1]))
                lo = CGPoint(x: min(lo.x, p.x), y: min(lo.y, p.y)); hi = CGPoint(x: max(hi.x, p.x), y: max(hi.y, p.y))
            }
            let full = CGRect(x: lo.x, y: lo.y, width: hi.x - lo.x, height: hi.y - lo.y)
            let clipped = full.intersection(CGRect(x: 0, y: 0, width: W, height: H))
            guard !clipped.isNull, min(clipped.width, clipped.height) >= 90 else { continue }
            let visible = Float(clipped.width * clipped.height / max(full.width * full.height, 1))
            // Mostly in view, or too big to ever fit (a row of desks, a bed up close) but filling
            // a good part of the picture: a partial view still shows what it is.
            let fills = Float(clipped.width * clipped.height / (W * H))
            guard visible >= 0.55 || fills >= 0.3,
                  !occluded(b, rect: clipped, frame: frame, toCam: toCam, W: W) else { continue }
            let score = max(visible, min(fills, 0.5)) * min(1, Float(clipped.width * clipped.height / (0.12 * W * H)))
            let st = states[b.id] ?? State()
            guard now.timeIntervalSince(st.lastLook) > 1.2 else { continue }
            var priority = score
            if st.views < 3 { priority *= 2 }
            if score > st.bestScore * 1.15 { priority *= 1.5 }
            if priority > (best?.priority ?? 0.15) { best = (b, clipped, score, priority) }
        }
        guard let pick = best, let crop = crop(img, pick.rect, W: W, H: H) else { return }

        let id = pick.box.id, score = pick.score
        var st = states[id] ?? State()
        st.lastLook = now
        states[id] = st
        visionBusy = true
        Task { @MainActor [self] in
            defer { visionBusy = false }
            do {
                let results = try await ClassifyImageRequest().perform(on: crop, orientation: .right)
                record(id, results, crop: crop, score: score)
            } catch {
                debug.log("classify failed: \(error)")
            }
            considerModel(boxes: boxes)
        }
    }

    /// True when the depth at the middle of the box's image is well in front of the box.
    private func occluded(_ b: ObjectBox, rect: CGRect, frame: ARFrame, toCam: simd_float4x4, W: CGFloat) -> Bool {
        guard let dm = frame.sceneDepth?.depthMap else { return false }
        CVPixelBufferLockBaseAddress(dm, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(dm, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(dm) else { return false }
        let dw = CVPixelBufferGetWidth(dm), dh = CVPixelBufferGetHeight(dm)
        let s = CGFloat(dw) / W
        let x = min(dw - 1, max(0, Int(rect.midX * s))), y = min(dh - 1, max(0, Int(rect.midY * s)))
        let d = base.advanced(by: y * CVPixelBufferGetBytesPerRow(dm)).assumingMemoryBound(to: Float32.self)[x]
        let dist = -xformPoint(toCam, b.centerVec).z
        let half = simd_length(SIMD3(b.size[0], b.size[1], b.size[2])) / 2
        return d > 0.05 && d < dist - half - 0.2
    }

    /// The box region, padded, at most 640 px on the long side, in sensor orientation.
    private func crop(_ img: CVPixelBuffer, _ r: CGRect, W: CGFloat, H: CGFloat) -> CGImage? {
        let pad = r.insetBy(dx: -r.width * 0.1, dy: -r.height * 0.1).intersection(CGRect(x: 0, y: 0, width: W, height: H))
        let ciRect = CGRect(x: pad.minX, y: H - pad.maxY, width: pad.width, height: pad.height)   // CI is bottom-up
        let scale = min(1, 640 / max(pad.width, pad.height))
        let image = CIImage(cvPixelBuffer: img).cropped(to: ciRect)
            .transformed(by: CGAffineTransform(translationX: -ciRect.minX, y: -ciRect.minY))
            .transformed(by: CGAffineTransform(scaleX: scale, y: scale))
        return ci.createCGImage(image, from: image.extent.integral)
    }

    private func record(_ id: Int, _ results: [ClassificationObservation], crop: CGImage, score: Float) {
        guard var st = states[id] else { return }
        let before = label(id)?.name
        var added = 0
        for r in results.sorted(by: { $0.confidence > $1.confidence }) where r.confidence >= 0.08 {
            guard !Self.ignored.contains(r.identifier) else { continue }
            labelSeen[r.identifier, default: 0] += 1
            let weight: Float = Self.generic.contains(r.identifier) ? 0.4 : 1
            st.votes[r.identifier, default: 0] += r.confidence * weight * score
            added += 1
            if added == 5 { break }
        }
        st.views += 1
        if score > st.bestScore { st.bestScore = score; st.bestCrop = crop }
        states[id] = st
        if label(id)?.name != before { onChange?() }
    }

    // MARK: language model

    /// Sends one box's best crop to the language model when it has none yet, or when a much
    /// better view has come in since it last looked.
    private func considerModel(boxes: [ObjectBox]) {
        if Date().timeIntervalSince(modelCheckedAt) > 5 { checkModel() }
        guard modelAvailable, !modelBusy, Date() > modelBackoff,
              ProcessInfo.processInfo.thermalState.rawValue < ProcessInfo.ThermalState.serious.rawValue else { return }
        let now = Date()
        let candidates = boxes.compactMap { b -> (ObjectBox, State)? in
            guard let s = states[b.id], s.bestCrop != nil, s.views >= 2 else { return nil }
            let due = s.modelName == nil
                || (s.bestScore > s.modelScore * 1.3 && now.timeIntervalSince(s.modelAt) > 8)
            return due ? (b, s) : nil
        }
        guard let (box, st) = candidates.max(by: { $0.1.bestScore < $1.1.bestScore }), let crop = st.bestCrop else { return }
        modelBusy = true
        setModelStatus("naming box \(box.id)…")
        debug.log("AI naming box \(box.id) (\(st.views) looks, view \(String(format: "%.2f", st.bestScore)))")
        let guesses = topVotes(box.id, 3).joined(separator: ", ")
        let size = BoxNamer.sizeText(box)
        let id = box.id, score = st.bestScore
        Task { @MainActor [self] in
            defer { modelBusy = false }
            let t0 = Date()
            do {
                let content = try await BoxNamer.name(crops: [crop], size: size, guesses: guesses)
                let name = content.name.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
                guard var s = states[id] else { return }
                let before = label(id)?.name
                s.modelName = name.isEmpty ? nil : name
                s.modelKind = content.kind.rawValue
                s.modelConfidence = Float(min(1, max(0, content.confidence)))
                s.modelScore = score; s.modelAt = Date()
                states[id] = s
                let ms = Int(Date().timeIntervalSince(t0) * 1000)
                debug.log(String(format: "AI box %d: %@, %@ (%.2f) in %d ms; vision said %@", id, name,
                                 content.kind.rawValue, s.modelConfidence, ms, guesses))
                setModelStatus("last: \(name) in \(ms) ms")
                if label(id)?.name != before { onChange?() }
            } catch {
                debug.log("AI naming failed: \(error)")
                setModelStatus("error, retry in 15 s")
                modelBackoff = Date().addingTimeInterval(15)
            }
        }
    }

    private func setModelStatus(_ s: String) { debug.update { $0.model = s } }

    // MARK: vocabulary

    /// Classifier labels that describe the room or a material rather than a thing.
    static let ignored: Set<String> = [
        "structure", "interior_room", "indoor", "room", "wall", "floor", "ceiling", "wood_processed", "wood_natural",
        "material", "textile", "housewares", "consumer_electronics", "liquid", "outdoor", "sky", "people", "adult",
        "baseboard", "hardwood", "tile", "carpet", "rug", "shadow", "light", "lighting", "plank", "stone",
        "haze", "fog", "illustrations", "brick", "blue_sky", "night_sky", "sunset_sunrise", "cloudy", "texture",
    ]
    /// Broad parent labels that count for less than a specific one.
    static let generic: Set<String> = [
        "furniture", "seat", "container", "appliance", "electronics", "decoration", "art", "utensil", "tableware",
        "machine", "equipment", "object", "clothing", "office_supplies", "tool", "storage",
        "conveyance", "portal", "vehicle", "aircraft", "cord", "foliage", "art",
    ]

    static func pretty(_ id: String) -> String { id.replacingOccurrences(of: "_", with: " ") }
}
