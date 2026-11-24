import Foundation
import simd

// A saved scan on disk:
//
//   Scans/<id>/
//     scan.json        ScanMeta
//     mesh.ply         fused mesh (binary PLY, see MeshExporter)
//     keyframes.json   [Keyframe]
//     frames/0000.jpg  photo, sensor (landscape) orientation
//     depth/0000.f16   LiDAR depth in meters, Float16, depthWidth x depthHeight
//     depth/0000.conf  ARKit confidence (0-2), UInt8
//
// While recording, the same layout is filled in Scans/.recording-<uuid>/ and moved into
// place on save, so an interrupted recording never shows up as a scan.

/// One stored camera view. Image, depth and intrinsics are all in the sensor's landscape
/// orientation; rotate 90° clockwise for portrait display.
struct Keyframe: Codable, Identifiable, Equatable {
    var id: Int
    var timestamp: Double
    var image: String
    var depth: String
    var confidence: String?
    var width: Int
    var height: Int
    var depthWidth: Int
    var depthHeight: Int
    /// fx, fy, cx, cy in pixels at `width` x `height`.
    var intrinsics: [Float]
    /// Camera-to-world, column-major. ARKit camera axes: x right, y up, -z forward.
    var camToWorld: [Float]
    /// Variance of the Laplacian of the luma image; higher is sharper.
    var sharpness: Float

    var transform: simd_float4x4 {
        let m = camToWorld
        return simd_float4x4(columns: (SIMD4(m[0], m[1], m[2], m[3]), SIMD4(m[4], m[5], m[6], m[7]),
                                       SIMD4(m[8], m[9], m[10], m[11]), SIMD4(m[12], m[13], m[14], m[15])))
    }

    static func flatten(_ t: simd_float4x4) -> [Float] {
        [t.columns.0, t.columns.1, t.columns.2, t.columns.3].flatMap { [$0.x, $0.y, $0.z, $0.w] }
    }
}

struct ScanMeta: Codable, Equatable {
    var id: String
    var version: Int
    var created: Date
    var keyframeCount: Int
    var triangleCount: Int
    var boundsMin: [Float]
    var boundsMax: [Float]
    /// Keyframe used as the scan's thumbnail.
    var coverKeyframe: Int?
    /// Where the phone was when the scan was saved, to offer continuing it nearby.
    var location: GeoPoint?
    /// Last time scanning was continued.
    var updated: Date?
}

struct GeoPoint: Codable, Equatable {
    var latitude: Double
    var longitude: Double
    /// Horizontal accuracy in meters.
    var accuracy: Double

    /// Great-circle distance in meters.
    func distance(to o: GeoPoint) -> Double {
        let r = 6_371_000.0, rad = Double.pi / 180
        let dLat = (o.latitude - latitude) * rad, dLon = (o.longitude - longitude) * rad
        let a = sin(dLat / 2) * sin(dLat / 2) + cos(latitude * rad) * cos(o.latitude * rad) * sin(dLon / 2) * sin(dLon / 2)
        return 2 * r * asin(min(1, a.squareRoot()))
    }
}

struct ScanBundle: Identifiable, Hashable {
    let url: URL
    let meta: ScanMeta

    var id: String { url.path }
    var meshURL: URL { url.appendingPathComponent(Self.meshFile) }

    static func == (a: ScanBundle, b: ScanBundle) -> Bool { a.url == b.url }
    func hash(into h: inout Hasher) { h.combine(url) }

    static let metaFile = "scan.json"
    static let keyframesFile = "keyframes.json"
    static let meshFile = "mesh.ply"
    /// ARKit world map, to line up with the room again when continuing.
    static let worldMapFile = "worldmap.arexperience"
    /// Fusion volume, so continuing adds to the same surfaces.
    static let volumeFile = "volume.tsdf"
    static let recordingPrefix = ".recording-"

    /// Where scans live. Tests point this at a temporary folder.
    static var root: URL = {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        return docs.appendingPathComponent("Scans", isDirectory: true)
    }()

    // MARK: reading

    static func load(_ url: URL) throws -> ScanBundle {
        let data = try Data(contentsOf: url.appendingPathComponent(metaFile))
        return ScanBundle(url: url, meta: try decoder.decode(ScanMeta.self, from: data))
    }

    /// All saved scans, newest first. Unreadable folders are skipped.
    static func list() -> [ScanBundle] {
        let fm = FileManager.default
        guard let items = try? fm.contentsOfDirectory(at: root, includingPropertiesForKeys: nil) else { return [] }
        return items
            .filter { !$0.lastPathComponent.hasPrefix(".") }
            .compactMap { try? load($0) }
            .sorted { $0.meta.created > $1.meta.created }
    }

    func keyframes() throws -> [Keyframe] {
        let data = try Data(contentsOf: url.appendingPathComponent(Self.keyframesFile))
        return try Self.decoder.decode([Keyframe].self, from: data)
    }

    func imageURL(_ k: Keyframe) -> URL { url.appendingPathComponent(k.image) }

    func depth(_ k: Keyframe) throws -> [Float] {
        try Self.readDepth(url.appendingPathComponent(k.depth), count: k.depthWidth * k.depthHeight)
    }

    var coverImageURL: URL? {
        guard let i = meta.coverKeyframe else { return nil }
        return url.appendingPathComponent(Self.imagePath(i))
    }

    func delete() throws { try FileManager.default.removeItem(at: url) }

    var worldMapURL: URL { url.appendingPathComponent(Self.worldMapFile) }
    var volumeURL: URL { url.appendingPathComponent(Self.volumeFile) }

    /// Saved with what's needed to pick up scanning later.
    var canContinue: Bool {
        FileManager.default.fileExists(atPath: worldMapURL.path) && FileManager.default.fileExists(atPath: volumeURL.path)
    }

    // MARK: recording

    static func imagePath(_ i: Int) -> String { String(format: "frames/%04d.jpg", i) }
    static func depthPath(_ i: Int) -> String { String(format: "depth/%04d.f16", i) }
    static func confidencePath(_ i: Int) -> String { String(format: "depth/%04d.conf", i) }

    static func makeRecordingDirectory() throws -> URL {
        let dir = root.appendingPathComponent(recordingPrefix + UUID().uuidString, isDirectory: true)
        for sub in ["frames", "depth"] {
            try FileManager.default.createDirectory(at: dir.appendingPathComponent(sub, isDirectory: true),
                                                    withIntermediateDirectories: true)
        }
        return dir
    }

    /// Removes recordings left behind by a crash or a kill mid-scan.
    static func removeStaleRecordings() {
        let fm = FileManager.default
        guard let items = try? fm.contentsOfDirectory(at: root, includingPropertiesForKeys: nil) else { return }
        for u in items where u.lastPathComponent.hasPrefix(recordingPrefix) { try? fm.removeItem(at: u) }
    }

    /// Writes mesh and metadata into a recording folder and moves it into place as a scan.
    /// With `existing`, the recording folder is that scan's own folder and it is updated in place.
    static func finalize(recording dir: URL, meshes: [MeshOut], keyframes: [Keyframe],
                         created: Date = Date(), location: GeoPoint? = nil,
                         existing: ScanBundle? = nil) throws -> ScanBundle {
        let ply = try MeshExporter.plyData(meshes)
        try ply.write(to: dir.appendingPathComponent(meshFile), options: .atomic)
        try encoder.encode(keyframes).write(to: dir.appendingPathComponent(keyframesFile), options: .atomic)

        var lo = SIMD3<Float>(repeating: .infinity), hi = SIMD3<Float>(repeating: -.infinity)
        var tris = 0
        for m in meshes {
            tris += m.idx.count / 3
            for p in m.pos { lo = simd_min(lo, p); hi = simd_max(hi, p) }
        }
        let cover = keyframes.isEmpty ? nil : keyframes[keyframes.count / 2].id

        if let old = existing {
            var meta = old.meta
            meta.keyframeCount = keyframes.count; meta.triangleCount = tris
            meta.boundsMin = [lo.x, lo.y, lo.z]; meta.boundsMax = [hi.x, hi.y, hi.z]
            meta.coverKeyframe = meta.coverKeyframe ?? cover
            meta.location = location ?? meta.location
            meta.updated = created
            try encoder.encode(meta).write(to: dir.appendingPathComponent(metaFile), options: .atomic)
            return ScanBundle(url: old.url, meta: meta)
        }

        let fmt = DateFormatter()
        fmt.locale = Locale(identifier: "en_US_POSIX")
        fmt.dateFormat = "yyyyMMdd-HHmmss"
        var id = fmt.string(from: created)
        var dest = root.appendingPathComponent(id, isDirectory: true)
        var n = 2
        while FileManager.default.fileExists(atPath: dest.path) {
            id = fmt.string(from: created) + "-\(n)"; n += 1
            dest = root.appendingPathComponent(id, isDirectory: true)
        }

        let meta = ScanMeta(id: id, version: 1, created: created, keyframeCount: keyframes.count,
                            triangleCount: tris, boundsMin: [lo.x, lo.y, lo.z], boundsMax: [hi.x, hi.y, hi.z],
                            coverKeyframe: cover, location: location)
        try encoder.encode(meta).write(to: dir.appendingPathComponent(metaFile), options: .atomic)
        try FileManager.default.moveItem(at: dir, to: dest)
        return ScanBundle(url: dest, meta: meta)
    }

    // MARK: depth files

    static func writeDepth(_ src: UnsafePointer<Float32>, width: Int, height: Int, stride: Int, to url: URL) throws {
        var out = [Float16](repeating: 0, count: width * height)
        for y in 0..<height {
            for x in 0..<width { out[y * width + x] = Float16(src[y * stride + x]) }
        }
        try out.withUnsafeBytes { Data($0) }.write(to: url)
    }

    static func readDepth(_ url: URL, count: Int) throws -> [Float] {
        let data = try Data(contentsOf: url)
        guard data.count == count * MemoryLayout<Float16>.size else {
            throw NSError(domain: "ScanBundle", code: 2,
                          userInfo: [NSLocalizedDescriptionKey: "Depth file has the wrong size: \(url.lastPathComponent)"])
        }
        return data.withUnsafeBytes { raw in raw.bindMemory(to: Float16.self).map { Float($0) } }
    }

    // MARK: coding

    static let encoder: JSONEncoder = {
        let e = JSONEncoder()
        e.outputFormatting = [.prettyPrinted, .sortedKeys]
        e.dateEncodingStrategy = .iso8601
        return e
    }()

    static let decoder: JSONDecoder = {
        let d = JSONDecoder()
        d.dateDecodingStrategy = .iso8601
        return d
    }()
}

/// Decides which camera frames become keyframes: one every ~25 cm or ~15° of turning,
/// skipping frames taken while the phone moves fast enough to blur the photo.
struct KeyframeSelector {
    var minTranslation: Float = 0.25
    var minRotation: Float = 15 * .pi / 180
    var minInterval: Double = 0.3
    var maxAngularSpeed: Float = 1.0    // rad/s
    var maxLinearSpeed: Float = 0.6     // m/s
    var maxCount = 250

    private(set) var count = 0
    /// Why the last frame was not taken, for the debug panel.
    private(set) var lastReason = "–"
    private var last: (t: simd_float4x4, time: Double)?
    private var prev: (t: simd_float4x4, time: Double)?

    /// Feed every tracked frame. Returns true when this frame should be stored; the caller
    /// then stores it and calls `didCapture`. `canTake` false still updates motion tracking.
    mutating func consider(_ t: simd_float4x4, time: Double, canTake: Bool = true) -> Bool {
        defer { prev = (t, time) }
        if count >= maxCount { lastReason = "limit of \(maxCount) reached"; return false }
        guard canTake, let p = prev, time > p.time else { lastReason = "busy"; return false }

        let dt = Float(time - p.time)
        if Self.angle(p.t, t) / dt > maxAngularSpeed { lastReason = "turning too fast"; return false }
        if simd_distance(Self.position(p.t), Self.position(t)) / dt > maxLinearSpeed {
            lastReason = "moving too fast"; return false
        }

        guard let k = last else { return true }
        if time - k.time < minInterval { lastReason = "too soon"; return false }
        let moved = simd_distance(Self.position(k.t), Self.position(t)) >= minTranslation
            || Self.angle(k.t, t) >= minRotation
        if !moved { lastReason = "move or turn more" }
        return moved
    }

    mutating func didCapture(_ t: simd_float4x4, time: Double) {
        last = (t, time)
        count += 1
    }

    mutating func reset() { count = 0; last = nil; prev = nil }

    static func position(_ t: simd_float4x4) -> SIMD3<Float> {
        SIMD3(t.columns.3.x, t.columns.3.y, t.columns.3.z)
    }

    /// Rotation angle between two poses, in radians.
    static func angle(_ a: simd_float4x4, _ b: simd_float4x4) -> Float {
        func r(_ m: simd_float4x4) -> simd_float3x3 {
            simd_float3x3(SIMD3(m.columns.0.x, m.columns.0.y, m.columns.0.z),
                          SIMD3(m.columns.1.x, m.columns.1.y, m.columns.1.z),
                          SIMD3(m.columns.2.x, m.columns.2.y, m.columns.2.z))
        }
        let rel = r(a).transpose * r(b)
        let tr = rel.columns.0.x + rel.columns.1.y + rel.columns.2.z
        return acos(min(1, max(-1, (tr - 1) / 2)))
    }
}
