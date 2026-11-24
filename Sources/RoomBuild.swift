import Foundation
import FoundationModels

/// The automatic room build after a scan (see `ROOM_MODEL_PLAN.md`): the 3D objects right away,
/// then names from the best photos of every box, a clean-up, and new objects where names changed.
enum RoomBuild {
    /// Objects for every box that needs one, saved with their scan pieces. Fast enough to run
    /// while saving; call off the main thread.
    @discardableResult
    static func objects(_ room: RoomModel, meshes: [MeshOut], scan: ScanBundle, force: Bool = false) -> RoomModel {
        let t0 = Date()
        let built = RoomObjects.build(room, meshes: meshes, force: force)
        scan.savePieces(built.pieces)
        for n in built.notes { Debug.shared.log("build: " + n) }
        Debug.shared.log(String(format: "build: %d object(s) in %.1f s", built.notes.count, Date().timeIntervalSince(t0)))
        try? scan.saveRoom(built.room)
        return built.room
    }

    /// True when the AI step still has work: boxes not yet named from their photos.
    static func needsNaming(_ room: RoomModel) -> Bool {
        room.namedFromPhotos != true && !(room.objects ?? []).isEmpty
    }

    /// Names every box you didn't name from its best 2–3 photos, biggest first (at most 30), then
    /// cleans up and builds the objects whose names changed. `progress` gets a line per step.
    static func nameFromPhotos(_ room: RoomModel, meshes: [MeshOut], scan: ScanBundle,
                               progress: @escaping @Sendable (String) -> Void) async -> RoomModel {
        guard case .available = SystemLanguageModel.default.availability else {
            Debug.shared.log("build: naming skipped, the on-device model is not available")
            return room
        }
        let kfs = (try? scan.keyframes()) ?? []
        var r = room
        let boxes = Array((r.objects ?? []).filter { $0.source != "user" }.sorted { $0.volume > $1.volume }.prefix(30))
        let t0 = Date()
        var renamed = 0, answered = 0, failedInARow = 0, asked = 0
        for (i, b) in boxes.enumerated() {
            // The model keeps failing (not ready, or not on this device): try again another time.
            if failedInARow >= 3 { break }
            progress("Naming \(i + 1) of \(boxes.count)…")
            let views = BoxViews.best(b, keyframes: kfs) { try? scan.depth($0) }
            let crops = views.compactMap { BoxViews.crop($0, image: scan.imageURL($0.keyframe)) }
            guard !crops.isEmpty else { continue }
            asked += 1
            do {
                let a = try await BoxNamer.name(crops: crops, size: BoxNamer.sizeText(b))
                answered += 1; failedInARow = 0
                guard a.confidence >= 0.4, let name = RoomReview.cleanName(a.name),
                      let k = r.objects?.firstIndex(where: { $0.id == b.id }), r.objects?[k].source != "user" else { continue }
                if r.objects![k].label != name { renamed += 1 }
                r.objects![k].label = name
                r.objects![k].source = "model"
                r.objects![k].confidence = Float(min(1, max(0, a.confidence)))
                r.objects![k].kind = a.kind.rawValue
                Debug.shared.log(String(format: "build: box %d named %@ (%@, %.2f) from %d photo(s)", b.id, name,
                                        a.kind.rawValue, a.confidence, crops.count))
            } catch {
                failedInARow += 1
                Debug.shared.log("build: naming box \(b.id) failed: \(error.localizedDescription)")
            }
        }
        guard answered > 0 || asked == 0 else {
            Debug.shared.log("build: naming failed for every box tried; will try again next time")
            return room
        }
        Debug.shared.log(String(format: "build: named %d box(es), %d changed, in %.0f s", boxes.count, renamed,
                                Date().timeIntervalSince(t0)))
        r.namedFromPhotos = true
        progress("Cleaning up…")
        let (fixed, notes) = RoomCorrector.correct(r)
        for n in notes { Debug.shared.log("clean up: " + n) }
        progress("Building the objects…")
        return objects(fixed, meshes: meshes, scan: scan)
    }
}
