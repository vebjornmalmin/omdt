import CoreGraphics
import Foundation
import FoundationModels

/// What kind of thing a box holds. Wall, floor, ceiling, door and window are part of the room
/// itself and get taken out of the furniture list.
@Generable
enum ObjectKind: String, CaseIterable {
    case furniture, object, wall, floor, ceiling, door, window, unclear
}

/// What the on-device language model returns for one photo crop.
@Generable
struct ObjectName {
    @Guide(description: "Everyday name of the main object in the photo, 1 to 3 lowercase words, for example: sofa, office chair, floor lamp, bookshelf, bed, desk")
    var name: String
    @Guide(description: "furniture for things like beds, sofas, tables, chairs and cabinets; object for smaller things such as lamps, plants, screens and pictures; wall, floor, ceiling, door or window when the photo mainly shows that part of the room and no separate object; unclear when you cannot tell")
    var kind: ObjectKind
    @Guide(description: "How sure you are, from 0 to 1")
    var confidence: Double
}

/// Asks the on-device model what a box holds, from photo crops of it plus its size. Used live
/// while scanning (one crop) and from the Room screen (the best saved photos).
enum BoxNamer {
    static let instructions = """
        You name objects in photos taken while scanning a room. Look at the photo and give the \
        everyday name of the main object in the middle of it, and what kind of thing it is. \
        Sometimes the photo only shows a piece of wall, floor or ceiling, or a door or a window, \
        and no separate object; then say so in the kind. A picture, shelf or screen hanging on a \
        wall is an object, not a wall. Use the size and the classifier guesses as hints, but \
        trust the photo.
        """

    struct Unavailable: LocalizedError {
        var errorDescription: String? { "Apple Intelligence is not available right now." }
    }

    /// Size as the prompt gives it: long side, short side, height.
    static func sizeText(_ b: ObjectBox) -> String {
        String(format: "%.1f m wide, %.1f m deep, %.1f m tall", max(b.size[0], b.size[2]), min(b.size[0], b.size[2]), b.size[1])
    }

    /// One answer for the crops (sensor orientation; at most two are sent).
    static func name(crops: [CGImage], size: String, guesses: String = "") async throws -> ObjectName {
        guard case .available = SystemLanguageModel.default.availability else { throw Unavailable() }
        let session = LanguageModelSession(instructions: instructions)
        let shown = Array(crops.prefix(2))
        let r = try await session.respond(generating: ObjectName.self) {
            "The object is about \(size). An image classifier guessed: \(guesses.isEmpty ? "nothing" : guesses)."
            if shown.count > 1 { "The photos show the same object from different sides." }
            for c in shown { Attachment(c, orientation: .right) }
        }
        return r.content
    }
}

/// Development check, run when the app is started with `-selftest-naming`: names the biggest boxes
/// of the newest saved scan that has photos from their best photos, exactly as "Ask the AI again"
/// does, and writes the answers to the log.
enum NamingSelfTest {
    static var requested: Bool { ProcessInfo.processInfo.arguments.contains("-selftest-naming") }

    static func run() async {
        let log = Debug.shared
        log.log("self-test naming: started")
        guard let scan = ScanBundle.list().first(where: { (try? $0.keyframes())?.isEmpty == false && $0.loadRoom()?.objects?.isEmpty == false }),
              let room = scan.loadRoom(), let kfs = try? scan.keyframes() else {
            log.log("self-test naming: no saved scan with photos and a room"); return
        }
        let boxes = (room.objects ?? []).sorted { $0.volume > $1.volume }.prefix(3)
        log.log("self-test naming: scan \(scan.meta.id), \(boxes.count) box(es)")
        for b in boxes {
            let views = BoxViews.best(b, keyframes: kfs) { try? scan.depth($0) }
            let crops = views.compactMap { BoxViews.crop($0, image: scan.imageURL($0.keyframe)) }
            guard !crops.isEmpty else { log.log("self-test naming: box \(b.id): no photo"); continue }
            let t0 = Date()
            do {
                let r = try await BoxNamer.name(crops: crops, size: BoxNamer.sizeText(b))
                log.log(String(format: "self-test naming: box %d (%@) → %@, %@ (%.2f) from %d photo(s) in %d ms", b.id,
                               b.label ?? "unnamed", r.name, r.kind.rawValue, r.confidence, crops.count,
                               Int(Date().timeIntervalSince(t0) * 1000)))
            } catch {
                log.log("self-test naming: box \(b.id) FAILED: \(error)")
            }
        }
        log.log("self-test naming: done")
    }
}
