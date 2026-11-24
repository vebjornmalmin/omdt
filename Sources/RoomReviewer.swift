import Foundation
import FoundationModels

@Generable
enum ReviewActionOut: String, CaseIterable {
    case merge, rename, remove
}

@Generable
struct ReviewEditOut {
    @Guide(description: "merge joins two touching boxes that are pieces of one thing; rename gives a box a name that fits its size and place better; remove takes out a box that is not a real thing")
    var action: ReviewActionOut
    @Guide(description: "Number of the box to change, from the list")
    var box: Int
    @Guide(description: "For merge, the number of the other box; otherwise 0")
    var other: Int
    @Guide(description: "For rename and merge, the everyday name of the whole thing, 1 to 3 lowercase words; otherwise empty")
    var name: String
    @Guide(description: "A few words on why")
    var reason: String
}

@Generable
struct RoomReviewOut {
    @Guide(description: "What kind of room this is, 1 to 2 lowercase words, for example: bedroom, office, living room, kitchen, bathroom, hallway")
    var roomType: String
    @Guide(description: "Changes that fix clear mistakes, each box at most once. Empty when the list looks right.", .maximumCount(8))
    var edits: [ReviewEditOut]
}

/// Runs the whole-room review on the on-device model: it reads the room as text (see
/// `RoomReview.describe`) and suggests merges, renames and removals; `RoomReview.apply` checks
/// each one against the geometry. Repeats on the changed room until nothing more passes, up to
/// `rounds` times.
enum RoomReviewer {
    struct Result {
        var room: RoomModel
        var applied: [String]
        var rejected: [String]
        var roomType: String?
        var rounds: Int
        var ms: Int
    }

    struct Unavailable: LocalizedError {
        var reason: String
        var errorDescription: String? { "Apple Intelligence is not available (\(reason))." }
    }

    static let instructions = """
        You check a list of boxes that a 3D scanner found in one room. Each box should hold one piece \
        of furniture or one object. Sizes are width × depth in cm, then height. Fix only clear mistakes:
        - merge: two boxes that touch and are pieces of the same thing, such as a bed or sofa found as \
        several pieces at the same height, or a pillow on a bed. Give the name of the whole thing.
        - rename: a box whose name does not fit its size, height or place. Image classifier guesses are \
        often wrong; AI names from a photo are usually right. A box named "chair" that is 2.5 m wide is \
        more likely a sofa or a desk. Name an unnamed box only when its size and place make it obvious.
        - remove: a box that is not a real thing, such as one that covers most of the room, or a thin \
        piece of wall, floor or ceiling.
        Never change boxes named by the user. Use only box numbers from the list. Do not merge things \
        that only stand next to each other, like a chair at a desk or a lamp on a table. Return no edits \
        when the list looks right.
        """

    static func review(_ room: RoomModel, rounds: Int = 2) async throws -> Result {
        let model = SystemLanguageModel.default
        if case .unavailable(let r) = model.availability { throw Unavailable(reason: "\(r)") }
        let t0 = Date()
        var current = room
        var applied: [String] = [], rejected: [String] = []
        var roomType: String?
        var done = 0
        for _ in 0..<rounds {
            done += 1
            let t = Date()
            let out: RoomReviewOut
            do {
                out = try await ask(current)
            } catch where done > 1 {
                // Keep what the earlier rounds found.
                Debug.shared.log("room review round \(done) failed, keeping round \(done - 1): \(error)")
                break
            }
            roomType = RoomReview.cleanName(out.roomType) ?? roomType
            // The same suggestion twice is a model stuck repeating itself; keep the first.
            var seen = Set<String>()
            let edits = out.edits.compactMap { e -> ReviewEdit? in
                let edit = ReviewEdit(action: ReviewEdit.Action(rawValue: e.action.rawValue) ?? .rename, box: e.box,
                                      other: e.action == .merge ? e.other : nil,
                                      name: e.name.isEmpty ? nil : e.name, reason: e.reason)
                return seen.insert(RoomReview.describe(edit)).inserted ? edit : nil
            }
            Debug.shared.log("room review round \(done): \(roomType ?? "?"), \(edits.count) suggestion(s) in "
                             + "\(Int(Date().timeIntervalSince(t) * 1000)) ms")
            let o = RoomReview.apply(edits, to: current)
            for n in o.applied { Debug.shared.log("room review: " + n) }
            for n in o.rejected { Debug.shared.log("room review: skipped " + n) }
            current = o.room
            applied += o.applied; rejected += o.rejected
            if o.applied.isEmpty { break }
        }
        current.roomType = roomType ?? current.roomType
        let ms = Int(Date().timeIntervalSince(t0) * 1000)
        return Result(room: current, applied: applied, rejected: rejected, roomType: roomType, rounds: done, ms: ms)
    }

    /// One pass. Falls back to fewer boxes when the list doesn't fit the context window.
    private static func ask(_ room: RoomModel) async throws -> RoomReviewOut {
        var limit = RoomReview.maxBoxes
        while true {
            let session = LanguageModelSession(instructions: instructions)
            do {
                let r = try await session.respond(to: RoomReview.describe(room, maxBoxes: limit),
                                                  generating: RoomReviewOut.self,
                                                  options: GenerationOptions(temperature: 0.3, maximumResponseTokens: 900))
                return r.content
            } catch LanguageModelSession.GenerationError.exceededContextWindowSize where limit > 10 {
                limit /= 2
                Debug.shared.log("room review: too long, trying the \(limit) biggest boxes")
            }
        }
    }
}
