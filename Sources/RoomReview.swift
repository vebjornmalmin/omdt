import Foundation
import simd

/// One change the room review asks for.
struct ReviewEdit: Equatable {
    enum Action: String { case merge, rename, remove }
    var action: Action
    var box: Int
    /// The second box of a merge.
    var other: Int?
    /// New name, for a rename or a merge.
    var name: String?
    var reason: String = ""
}

/// The whole-room review: the room written out as text for the on-device model, and the checks
/// its answers have to pass before they change anything. The model only reads names, sizes and
/// positions, so it can catch what one photo at a time can't (a "chair" 2.7 m wide, four flat
/// pieces at bed height), but geometry has the last word:
/// - merge: the boxes touch, the joined box is a believable size for its name, the two fill it,
///   and it doesn't swallow a third box;
/// - rename: the box's size fits the new name;
/// - remove: only boxes nobody is sure about (unnamed, photo guesses, unclear, tiny, or as big
///   as the room), never one the model named with confidence from a photo.
/// Boxes named by hand are never touched.
enum RoomReview {
    /// Most boxes written out, biggest first, to stay inside the model's context window.
    static let maxBoxes = 40

    // MARK: describing

    /// The room as one line per box, in the room's own directions, measured from its corner.
    static func describe(_ room: RoomModel, maxBoxes: Int = maxBoxes) -> String {
        let shell = room.shell
        let all = (room.objects ?? []).sorted { $0.volume > $1.volume }
        let shown = Array(all.prefix(maxBoxes))
        let corner = roomCorner(shell)
        let bounds = Dictionary(uniqueKeysWithValues: shown.map { ($0.id, RoomCorrector.roomBounds($0, shell: shell)) })
        let e = shell.extent.count == 2 ? shell.extent : [0, 0]
        var lines = [String(format: "Room: %.1f × %.1f m floor, %.1f m high. %d things%@.", e[0], e[1], shell.height,
                            all.count, all.count > shown.count ? ", the \(shown.count) biggest listed" : "")]
        for b in shown {
            let (l, h) = bounds[b.id]!
            let cm = { (x: Float) in Int((x * 100).rounded()) }
            var s = "#\(b.id) \(nameText(b)): \(cm(h.x - l.x))×\(cm(h.z - l.z)) cm, \(cm(h.y - l.y)) cm tall, "
            s += "\(cm(max(0, l.y - shell.floorY)))–\(cm(h.y - shell.floorY)) cm above the floor, "
            s += String(format: "at %.1f, %.1f m", (l.x + h.x) / 2 - corner.x, (l.z + h.z) / 2 - corner.y)
            if againstWall(l, h, shell: shell) { s += ", against a wall" }
            if RoomCorrector.footprint(b, shell: shell) > 0.6 * max(shell.area, 0.1) {
                lines.append(s + ", covers most of the room"); continue
            }
            var on: [Int] = [], under: [Int] = [], beside: [Int] = []
            for c in shown where c.id != b.id && RoomCorrector.footprint(c, shell: shell) <= 0.6 * max(shell.area, 0.1) {
                let (cl, ch) = bounds[c.id]!
                guard gap(l, h, cl, ch) <= 0.08 else { continue }
                if overlapShare(l, h, over: cl, ch) > 0.6 && l.y >= ch.y - 0.05 { on.append(c.id) }
                else if overlapShare(cl, ch, over: l, h) > 0.6 && cl.y >= h.y - 0.05 { under.append(c.id) }
                else { beside.append(c.id) }
            }
            let ids = { (x: [Int]) in x.map { "#\($0)" }.joined(separator: ", ") }
            if !on.isEmpty { s += "; on top of \(ids(on))" }
            if !under.isEmpty { s += "; holds \(ids(under))" }
            if !beside.isEmpty { s += "; touches \(ids(beside))" }
            lines.append(s)
        }
        return lines.joined(separator: "\n")
    }

    private static func nameText(_ b: ObjectBox) -> String {
        guard let l = b.label else { return "unnamed" }
        let pct = Int(((b.confidence ?? 0) * 100).rounded())
        switch b.source {
        case "user": return "\(l) (named by the user)"
        case "model": return "\(l) (AI from a photo, \(pct)%)"
        case "review": return "\(l) (earlier review)"
        default: return "\(l) (image classifier guess, \(pct)%)"
        }
    }

    /// The room-frame (u, v) corner with the smallest coordinates.
    private static func roomCorner(_ shell: RoomShell) -> SIMD2<Float> {
        let c = cos(shell.yaw), s = sin(shell.yaw)
        return shell.outline.reduce(SIMD2<Float>(repeating: .infinity)) { m, p in
            simd_min(m, SIMD2(p[0] * c + p[1] * s, -p[0] * s + p[1] * c))
        }
    }

    /// True when a side of the box is within 10 cm of a wall (or past it), alongside it.
    static func againstWall(_ l: SIMD3<Float>, _ h: SIMD3<Float>, shell: RoomShell) -> Bool {
        let c = cos(shell.yaw), s = sin(shell.yaw)
        func toRoom(_ p: [Float]) -> SIMD2<Float> { SIMD2(p[0] * c + p[1] * s, -p[0] * s + p[1] * c) }
        for w in shell.walls {
            let a = toRoom(w.start), b = toRoom(w.end), n = toRoom(w.normal)
            if abs(n.x) > 0.9 {           // wall across u, at u = a.x, room on the n.x side
                guard min(h.z, max(a.y, b.y)) - max(l.z, min(a.y, b.y)) > 0 else { continue }
                if (n.x > 0 ? l.x - a.x : a.x - h.x) < 0.10 { return true }
            } else if abs(n.y) > 0.9 {
                guard min(h.x, max(a.x, b.x)) - max(l.x, min(a.x, b.x)) > 0 else { continue }
                if (n.y > 0 ? l.z - a.y : a.y - h.z) < 0.10 { return true }
            }
        }
        return false
    }

    /// Largest gap between two boxes along any axis (≤ 0 when they overlap).
    private static func gap(_ al: SIMD3<Float>, _ ah: SIMD3<Float>, _ bl: SIMD3<Float>, _ bh: SIMD3<Float>) -> Float {
        let g = simd_max(bl - ah, al - bh)
        return max(g.x, g.y, g.z)
    }

    /// Share of the first box's footprint that lies over the second's.
    private static func overlapShare(_ tl: SIMD3<Float>, _ th: SIMD3<Float>, over bl: SIMD3<Float>, _ bh: SIMD3<Float>) -> Float {
        let ou = max(0, min(th.x, bh.x) - max(tl.x, bl.x)), ov = max(0, min(th.z, bh.z) - max(tl.z, bl.z))
        return ou * ov / max((th.x - tl.x) * (th.z - tl.z), 1e-6)
    }

    // MARK: checking and applying

    struct Outcome {
        var room: RoomModel
        /// One line per change made.
        var applied: [String] = []
        /// One line per suggestion turned down, with why.
        var rejected: [String] = []
    }

    /// Applies the edits that pass the checks, in order.
    static func apply(_ edits: [ReviewEdit], to room: RoomModel) -> Outcome {
        var out = Outcome(room: room)
        for e in edits {
            switch check(e, in: out.room) {
            case .success(let (r, note)): out.room = r; out.applied.append(note)
            case .failure(let why): out.rejected.append("\(describe(e)): \(why.text)")
            }
        }
        return out
    }

    static func describe(_ e: ReviewEdit) -> String {
        switch e.action {
        case .merge: return "merge #\(e.box) and #\(e.other ?? 0)" + (e.name.map { " as \($0)" } ?? "")
        case .rename: return "rename #\(e.box) to \(e.name ?? "?")"
        case .remove: return "remove #\(e.box)"
        }
    }

    struct Rejection: Error { var text: String }

    private static func check(_ e: ReviewEdit, in room: RoomModel) -> Result<(RoomModel, String), Rejection> {
        let shell = room.shell
        var objects = room.objects ?? []
        guard let i = objects.firstIndex(where: { $0.id == e.box }) else { return .failure(Rejection(text: "no box #\(e.box)")) }
        let b = objects[i]
        guard b.source != "user" || e.action == .merge else { return .failure(Rejection(text: "named by the user")) }
        let newName = e.name.flatMap(cleanName)

        switch e.action {
        case .rename:
            guard let name = newName else { return .failure(Rejection(text: "not a usable name")) }
            if RoomCorrector.structural.contains(name) {
                return remove(b, at: i, in: room, kind: name, reason: e.reason)
            }
            guard name != b.label?.lowercased() else { return .failure(Rejection(text: "same name")) }
            // A text-only guess overrules a confident name from a photo only when that name
            // can't be right for the box's size.
            if b.source == "model", (b.confidence ?? 0) >= 0.5, let old = b.label, fitsName(b, old, shell: shell, slack: 1.15) {
                return .failure(Rejection(text: "the photo AI is sure it's a \(old), and that fits its size"))
            }
            guard fitsName(b, name, shell: shell, slack: 1.15) else {
                return .failure(Rejection(text: "\(sizeText(b, shell)) is too big for a \(name)"))
            }
            objects[i].label = name; objects[i].source = "review"; objects[i].confidence = 0.6; objects[i].kind = nil
            var r = room; r.objects = objects
            return .success((r, "renamed #\(b.id) \(b.label ?? "unnamed") → \(name)" + because(e.reason)))

        case .remove:
            return remove(b, at: i, in: room, kind: "review", reason: e.reason)

        case .merge:
            guard let oid = e.other, oid != b.id, let j = objects.firstIndex(where: { $0.id == oid }) else {
                return .failure(Rejection(text: "no second box"))
            }
            let o = objects[j]
            if b.source == "user", o.source == "user", b.label?.lowercased() != o.label?.lowercased() {
                return .failure(Rejection(text: "both named by the user"))
            }
            let (al, ah) = RoomCorrector.roomBounds(b, shell: shell), (bl, bh) = RoomCorrector.roomBounds(o, shell: shell)
            guard gap(al, ah, bl, bh) <= 0.12 else { return .failure(Rejection(text: "they don't touch")) }
            // One standing on the other is two things when both have names of different kinds
            // (a lamp on a desk); a pillow or something unnamed on a bed can be part of it.
            for (t, tl, th, u, ul, uh) in [(b, al, ah, o, bl, bh), (o, bl, bh, b, al, ah)]
            where overlapShare(tl, th, over: ul, uh) > 0.6 && tl.y >= uh.y - 0.05 {
                let tc = BoxCategory.of(t.label), uc = BoxCategory.of(u.label)
                if t.label != nil, u.label != nil, tc != uc, tc != .other, tc != .decor {
                    return .failure(Rejection(text: "the \(t.label!) stands on the \(u.label!)"))
                }
            }
            var m = RoomCorrector.merge(b, o, shell: shell)
            if m.source != "user", let name = newName, !RoomCorrector.structural.contains(name) {
                if name != m.label?.lowercased() { m.label = name; m.source = "review"; m.confidence = 0.6; m.kind = nil }
            }
            guard RoomCorrector.fits(m, shell: shell) else {
                return .failure(Rejection(text: "\(sizeText(m, shell)) together is too big for \(m.label.map { "a \($0)" } ?? "one thing")"))
            }
            let fill = RoomCorrector.footprint(m, shell: shell)
                / (RoomCorrector.footprint(b, shell: shell) + RoomCorrector.footprint(o, shell: shell))
            guard fill <= 2.5 else { return .failure(Rejection(text: "the two fill too little of the joined box")) }
            let (ml, mh) = RoomCorrector.roomBounds(m, shell: shell)
            if let c = objects.first(where: { $0.id != b.id && $0.id != o.id && inside($0, ml, mh, shell: shell) }) {
                return .failure(Rejection(text: "the joined box would swallow #\(c.id)"))
            }
            let gone = m.id == b.id ? o : b
            objects[m.id == b.id ? i : j] = m
            objects.removeAll { $0.id == gone.id }
            var r = room; r.objects = objects.sorted { $0.volume > $1.volume }
            return .success((r, "merged #\(gone.id) into #\(m.id)" + (m.label.map { " (\($0))" } ?? "") + because(e.reason)))
        }
    }

    /// Takes a box out when nobody is sure what it is.
    private static func remove(_ b: ObjectBox, at i: Int, in room: RoomModel, kind: String, reason: String) -> Result<(RoomModel, String), Rejection> {
        let shell = room.shell
        let unsure = b.label == nil || b.source == "vision" || b.source == "review"
            || b.kind == "unclear" || RoomCorrector.structural.contains(b.kind ?? "")
            || (b.confidence ?? 0) < 0.5
        let tiny = b.volume < 0.02
        let roomSized = RoomCorrector.footprint(b, shell: shell) > 0.6 * max(shell.area, 0.1)
        guard unsure || tiny || roomSized else {
            return .failure(Rejection(text: "the photo AI is sure it's a \(b.label ?? "thing")"))
        }
        var r = room
        var objects = room.objects ?? []
        var gone = objects.remove(at: i)
        gone.kind = kind
        r.objects = objects
        r.removed = (room.removed ?? []) + [gone]
        return .success((r, "removed #\(b.id) \(b.label ?? "unnamed")" + because(reason)))
    }

    /// Lowercase letters and spaces, 1 to 4 words; nil when nothing usable is left.
    static func cleanName(_ s: String) -> String? {
        let words = s.lowercased().split { !$0.isLetter && $0 != "-" }.prefix(4)
        let n = words.joined(separator: " ")
        return n.isEmpty || n.count > 32 || ["unnamed", "unknown", "none", "nothing", "object", "thing"].contains(n) ? nil : n
    }

    /// True when a box's size suits a name, with some slack.
    static func fitsName(_ b: ObjectBox, _ name: String, shell: RoomShell, slack: Float) -> Bool {
        let (l, h) = RoomCorrector.roomBounds(b, shell: shell)
        let du = h.x - l.x, dv = h.z - l.z
        let limit = RoomCorrector.maxSize(BoxCategory.of(name))
        return max(du, dv) <= limit.long * slack && min(du, dv) <= limit.short * slack && h.y - l.y <= limit.height * slack
    }

    /// True when most of `c` lies inside the room-frame bounds.
    private static func inside(_ c: ObjectBox, _ l: SIMD3<Float>, _ h: SIMD3<Float>, shell: RoomShell) -> Bool {
        let (cl, ch) = RoomCorrector.roomBounds(c, shell: shell)
        let e = simd_max(simd_min(ch, h + 0.03) - simd_max(cl, l - 0.03), .zero)
        let v = simd_max(ch - cl, SIMD3(repeating: 0.01))
        return e.x * e.y * e.z / (v.x * v.y * v.z) > 0.5
    }

    private static func sizeText(_ b: ObjectBox, _ shell: RoomShell) -> String {
        let (l, h) = RoomCorrector.roomBounds(b, shell: shell)
        return String(format: "%.1f × %.1f m", max(h.x - l.x, h.z - l.z), min(h.x - l.x, h.z - l.z))
    }

    private static func because(_ reason: String) -> String {
        let r = reason.trimmingCharacters(in: .whitespacesAndNewlines)
        return r.isEmpty ? "" : " (\(r))"
    }
}
