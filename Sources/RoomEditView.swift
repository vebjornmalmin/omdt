import SceneKit
import SwiftUI

/// Walls whose outside faces the camera, hidden every frame so you can look into the room.
final class WallCuller: @unchecked Sendable {
    var walls: [(node: SCNNode, mid: SIMD2<Float>, normal: SIMD2<Float>)] = []

    func update(camera: SIMD3<Float>) {
        let c = SIMD2(camera.x, camera.z)
        for w in walls {
            let hide = simd_dot(w.normal, c - w.mid) < 0     // camera is on the outside of this wall
            if w.node.isHidden != hide { w.node.isHidden = hide }
        }
    }
}

@MainActor
final class RoomEditModel: ObservableObject {
    let scan: ScanBundle
    @Published private(set) var editor: RoomEditor
    @Published var selected: Int? {
        didSet {
            merging = false
            if oldValue != selected { boxNote = nil; syncBoxes(); loadPhotos() }
        }
    }
    @Published var merging = false
    @Published var showScan = false { didSet { ghost.isHidden = !showScan } }

    let scene = SCNScene()
    let culler = WallCuller()
    private let ghost = SCNNode()
    private let boxesRoot = SCNNode()
    private var boxNodes: [Int: (node: SCNNode, box: ObjectBox, selected: Bool)] = [:]
    private let onSave: (RoomModel) -> Void

    init(scan: ScanBundle, room: RoomModel, scanGeometry: SCNGeometry?, onSave: @escaping (RoomModel) -> Void) {
        self.scan = scan
        self.editor = RoomEditor(room: room)
        self.onSave = onSave
        buildScene(scanGeometry)
        syncBoxes()
    }

    var shell: RoomShell { editor.room.shell }
    var objects: [ObjectBox] { editor.objects }
    var selectedBox: ObjectBox? { selected.flatMap { editor.box($0) } }

    /// Middle of the floor polygon, at half the room's height.
    var center: SIMD3<Float> {
        let o = shell.outline
        guard !o.isEmpty else { return .zero }
        let x = o.map { $0[0] }.reduce(0, +) / Float(o.count), z = o.map { $0[1] }.reduce(0, +) / Float(o.count)
        return SIMD3(x, shell.floorY + shell.height / 2, z)
    }

    static let fieldOfView: Float = 60

    /// Where the camera starts: looking down the room's long axis at 50°, far enough back that
    /// the short side fits the width of a portrait screen.
    var cameraStart: SIMD3<Float> {
        let e = shell.extent.count == 2 ? shell.extent : [4, 4]
        let long = max(e[0], e[1]) / 2 + 0.6, short = min(e[0], e[1]) / 2 + 0.6
        // Direction of the long axis on the floor.
        let yaw = shell.yaw + (e[0] >= e[1] ? 0 : .pi / 2)
        let axis = SIMD3(cos(yaw), 0, sin(yaw))
        let tanV = tan(Self.fieldOfView / 2 * .pi / 180), tanH = tanV * 0.46
        let dist = max(short / tanH, long / tanV) * 0.85
        let pitch: Float = 50 * .pi / 180
        return center - axis * (dist * cos(pitch)) + SIMD3(0, dist * sin(pitch), 0)
    }

    private func buildScene(_ scanGeometry: SCNGeometry?) {
        let root = scene.rootNode
        root.addChildNode(SCNNode(geometry: RoomScene.floorGeometry(shell)))
        for w in shell.walls {
            let n = RoomScene.wallNode(w, shell: shell)
            root.addChildNode(n)
            let mid = (SIMD2(w.start[0], w.start[1]) + SIMD2(w.end[0], w.end[1])) / 2
            culler.walls.append((n, mid, SIMD2(w.normal[0], w.normal[1])))
        }
        root.addChildNode(boxesRoot)

        if let g = scanGeometry?.copy() as? SCNGeometry {
            let m = MeshGeometry.vertexColorMaterial()
            m.transparency = 0.45
            m.writesToDepthBuffer = false
            g.materials = [m]
            ghost.geometry = g
            ghost.renderingOrder = 50
        }
        ghost.isHidden = true
        root.addChildNode(ghost)

        let ambient = SCNNode(); ambient.light = SCNLight(); ambient.light?.type = .ambient
        ambient.light?.intensity = 650
        let sun = SCNNode(); sun.light = SCNLight(); sun.light?.type = .directional
        sun.light?.intensity = 900
        sun.simdEulerAngles = SIMD3(-1.0, 0.6, 0)
        root.addChildNode(ambient); root.addChildNode(sun)
    }

    // MARK: box nodes

    /// Brings the box nodes in line with the editor: rebuilds changed boxes, moves moved ones.
    func syncBoxes() {
        let list = objects
        let ids = Set(list.map(\.id))
        for (id, e) in boxNodes where !ids.contains(id) { e.node.removeFromParentNode(); boxNodes[id] = nil }
        for b in list {
            let sel = b.id == selected
            if let e = boxNodes[b.id] {
                var moved = e.box; moved.center = b.center
                if moved == b && e.selected == sel {                // only the position changed
                    if e.box.center != b.center { RoomScene.place(e.node, b); boxNodes[b.id]?.box = b }
                    continue
                }
                e.node.removeFromParentNode()
            }
            let node = makeBoxNode(b, selected: sel)
            boxesRoot.addChildNode(node)
            boxNodes[b.id] = (node, b, sel)
        }
    }

    private func makeBoxNode(_ b: ObjectBox, selected sel: Bool) -> SCNNode {
        let node = SCNNode()
        node.name = "box-\(b.id)"
        RoomScene.place(node, b)
        let cat = BoxCategory.of(b.label)

        let solid = RoomScene.boxNode(b, transparency: sel ? 1 : 0.92)
        solid.simdPosition = .zero; solid.simdEulerAngles = .zero
        solid.categoryBitMask = 2
        if sel { solid.geometry?.firstMaterial?.emission.contents = UIColor(white: 0.25, alpha: 1) }
        node.addChildNode(solid)

        // Edges.
        let h = SIMD3(b.size[0], b.size[1], b.size[2]) / 2 + 0.002
        var pts: [SCNVector3] = []
        for dy: Float in [-1, 1] {
            for (du, dv): (Float, Float) in [(-1, -1), (1, -1), (1, 1), (-1, 1)] {
                pts.append(SCNVector3(SIMD3(du * h.x, dy * h.y, dv * h.z)))
            }
        }
        let idx: [UInt32] = [0, 1, 1, 2, 2, 3, 3, 0, 4, 5, 5, 6, 6, 7, 7, 4, 0, 4, 1, 5, 2, 6, 3, 7]
        let lines = SCNGeometry(sources: [SCNGeometrySource(vertices: pts)],
                                elements: [SCNGeometryElement(indices: idx, primitiveType: .line)])
        let lm = SCNMaterial(); lm.lightingModel = .constant
        lm.diffuse.contents = sel ? UIColor.white : UIColor(white: 0.15, alpha: 0.8)
        lm.readsFromDepthBuffer = !sel
        lines.materials = [lm]
        let edges = SCNNode(geometry: lines)
        edges.renderingOrder = 60
        node.addChildNode(edges)

        // Name tag above the box.
        if let name = b.label {
            let tag = SCNNode(geometry: BoxOverlay.labelPlane(name, color: UIColor(cgColor: RoomScene.color(cat.rgb)),
                                                              height: sel ? 0.14 : 0.11))
            tag.simdPosition = SIMD3(0, h.y + 0.12, 0)
            let bc = SCNBillboardConstraint(); bc.freeAxes = .all
            tag.constraints = [bc]
            tag.renderingOrder = 100
            node.addChildNode(tag)
        }
        return node
    }

    // MARK: edits

    func beginMove() { editor.beginMove() }

    func move(to xz: SIMD2<Float>) {
        guard let id = selected else { return }
        editor.move(id, to: xz)
        syncBoxes()
    }

    func endMove() { save() }

    /// A tap on box `id` (or on nothing).
    func tapped(_ id: Int?) {
        if merging, let sel = selected, let other = id, other != sel {
            let keep = editor.merge(sel, other)
            Debug.shared.log("room edit: merged box \(other) into \(sel)")
            commit()
            selected = keep
            loadPhotos()
            return
        }
        selected = id == selected ? nil : id
    }

    func rotate() {
        guard let id = selected else { return }
        editor.rotate(id); commit()
    }

    func rename(_ name: String) {
        guard let id = selected else { return }
        editor.rename(id, to: name); commit()
    }

    // MARK: photos and naming

    /// The best saved photos of the selected box (portrait), and their crops for the model.
    @Published private(set) var photos: [UIImage] = []
    private var photoCrops: [CGImage] = []
    private var keyframes: [Keyframe]?
    @Published private(set) var asking = false

    private func loadPhotos() {
        photos = []; photoCrops = []
        guard let b = selectedBox else { return }
        let scan = scan, cached = keyframes
        Task { @MainActor in
            let found = await Task.detached(priority: .userInitiated) { () -> ([Keyframe], [CGImage]) in
                let kfs = cached ?? (try? scan.keyframes()) ?? []
                let views = BoxViews.best(b, keyframes: kfs) { try? scan.depth($0) }
                return (kfs, views.compactMap { BoxViews.crop($0, image: scan.imageURL($0.keyframe)) })
            }.value
            keyframes = found.0
            guard selected == b.id else { return }
            photoCrops = found.1
            photos = found.1.map { UIImage(cgImage: $0, scale: 1, orientation: .right) }
        }
    }

    /// Names the selected box again from its best photos.
    func askAgain() {
        guard let b = selectedBox, !asking else { return }
        guard !photoCrops.isEmpty else { boxNote = "No good photo of this box to ask about."; return }
        asking = true; boxNote = nil
        let crops = photoCrops, id = b.id
        Task { @MainActor in
            defer { asking = false }
            do {
                let t0 = Date()
                let r = try await BoxNamer.name(crops: crops, size: BoxNamer.sizeText(b))
                let ms = Int(Date().timeIntervalSince(t0) * 1000)
                Debug.shared.log(String(format: "room edit: AI named box %d %@ (%@, %.2f) from %d photo(s) in %d ms",
                                        id, r.name, r.kind.rawValue, r.confidence, crops.count, ms))
                guard editor.box(id) != nil else { return }
                editor.setModelName(id, name: r.name, confidence: Float(r.confidence), kind: r.kind.rawValue)
                commit()
                guard selected == id else { return }
                let structural = RoomCorrector.structural.contains(r.kind.rawValue)
                boxNote = structural ? "The AI thinks this is part of the room (\(r.kind.rawValue)). Clean up will take it out."
                                       : "The AI says: \(r.name)."
            } catch {
                Debug.shared.log("room edit: AI naming failed: \(error)")
                guard selected == id else { return }
                boxNote = error is BoxNamer.Unavailable ? error.localizedDescription
                                                          : "The AI couldn't name it right now. Try again in a moment."
            }
        }
    }

    /// Splitting: loading the scan's points (first time only). `boxNote` shows the last split's
    /// or naming's result under the selected box.
    @Published private(set) var splitting = false
    @Published var boxNote: String?
    private var scanPoints: [SIMD3<Float>]?

    /// Cuts the selected box in two where the scan shows a gap or a step between two things.
    func split() {
        guard let id = selected, !splitting else { return }
        splitting = true; boxNote = nil
        let url = scan.meshURL, cached = scanPoints
        Task { @MainActor in
            defer { splitting = false }
            let pts: [SIMD3<Float>]
            if let cached {
                pts = cached
            } else {
                let loaded = await Task.detached(priority: .userInitiated) { () -> [SIMD3<Float>]? in
                    guard let mesh = try? MeshExporter.readPLY(url) else { return nil }
                    return RoomFitter.sample([mesh], cell: 0.03).map(\.p)
                }.value
                guard let loaded else { if selected == id { boxNote = "Couldn't read the scan to split this box." }; return }
                scanPoints = loaded; pts = loaded
            }
            guard selected == id else { return }
            if let (a, b) = editor.split(id, points: pts) {
                Debug.shared.log("room edit: split box \(a) into \(a) and \(b)")
                commit()
                selected = a
                loadPhotos()
                boxNote = "Split in two. Tap the other part to name it."
            } else {
                boxNote = "No clear gap or step to cut this box at."
            }
        }
    }

    func delete() {
        guard let id = selected else { return }
        editor.delete(id)
        Debug.shared.log("room edit: deleted box \(id)")
        selected = nil
        commit()
    }

    /// Last clean-up's result, shown in the panel.
    @Published var cleanUpNote: String?

    func cleanUp() {
        reviewNote = nil; reviewChanges = []
        let notes = editor.cleanUp()
        for n in notes { Debug.shared.log("clean up: " + n) }
        if notes.isEmpty {
            cleanUpNote = "Nothing to fix."
        } else {
            let removed = notes.filter { $0.hasPrefix("removed") || $0.hasPrefix("kept") }.count
            let merged = notes.filter { $0.hasPrefix("merged") }.count
            let stood = notes.filter { $0.contains("stands on the floor") }.count
            var parts: [String] = []
            if removed > 0 { parts.append("\(removed) wall, door or window piece\(removed == 1 ? "" : "s") removed") }
            if merged > 0 { parts.append("\(merged) piece\(merged == 1 ? "" : "s") merged") }
            if stood > 0 { parts.append("\(stood) put on the floor") }
            cleanUpNote = parts.joined(separator: ", ").capitalizedFirstLetter + ". Undo puts it back."
        }
        fixSelection(); commit()
    }

    /// The room review: running, and its last result for the panel.
    @Published private(set) var reviewing = false
    @Published var reviewNote: String?
    @Published var reviewChanges: [String] = []

    func review() {
        guard !reviewing else { return }
        reviewing = true; reviewNote = nil; reviewChanges = []; cleanUpNote = nil; selected = nil
        let before = editor.room
        Task { @MainActor in
            defer { reviewing = false }
            do {
                Debug.shared.log("room review started: \(before.objects?.count ?? 0) boxes")
                let r = try await RoomReviewer.review(before)
                Debug.shared.log("room review: \(r.roomType ?? "?"), \(r.applied.count) changed, \(r.rejected.count) skipped, "
                                 + "\(r.rounds) round(s) in \(r.ms) ms")
                guard editor.room == before else {
                    reviewNote = "The room changed during the review. Run it again."
                    return
                }
                editor.replace(with: r.room)
                let secs = String(format: "%.0f s", Double(r.ms) / 1000)
                var note = (r.roomType?.capitalizedFirstLetter).map { "\($0). " } ?? ""
                if r.applied.isEmpty {
                    note += "No changes needed (\(secs))."
                } else {
                    note += "\(r.applied.count) change\(r.applied.count == 1 ? "" : "s") (\(secs)). Undo puts it back."
                }
                if !r.rejected.isEmpty {
                    note += " \(r.rejected.count) suggestion\(r.rejected.count == 1 ? "" : "s") didn't fit the geometry."
                }
                reviewNote = note
                reviewChanges = r.applied
                fixSelection(); commit()
            } catch {
                Debug.shared.log("room review failed: \(error)")
                reviewNote = error.localizedDescription
            }
        }
    }

    func undo() { editor.undo(); fixSelection(); commit(); loadPhotos() }
    func redo() { editor.redo(); fixSelection(); commit(); loadPhotos() }

    private func fixSelection() { if let s = selected, editor.box(s) == nil { selected = nil } }

    private func commit() { syncBoxes(); save() }

    private func save() {
        do { try scan.saveRoom(editor.room) } catch { Debug.shared.log("room edit: save failed: \(error)") }
        onSave(editor.room)
    }

    // MARK: export

    var jsonURL: URL { scan.url.appendingPathComponent(ScanBundle.roomFile) }

    func exportUSDZ() -> URL? {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("Room \(scan.meta.id).usdz")
        do {
            try RoomScene.writeUSDZ(editor.room, to: url)
            Debug.shared.log("room export: \(objects.count) boxes to \(url.lastPathComponent)")
            return url
        } catch {
            Debug.shared.log("room export failed: \(error)")
            return nil
        }
    }
}

// MARK: view

struct RoomEditView: View {
    @StateObject private var model: RoomEditModel
    @State private var renaming = false
    @State private var draftName = ""
    @State private var share: ShareItem?

    init(scan: ScanBundle, room: RoomModel, scanGeometry: SCNGeometry?, onSave: @escaping (RoomModel) -> Void) {
        _model = StateObject(wrappedValue: RoomEditModel(scan: scan, room: room, scanGeometry: scanGeometry, onSave: onSave))
    }

    var body: some View {
        ZStack(alignment: .bottom) {
            Color.black.ignoresSafeArea()
            RoomSceneView(model: model).ignoresSafeArea(edges: .bottom)
            panel
        }
        .navigationTitle("Room")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItemGroup(placement: .primaryAction) {
                Button { model.undo() } label: { Image(systemName: "arrow.uturn.backward") }
                    .disabled(!model.editor.canUndo).accessibilityLabel("Undo")
                Button { model.redo() } label: { Image(systemName: "arrow.uturn.forward") }
                    .disabled(!model.editor.canRedo).accessibilityLabel("Redo")
                Button { model.showScan.toggle() } label: {
                    Image(systemName: model.showScan ? "eye.fill" : "eye")
                }
                .accessibilityLabel(model.showScan ? "Hide the scan" : "Show the scan")
                Menu {
                    Button { if let url = model.exportUSDZ() { share = ShareItem(url: url) } } label: {
                        Label("3D model (USDZ)", systemImage: "cube")
                    }
                    ShareLink(item: model.jsonURL) { Label("Room data (JSON)", systemImage: "curlybraces") }
                } label: { Image(systemName: "square.and.arrow.up") }
                .accessibilityLabel("Export")
            }
        }
        .alert("Name", isPresented: $renaming) {
            TextField("What is it?", text: $draftName)
            Button("Save") { model.rename(draftName) }
            Button("Cancel", role: .cancel) {}
        }
        .sheet(item: $share) { ActivityView(items: [$0.url]).presentationDetents([.medium, .large]) }
    }

    @ViewBuilder private var panel: some View {
        VStack(alignment: .leading, spacing: 10) {
            if let b = model.selectedBox {
                HStack(spacing: 10) {
                    Circle().fill(Color(cgColor: RoomScene.color(BoxCategory.of(b.label).rgb))).frame(width: 14, height: 14)
                    VStack(alignment: .leading, spacing: 2) {
                        Text(b.label ?? "Not named").font(.headline)
                        Text(details(b)).font(.caption).foregroundStyle(.secondary).monospacedDigit()
                    }
                    Spacer()
                    Button { model.selected = nil } label: { Image(systemName: "xmark.circle.fill").font(.title2) }
                        .foregroundStyle(.secondary).accessibilityLabel("Deselect")
                }
                if !model.photos.isEmpty {
                    ScrollView(.horizontal, showsIndicators: false) {
                        HStack(spacing: 6) {
                            ForEach(model.photos.indices, id: \.self) { i in
                                Image(uiImage: model.photos[i]).resizable().scaledToFill()
                                    .frame(width: 72, height: 96).clipShape(RoundedRectangle(cornerRadius: 6))
                            }
                        }
                    }
                    .accessibilityLabel("Photos of this box")
                }
                HStack(spacing: 8) {
                    // Menus opening upward list the first item nearest the button.
                    Menu {
                        Button("Type a name…") { draftName = b.label ?? ""; renaming = true }
                        if b.source != "user" {
                            Button("Ask the AI again") { model.askAgain() }.disabled(model.photos.isEmpty)
                        }
                        if b.label != nil { Button("Clear name", role: .destructive) { model.rename("") } }
                        Divider()
                        ForEach(BoxCategory.quickNames, id: \.self) { n in Button(n) { model.rename(n) } }
                    } label: {
                        if model.asking { ProgressView().frame(maxWidth: .infinity) } else { tool("Name", "character.cursor.ibeam") }
                    }
                    Button { model.rotate() } label: { tool("Turn", "rotate.right") }
                    Button { model.merging.toggle() } label: {
                        tool(model.merging ? "Cancel" : "Merge", "square.on.square")
                    }
                    Button { model.split() } label: {
                        if model.splitting { ProgressView().frame(maxWidth: .infinity) } else { tool("Split", "square.split.2x1") }
                    }
                    .disabled(model.splitting)
                    Button(role: .destructive) { model.delete() } label: { tool("Delete", "trash") }
                }
                .buttonStyle(.bordered)
                if model.merging {
                    Text("Tap another box to merge it into this one.").font(.footnote).foregroundStyle(.yellow)
                } else if let note = model.boxNote {
                    Text(note).font(.footnote).foregroundStyle(.yellow)
                } else {
                    Text("Drag the box to move it. It snaps to walls.").font(.footnote).foregroundStyle(.secondary)
                }
            } else {
                let named = model.objects.filter { $0.label != nil }.count
                HStack {
                    VStack(alignment: .leading, spacing: 2) {
                        if let type = model.editor.room.roomType {
                            Text(type.capitalizedFirstLetter).font(.caption).foregroundStyle(.secondary)
                        }
                        Text("\(model.objects.count) things, \(named) named").font(.headline)
                            .lineLimit(1).minimumScaleFactor(0.7)
                    }
                    Spacer()
                    Button { model.cleanUp() } label: { Label("Clean up", systemImage: "wand.and.stars") }
                        .buttonStyle(.bordered)
                        .disabled(model.reviewing)
                    Button { model.review() } label: {
                        if model.reviewing { ProgressView().frame(width: 20) } else { Label("Review", systemImage: "sparkles") }
                    }
                    .buttonStyle(.bordered)
                    .disabled(model.reviewing)
                }
                if model.reviewing {
                    Text("The AI is reading the whole room…").font(.footnote).foregroundStyle(.secondary)
                }
                if let note = model.cleanUpNote {
                    Text(note).font(.footnote).foregroundStyle(.yellow)
                }
                if let note = model.reviewNote {
                    Text(note).font(.footnote).foregroundStyle(.yellow)
                    ForEach(model.reviewChanges.prefix(5), id: \.self) { line in
                        Text("· " + line).font(.caption2).foregroundStyle(.secondary).lineLimit(2)
                    }
                    if model.reviewChanges.count > 5 {
                        Text("· and \(model.reviewChanges.count - 5) more").font(.caption2).foregroundStyle(.secondary)
                    }
                }
                Text("Tap a box to select it, then drag to move it. Drag empty space to look around, pinch to zoom.")
                    .font(.footnote).foregroundStyle(.secondary)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding()
        .background(.ultraThinMaterial)
    }

    private func tool(_ title: String, _ icon: String) -> some View {
        VStack(spacing: 3) { Image(systemName: icon); Text(title).font(.caption2) }.frame(maxWidth: .infinity)
    }

    private func details(_ b: ObjectBox) -> String {
        let cm = b.size.map { Int(($0 * 100).rounded()) }
        let s = "\(cm[0]) × \(cm[2]) cm, \(cm[1]) cm high"
        switch b.source {
        case "user": return s + " · named by you"
        case "model": return s + " · named by AI, \(Int((b.confidence ?? 0) * 100))%"
        case "vision": return s + " · image guess, \(Int((b.confidence ?? 0) * 100))%"
        case "review": return s + " · named by room review"
        default: return s
        }
    }
}

private struct ShareItem: Identifiable {
    let id = UUID()
    let url: URL
}

private struct ActivityView: UIViewControllerRepresentable {
    let items: [Any]
    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: items, applicationActivities: nil)
    }
    func updateUIViewController(_ vc: UIActivityViewController, context: Context) {}
}

/// The room in 3D: orbit with one finger on empty space, drag the selected box with one finger.
private struct RoomSceneView: UIViewRepresentable {
    let model: RoomEditModel

    func makeCoordinator() -> Coordinator { Coordinator(model) }

    func makeUIView(context: Context) -> SCNView {
        let v = SCNView()
        v.scene = model.scene
        v.backgroundColor = UIColor(white: 0.08, alpha: 1)
        v.antialiasingMode = .multisampling4X
        v.allowsCameraControl = true
        v.delegate = context.coordinator

        let cam = SCNNode()
        cam.camera = SCNCamera()
        cam.camera?.zNear = 0.05
        cam.camera?.zFar = 100
        cam.camera?.fieldOfView = CGFloat(RoomEditModel.fieldOfView)
        cam.simdPosition = model.cameraStart
        cam.simdLook(at: model.center, up: SIMD3(0, 1, 0), localFront: SIMD3(0, 0, -1))
        model.scene.rootNode.addChildNode(cam)
        v.pointOfView = cam
        v.defaultCameraController.interactionMode = .orbitTurntable
        v.defaultCameraController.worldUp = SCNVector3(0, 1, 0)
        v.defaultCameraController.target = SCNVector3(model.center)
        v.defaultCameraController.minimumVerticalAngle = 5
        v.defaultCameraController.maximumVerticalAngle = 89

        let c = context.coordinator
        c.view = v
        let tap = UITapGestureRecognizer(target: c, action: #selector(Coordinator.onTap(_:)))
        let pan = UIPanGestureRecognizer(target: c, action: #selector(Coordinator.onPan(_:)))
        pan.maximumNumberOfTouches = 1
        pan.delegate = c
        c.pan = pan
        // The camera's own pan waits until ours has declined (it only starts on the selected box).
        for g in v.gestureRecognizers ?? [] where g is UIPanGestureRecognizer { g.require(toFail: pan) }
        v.addGestureRecognizer(tap)
        v.addGestureRecognizer(pan)
        return v
    }

    func updateUIView(_ v: SCNView, context: Context) {}

    @MainActor
    final class Coordinator: NSObject, UIGestureRecognizerDelegate, SCNSceneRendererDelegate {
        let model: RoomEditModel
        let culler: WallCuller
        weak var view: SCNView?
        weak var pan: UIPanGestureRecognizer?
        private var planeY: Float = 0
        private var offset = SIMD2<Float>(0, 0)

        init(_ model: RoomEditModel) { self.model = model; self.culler = model.culler }

        nonisolated func renderer(_ renderer: any SCNSceneRenderer, updateAtTime time: TimeInterval) {
            guard let pov = renderer.pointOfView else { return }
            culler.update(camera: pov.presentation.simdWorldPosition)
        }

        /// The box under a screen point, and where it was hit.
        private func boxHit(_ p: CGPoint) -> (id: Int, at: SIMD3<Float>)? {
            guard let v = view else { return nil }
            let hits = v.hitTest(p, options: [.categoryBitMask: 2, .searchMode: SCNHitTestSearchMode.closest.rawValue])
            for h in hits {
                var n: SCNNode? = h.node
                while let node = n {
                    if let name = node.name, name.hasPrefix("box-"), let id = Int(name.dropFirst(4)) {
                        return (id, h.simdWorldCoordinates)
                    }
                    n = node.parent
                }
            }
            return nil
        }

        /// Where a screen point's ray meets the horizontal plane at height `y`.
        private func point(_ p: CGPoint, y: Float) -> SIMD2<Float>? {
            guard let v = view else { return nil }
            let a = SIMD3<Float>(v.unprojectPoint(SCNVector3(Float(p.x), Float(p.y), 0)))
            let b = SIMD3<Float>(v.unprojectPoint(SCNVector3(Float(p.x), Float(p.y), 1)))
            let d = b - a
            guard abs(d.y) > 1e-5 else { return nil }
            let t = (y - a.y) / d.y
            guard t > 0 else { return nil }
            let q = a + d * t
            guard simd_distance(SIMD2(q.x, q.z), SIMD2(model.center.x, model.center.z)) < 30 else { return nil }
            return SIMD2(q.x, q.z)
        }

        @objc func onTap(_ g: UITapGestureRecognizer) {
            model.tapped(boxHit(g.location(in: view))?.id)
        }

        func gestureRecognizerShouldBegin(_ g: UIGestureRecognizer) -> Bool {
            guard g === pan else { return true }
            guard let sel = model.selected, let hit = boxHit(g.location(in: view)), hit.id == sel,
                  let b = model.selectedBox else { return false }
            planeY = hit.at.y
            offset = SIMD2(b.center[0], b.center[2]) - SIMD2(hit.at.x, hit.at.z)
            return true
        }

        @objc func onPan(_ g: UIPanGestureRecognizer) {
            switch g.state {
            case .began: model.beginMove()
            case .changed:
                if let p = point(g.location(in: view), y: planeY) { model.move(to: p + offset) }
            case .ended, .cancelled, .failed: model.endMove()
            default: break
            }
        }
    }
}

private extension String {
    var capitalizedFirstLetter: String { prefix(1).uppercased() + dropFirst() }
}
