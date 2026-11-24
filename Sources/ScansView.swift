import ImageIO
import SceneKit
import SwiftUI

struct ScansRoute: Identifiable {
    let id = UUID()
    /// Scan to open straight away (after saving).
    let open: ScanBundle?
}

// MARK: list

struct ScansListView: View {
    @Environment(\.dismiss) private var dismiss
    @State private var scans: [ScanBundle] = []
    @State private var path: [ScanBundle]
    @State private var loaded = false
    let onContinue: (ScanBundle) -> Void

    init(open: ScanBundle?, onContinue: @escaping (ScanBundle) -> Void) {
        _path = State(initialValue: open.map { [$0] } ?? [])
        self.onContinue = onContinue
    }

    var body: some View {
        NavigationStack(path: $path) {
            List {
                ForEach(scans) { s in
                    NavigationLink(value: s) { ScanRow(scan: s) }
                }
                .onDelete { idx in
                    for i in idx { try? scans[i].delete() }
                    scans.remove(atOffsets: idx)
                }
            }
            .overlay {
                if loaded && scans.isEmpty {
                    ContentUnavailableView("No saved scans", systemImage: "cube.transparent",
                                           description: Text("Scan a room and tap Save scan."))
                }
            }
            .navigationTitle("Scans")
            .navigationDestination(for: ScanBundle.self) { ScanDetailView(scan: $0, onContinue: onContinue) }
            .toolbar { ToolbarItem(placement: .confirmationAction) { Button("Done") { dismiss() } } }
            .task {
                scans = await Task.detached { ScanBundle.list() }.value
                loaded = true
            }
        }
    }
}

private struct ScanRow: View {
    let scan: ScanBundle

    var body: some View {
        HStack(spacing: 12) {
            Thumbnail(url: scan.coverImageURL, maxPixel: 200)
                .frame(width: 54, height: 72).clipShape(RoundedRectangle(cornerRadius: 6))
            VStack(alignment: .leading, spacing: 3) {
                Text(scan.meta.created, format: .dateTime.day().month().hour().minute()).font(.headline)
                Text("\(scan.meta.keyframeCount) photos · \(scan.meta.triangleCount.formatted()) tris")
                    .font(.subheadline).foregroundStyle(.secondary)
                Text(sizeText).font(.caption).foregroundStyle(.secondary)
            }
        }
    }

    private var sizeText: String {
        let lo = scan.meta.boundsMin, hi = scan.meta.boundsMax
        guard lo.count == 3, hi.count == 3, hi[0] >= lo[0] else { return "" }
        return String(format: "%.1f × %.1f × %.1f m", hi[0] - lo[0], hi[2] - lo[2], hi[1] - lo[1])
    }
}

// MARK: detail

@MainActor
final class ScanViewerModel: ObservableObject {
    enum Layer: String, CaseIterable, Identifiable {
        case scan = "Scan", room = "Room", both = "Both"
        var id: String { rawValue }
    }

    let scan: ScanBundle
    let scene = SCNScene()
    @Published private(set) var isLoading = true
    @Published private(set) var error: String?
    @Published private(set) var keyframes: [Keyframe] = []
    @Published var selected: Int? { didSet { updateHighlight() } }
    @Published var showCameras = true { didSet { updateVisibility() } }
    @Published private(set) var room: RoomShell?
    @Published private(set) var isSnapping = false
    @Published private(set) var roomError: String?
    @Published var layer = Layer.scan { didSet { updateVisibility() } }
    @Published private(set) var objects: [ObjectBox] = []
    private var removedBoxes: [ObjectBox]?
    private var roomType: String?
    /// The whole saved room (objects, colors, naming state); kept in step with what's shown.
    private var saved: RoomModel?
    /// What the automatic room build is doing, for the panel; nil when idle.
    @Published private(set) var buildStep: String?
    /// Box finder version of the shown boxes (nil: from before versions were saved).
    @Published private(set) var finder: Int?
    private let boxes = BoxOverlay(labelHeight: 0.1)

    private let meshNode = SCNNode()
    private let camerasNode = SCNNode()
    private let highlightNode = SCNNode()
    private var roomNode = SCNNode()
    private var mesh: MeshOut?

    init(scan: ScanBundle) { self.scan = scan }

    var center: SIMD3<Float> {
        let lo = scan.meta.boundsMin, hi = scan.meta.boundsMax
        guard lo.count == 3, hi.count == 3, hi[0] >= lo[0] else { return .zero }
        return (SIMD3(lo[0], lo[1], lo[2]) + SIMD3(hi[0], hi[1], hi[2])) / 2
    }

    var radius: Float {
        let lo = scan.meta.boundsMin, hi = scan.meta.boundsMax
        guard lo.count == 3, hi.count == 3, hi[0] >= lo[0] else { return 1 }
        return max(0.3, simd_length(SIMD3(hi[0], hi[1], hi[2]) - SIMD3(lo[0], lo[1], lo[2])) / 2)
    }

    func load() async {
        guard isLoading else { return }
        let scan = self.scan
        do {
            let (mesh, geo, frames, saved) = try await Task.detached(priority: .userInitiated) {
                let mesh = try MeshExporter.readPLY(scan.meshURL)
                let frames = (try? scan.keyframes()) ?? []
                return (mesh, MeshGeometry.make(mesh, MeshGeometry.vertexColorMaterial()), frames, scan.loadRoom())
            }.value
            self.mesh = mesh
            meshNode.geometry = geo
            scene.rootNode.addChildNode(meshNode)
            if !frames.isEmpty {
                camerasNode.geometry = MeshGeometry.frustums(frames, depth: 0.08, color: .systemYellow)
                scene.rootNode.addChildNode(camerasNode)
            }
            scene.rootNode.addChildNode(highlightNode)
            keyframes = frames
            scene.rootNode.addChildNode(boxes.root)
            if let saved {
                removedBoxes = saved.removed; roomType = saved.roomType; finder = saved.finder
                self.saved = saved
                show(saved.shell, objects: saved.objects ?? [])
            }
        } catch {
            self.error = error.localizedDescription
        }
        isLoading = false
        if saved != nil { Task { await autoBuild() } }
    }

    /// Builds the room's 3D objects where missing, then names every box from its photos and
    /// builds again what changed. Runs by itself when a scan opens.
    func autoBuild() async {
        guard let mesh, let start = saved, buildStep == nil else { return }
        let scan = self.scan
        var room = start
        if (room.objects ?? []).contains(where: RoomObjects.isStale) {
            buildStep = "Building the objects…"
            let r = room
            room = await Task.detached(priority: .userInitiated) { RoomBuild.objects(r, meshes: [mesh], scan: scan) }.value
            guard saved == start else { return restart() }
            took(room)
        }
        if RoomBuild.needsNaming(room) {
            buildStep = "Naming things from the photos…"
            let before = room
            let named = await Task.detached(priority: .userInitiated) {
                await RoomBuild.nameFromPhotos(before, meshes: [mesh], scan: scan) { step in
                    Task { @MainActor in self.buildStep = step }
                }
            }.value
            // Something else changed the room meanwhile (Snap again, an edit): build that instead.
            guard saved == before else { return restart() }
            took(named)
        }
        buildStep = nil
    }

    private func restart() {
        buildStep = nil
        Task { await autoBuild() }
    }

    /// Shows a room the build or an edit produced.
    private func took(_ room: RoomModel) {
        saved = room
        removedBoxes = room.removed; roomType = room.roomType; finder = room.finder
        show(room.shell, objects: room.objects ?? [])
    }

    /// Fits floor, ceiling and walls to the mesh and saves them as room.json.
    func snapToRoom() async {
        guard let mesh, !isSnapping else { return }
        isSnapping = true; roomError = nil
        let scan = self.scan, previous = objects, removed = removedBoxes, roomType = self.roomType
        let result = await Task.detached(priority: .userInitiated) { () -> Result<(RoomModel, [ObjectBox], [ObjectBox]?), Error> in
            Result {
                let shell = try RoomFitter.fit([mesh])
                let found = ObjectFinder.find(RoomFitter.sample([mesh], cell: 0.03), shell: shell)
                var tracker = BoxTracker(seed: previous)          // keep labels from earlier
                let objects = Array(tracker.update(found).prefix(found.count))
                let (room, notes) = RoomCorrector.correct(RoomModel(shell: shell, objects: objects, removed: removed,
                                                                     roomType: roomType, finder: ObjectFinder.version))
                for n in notes { Debug.shared.log("clean up: " + n) }
                try? scan.saveRoom(room)
                return (room, room.objects ?? [], room.removed)
            }
        }.value
        switch result {
        case .success(let (room, objects, removed)):
            let shell = room.shell
            removedBoxes = removed
            finder = ObjectFinder.version
            saved = room
            Debug.shared.log("snap to room \(scan.meta.id): \(shell.walls.count) walls, \(objects.count) boxes, "
                             + String(format: "%.1f m², %.2f m high, ", shell.area, shell.height) + shell.notes)
            show(shell, objects: objects)
            isSnapping = false
            await autoBuild()
            return
        case .failure(let e):
            Debug.shared.log("snap to room FAILED: \(e)")
            roomError = e.localizedDescription
        }
        isSnapping = false
    }

    private func show(_ shell: RoomShell, objects: [ObjectBox]) {
        roomNode.removeFromParentNode()
        roomNode = RoomGeometry.node(shell)
        scene.rootNode.addChildNode(roomNode)
        room = shell
        self.objects = objects
        boxes.update(objects)
        layer = .both
    }

    func keyframe(_ id: Int) -> Keyframe? { keyframes.first { $0.id == id } }

    /// The fitted room with its boxes, for the Room screen.
    var roomModel: RoomModel? { saved }
    var scanGeometry: SCNGeometry? { meshNode.geometry }

    /// Shows a room edited on the Room screen (already saved there).
    func apply(_ edited: RoomModel) {
        saved = edited
        room = edited.shell
        objects = edited.objects ?? []
        removedBoxes = edited.removed
        roomType = edited.roomType
        boxes.update(objects)
    }

    private func updateVisibility() {
        meshNode.isHidden = layer == .room
        roomNode.isHidden = layer == .scan
        boxes.root.isHidden = layer == .scan
        camerasNode.isHidden = !showCameras || layer == .room
    }

    private func updateHighlight() {
        if let id = selected, let k = keyframe(id) {
            highlightNode.geometry = MeshGeometry.frustums([k], depth: 0.25, color: .systemRed)
        } else {
            highlightNode.geometry = nil
        }
    }
}

struct ScanDetailView: View {
    @StateObject private var model: ScanViewerModel
    @State private var editingRoom = false
    let onContinue: (ScanBundle) -> Void

    init(scan: ScanBundle, onContinue: @escaping (ScanBundle) -> Void) {
        _model = StateObject(wrappedValue: ScanViewerModel(scan: scan))
        self.onContinue = onContinue
    }

    var body: some View {
        ZStack(alignment: .bottom) {
            Color.black.ignoresSafeArea()
            if model.isLoading {
                ProgressView("Loading mesh…")
            } else if let error = model.error {
                ContentUnavailableView("Can't open scan", systemImage: "exclamationmark.triangle",
                                       description: Text(error))
            } else {
                MeshSceneView(scene: model.scene, center: model.center, radius: model.radius)
                    .ignoresSafeArea(edges: .bottom)
                    .overlay(alignment: .topTrailing) { selectedPhoto }
                VStack(spacing: 0) {
                    roomBar
                    if !model.keyframes.isEmpty { photoStrip }
                }
            }
        }
        .navigationTitle(Text(model.scan.meta.created, format: .dateTime.day().month().hour().minute()))
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItemGroup(placement: .primaryAction) {
                if !model.keyframes.isEmpty {
                    Button { model.showCameras.toggle() } label: {
                        Image(systemName: model.showCameras ? "camera.fill" : "camera")
                    }
                    .accessibilityLabel(model.showCameras ? "Hide camera positions" : "Show camera positions")
                }
                ShareLink(item: model.scan.meshURL)
                Button { onContinue(model.scan) } label: { Label("Continue scanning", systemImage: "record.circle") }
                    .disabled(!model.scan.canContinue)
            }
        }
        .task { await model.load() }
        .navigationDestination(isPresented: $editingRoom) {
            if let room = model.roomModel {
                RoomEditView(scan: model.scan, room: room, scanGeometry: model.scanGeometry) { model.apply($0) }
            }
        }
    }

    /// Snap to room, then the layer switch and the room's size.
    @ViewBuilder private var roomBar: some View {
        VStack(spacing: 8) {
            if let room = model.room {
                Picker("Show", selection: $model.layer) {
                    ForEach(ScanViewerModel.Layer.allCases) { Text($0.rawValue).tag($0) }
                }
                .pickerStyle(.segmented)
                HStack {
                    Text(roomSummary(room)).font(.footnote.weight(.medium)).monospacedDigit()
                    Spacer()
                    Button { Task { await model.snapToRoom() } } label: {
                        if model.isSnapping { ProgressView() } else { Image(systemName: "arrow.clockwise") }
                    }
                    .accessibilityLabel("Snap again")
                }
                Text(room.notes).font(.caption2).foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
                if let step = model.buildStep {
                    HStack(spacing: 8) {
                        ProgressView()
                        Text(step).font(.footnote).foregroundStyle(.secondary)
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                }
                if (model.finder ?? 0) < ObjectFinder.version {
                    Text("These boxes were found by an older version. Tap ↻ to find them again; names are kept.")
                        .font(.footnote).foregroundStyle(.yellow).frame(maxWidth: .infinity, alignment: .leading)
                }
                if room.looksLikeSeveralRooms {
                    Text("This scan looks like more than one room. The room model works best with one room per scan.")
                        .font(.footnote).foregroundStyle(.yellow).frame(maxWidth: .infinity, alignment: .leading)
                }
                Button { editingRoom = true } label: {
                    Label("Open room", systemImage: "cube.transparent")
                        .font(.headline).frame(maxWidth: .infinity).padding(.vertical, 4)
                }
                .buttonStyle(.borderedProminent)
            } else {
                Button { Task { await model.snapToRoom() } } label: {
                    HStack {
                        if model.isSnapping { ProgressView() } else { Image(systemName: "cube.transparent") }
                        Text(model.isSnapping ? "Snapping…" : "Snap to room")
                    }
                    .font(.headline).frame(maxWidth: .infinity).padding(.vertical, 6)
                }
                .buttonStyle(.borderedProminent)
                .disabled(model.isSnapping)
            }
            if let e = model.roomError {
                Text(e).font(.footnote).foregroundStyle(.orange).frame(maxWidth: .infinity, alignment: .leading)
            }
        }
        .padding(.horizontal).padding(.vertical, 10)
        .background(.ultraThinMaterial)
    }

    private func roomSummary(_ r: RoomShell) -> String {
        let sides = r.walls.count, guessed = r.walls.filter { $0.coverage < 0.5 }.count
        var s = String(format: "%.1f × %.1f m · %.2f m high%@ · %.1f m² · %d walls",
                       r.extent.max() ?? 0, r.extent.min() ?? 0, r.height, r.ceilingObserved ? "" : " (guess)",
                       r.area, sides)
        if guessed > 0 { s += " (\(guessed) guessed)" }
        let named = model.objects.filter { $0.label != nil }.count
        s += " · \(model.objects.count) things, \(named) named"
        return s
    }

    @ViewBuilder private var selectedPhoto: some View {
        if let id = model.selected, let k = model.keyframe(id) {
            Thumbnail(url: model.scan.imageURL(k), maxPixel: 800)
                .frame(width: 150, height: 200)
                .clipShape(RoundedRectangle(cornerRadius: 10))
                .overlay(RoundedRectangle(cornerRadius: 10).stroke(.red, lineWidth: 2))
                .padding(12)
                .onTapGesture { model.selected = nil }
        }
    }

    private var photoStrip: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            LazyHStack(spacing: 6) {
                ForEach(model.keyframes) { k in
                    Thumbnail(url: model.scan.imageURL(k), maxPixel: 160)
                        .frame(width: 54, height: 72)
                        .clipShape(RoundedRectangle(cornerRadius: 5))
                        .overlay(RoundedRectangle(cornerRadius: 5)
                            .stroke(model.selected == k.id ? Color.red : .clear, lineWidth: 2))
                        .onTapGesture { model.selected = model.selected == k.id ? nil : k.id }
                }
            }
            .padding(.horizontal).padding(.vertical, 8)
        }
        .frame(height: 88)
        .background(.ultraThinMaterial)
    }
}

/// Orbitable SceneKit view framed on the scan.
private struct MeshSceneView: UIViewRepresentable {
    let scene: SCNScene
    let center: SIMD3<Float>
    let radius: Float

    func makeUIView(context: Context) -> SCNView {
        let v = SCNView()
        v.scene = scene
        v.backgroundColor = .black
        v.antialiasingMode = .multisampling4X
        v.autoenablesDefaultLighting = true
        v.allowsCameraControl = true

        let cam = SCNNode()
        cam.camera = SCNCamera()
        cam.camera?.zNear = 0.01
        cam.camera?.zFar = Double(radius * 20)
        cam.simdPosition = center + SIMD3(radius * 0.9, radius * 1.2, radius * 0.9)
        cam.simdLook(at: center, up: SIMD3(0, 1, 0), localFront: SIMD3(0, 0, -1))
        scene.rootNode.addChildNode(cam)
        v.pointOfView = cam

        v.defaultCameraController.interactionMode = .orbitTurntable
        v.defaultCameraController.worldUp = SCNVector3(0, 1, 0)
        v.defaultCameraController.target = SCNVector3(center)
        return v
    }

    func updateUIView(_ v: SCNView, context: Context) {}
}

/// A keyframe photo shown upright (keyframes are stored in landscape sensor orientation).
struct Thumbnail: View {
    let url: URL?
    let maxPixel: Int
    @State private var image: UIImage?

    var body: some View {
        ZStack {
            Color.gray.opacity(0.25)
            if let image { Image(uiImage: image).resizable().scaledToFill() }
        }
        .task(id: url) { image = await Self.load(url, maxPixel: maxPixel) }
    }

    private static let cache = NSCache<NSString, UIImage>()

    static func load(_ url: URL?, maxPixel: Int) async -> UIImage? {
        guard let url else { return nil }
        let key = "\(url.path)#\(maxPixel)" as NSString
        if let hit = cache.object(forKey: key) { return hit }
        let image = await Task.detached(priority: .utility) { () -> UIImage? in
            guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
                  let cg = CGImageSourceCreateThumbnailAtIndex(src, 0, [
                      kCGImageSourceCreateThumbnailFromImageAlways: true,
                      kCGImageSourceThumbnailMaxPixelSize: maxPixel,
                  ] as CFDictionary) else { return nil }
            return UIImage(cgImage: cg, scale: 1, orientation: .right)
        }.value
        if let image { cache.setObject(image, forKey: key) }
        return image
    }
}
