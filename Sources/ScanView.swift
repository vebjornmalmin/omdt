import SwiftUI
import ARKit

struct ARSceneView: UIViewRepresentable {
    let view: ARSCNView
    func makeUIView(context: Context) -> ARSCNView { view }
    func updateUIView(_ uiView: ARSCNView, context: Context) {}
}

struct ScanView: View {
    @StateObject private var scan = ScanController()
    @State private var scansSheet: ScansRoute?
    @AppStorage("showDebugPanel") private var showDebug = true
    @StateObject private var location = LocationProvider()
    /// A saved scan made close to here that can be continued.
    @State private var nearby: (scan: ScanBundle, meters: Double)?

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            if scan.isSupported {
                ARSceneView(view: scan.view).ignoresSafeArea()
            } else {
                Text("This device has no LiDAR scanner.\nRun on iPhone 12 Pro or later (Pro models).")
                    .multilineTextAlignment(.center).foregroundStyle(.secondary).padding()
            }

            VStack {
                HStack(spacing: 14) {
                    Label(scan.trackingText, systemImage: "viewfinder")
                    Spacer()
                    Label("\(scan.keyframeCount)", systemImage: "camera")
                        .monospacedDigit()
                    Label("\(scan.objectCount)", systemImage: "cube")
                        .monospacedDigit()
                    Label(scan.detailText, systemImage: "scope")
                    Text("\(scan.triangleCount.formatted()) tris")
                        .monospacedDigit()
                }
                .font(.footnote.weight(.medium))
                .padding(10)
                .background(.ultraThinMaterial, in: Capsule())
                .padding(.horizontal)

                if let c = scan.continuing {
                    Text("Continuing scan from \(c.meta.created.formatted(.dateTime.day().month().hour().minute()))"
                         + (scan.isLiningUp ? " · lining up" : ""))
                        .font(.caption.weight(.medium))
                        .padding(.horizontal, 10).padding(.vertical, 5)
                        .background(scan.isLiningUp ? Color.orange.opacity(0.8) : Color.green.opacity(0.7), in: Capsule())
                } else if let n = nearby, !scan.isScanning, scan.triangleCount == 0 {
                    Button { scan.continueScan(n.scan) } label: {
                        Label("Continue the scan from \(n.scan.meta.created.formatted(.dateTime.day().month().hour().minute())) "
                              + "(\(Int(n.meters)) m away)", systemImage: "location.fill")
                            .font(.caption.weight(.medium))
                    }
                    .buttonStyle(.borderedProminent).controlSize(.small)
                }

                if showDebug { DebugPanel().padding(.horizontal).padding(.top, 4) }

                Spacer()

                HStack(spacing: 28) {
                    roundButton("arrow.counterclockwise") { scan.reset() }
                    Button { scan.toggleScanning() } label: {
                        ZStack {
                            Circle().strokeBorder(.white, lineWidth: 4).frame(width: 78, height: 78)
                            RoundedRectangle(cornerRadius: scan.isScanning ? 6 : 31)
                                .fill(.red)
                                .frame(width: scan.isScanning ? 30 : 62, height: scan.isScanning ? 30 : 62)
                        }
                    }
                    .disabled(!scan.isSupported)
                    roundButton(scan.showColors ? "paintpalette.fill" : "square.grid.3x3") {
                        scan.setShowColors(!scan.showColors)
                    }
                    roundButton(scan.showObjects ? "cube.transparent.fill" : "cube.transparent") {
                        scan.showObjects.toggle()
                    }
                }
                .padding(.bottom, 6)

                HStack(spacing: 10) {
                    Button { scan.save(location: location.current) } label: {
                        HStack {
                            if scan.isSaving { ProgressView() }
                            Text(scan.isSaving ? scan.saveStep : "Save scan").lineLimit(1).minimumScaleFactor(0.7)
                        }
                        .font(.headline).frame(maxWidth: .infinity).padding(.vertical, 12)
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(scan.triangleCount == 0 || scan.isSaving)

                    Button {
                        scansSheet = ScansRoute(open: nil)
                    } label: {
                        Image(systemName: "square.stack.3d.up").font(.headline).padding(.vertical, 12)
                    }
                    .buttonStyle(.bordered)
                    .accessibilityLabel("Saved scans")

                    Button { showDebug.toggle() } label: {
                        Image(systemName: showDebug ? "ladybug.fill" : "ladybug").font(.headline).padding(.vertical, 12)
                    }
                    .buttonStyle(.bordered)
                    .accessibilityLabel(showDebug ? "Hide debug info" : "Show debug info")
                }
                .padding(.horizontal).padding(.bottom, 8)
            }
        }
        .sheet(item: $scansSheet) { route in
            ScansListView(open: route.open) { picked in
                scansSheet = nil
                scan.continueScan(picked)
            }
        }
        .task(id: "\(location.current.map { "\(Int($0.latitude * 5000)),\(Int($0.longitude * 5000))" } ?? "-")\(scansSheet == nil)") {
            // Look for saved scans close by whenever the position moves ~20 m or the sheet closes.
            guard scansSheet == nil else { return }
            let all = await Task.detached { ScanBundle.list() }.value
            nearby = location.nearbyScans(all).first
        }
        // The camera stops while the scans sheet covers it and comes back live afterwards.
        .onChange(of: scansSheet?.id) { _, id in id == nil ? scan.resumeCamera() : scan.suspendCamera() }
        .onChange(of: scan.savedScan) { _, saved in
            if let saved { scansSheet = ScansRoute(open: saved); scan.savedScan = nil }
        }
        .alert("Error", isPresented: Binding(get: { scan.errorMessage != nil },
                                             set: { if !$0 { scan.errorMessage = nil } })) {
            Button("OK", role: .cancel) {}
        } message: { Text(scan.errorMessage ?? "") }
        .onAppear { UIApplication.shared.isIdleTimerDisabled = true }
        .task {
            scan.startCamera()
            location.start()
            // `devicectl device process launch ... com.vebjorn.LidarScan -autostart` starts scanning
            // without a tap, so a run can be observed from the Mac.
            if ProcessInfo.processInfo.arguments.contains("-autostart") {
                try? await Task.sleep(for: .seconds(1))
                if !scan.isScanning { scan.toggleScanning() }
            }
        }
    }

    private func roundButton(_ icon: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Image(systemName: icon).font(.title3).foregroundStyle(.white)
                .frame(width: 52, height: 52).background(.ultraThinMaterial, in: Circle())
        }
    }
}
