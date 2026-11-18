import SwiftUI
import ARKit

struct ARSceneView: UIViewRepresentable {
    let view: ARSCNView
    func makeUIView(context: Context) -> ARSCNView { view }
    func updateUIView(_ uiView: ARSCNView, context: Context) {}
}

struct ShareSheet: UIViewControllerRepresentable {
    let urls: [URL]
    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: urls, applicationActivities: nil)
    }
    func updateUIViewController(_ vc: UIActivityViewController, context: Context) {}
}

struct ScanView: View {
    @StateObject private var scan = ScanController()

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
                    Label(scan.detailText, systemImage: "scope")
                    Text("\(scan.triangleCount.formatted()) tris")
                        .monospacedDigit()
                }
                .font(.footnote.weight(.medium))
                .padding(10)
                .background(.ultraThinMaterial, in: Capsule())
                .padding(.horizontal)

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
                }
                .padding(.bottom, 6)

                Button { scan.export() } label: {
                    HStack {
                        if scan.isExporting { ProgressView() }
                        Text(scan.isExporting ? "Exporting…" : "Export PLY")
                    }
                    .font(.headline).frame(maxWidth: .infinity).padding(.vertical, 12)
                }
                .buttonStyle(.borderedProminent)
                .disabled(scan.triangleCount == 0 || scan.isExporting)
                .padding(.horizontal).padding(.bottom, 8)
            }
        }
        .sheet(item: $scan.shareItem) { ShareSheet(urls: $0.urls) }
        .alert("Error", isPresented: Binding(get: { scan.errorMessage != nil },
                                             set: { if !$0 { scan.errorMessage = nil } })) {
            Button("OK", role: .cancel) {}
        } message: { Text(scan.errorMessage ?? "") }
        .onAppear { UIApplication.shared.isIdleTimerDisabled = true }
    }

    private func roundButton(_ icon: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Image(systemName: icon).font(.title3).foregroundStyle(.white)
                .frame(width: 52, height: 52).background(.ultraThinMaterial, in: Circle())
        }
    }
}
