import SwiftUI

/// Live diagnostics overlay for development. Copy puts the stats and full log on the
/// clipboard; Share sends Documents/debug.log.
struct DebugPanel: View {
    @State private var copied = false
    private let debug = Debug.shared

    var body: some View {
        TimelineView(.periodic(from: .now, by: 0.25)) { _ in
            VStack(alignment: .leading, spacing: 6) {
                Text(statsText).foregroundStyle(.white)
                Divider().overlay(.white.opacity(0.3))
                Text(debug.recentLines(7).map(Self.trim).joined(separator: "\n"))
                    .foregroundStyle(.white.opacity(0.75))
                    .lineLimit(7)
                HStack(spacing: 8) {
                    Button(copied ? "Copied" : "Copy log") {
                        UIPasteboard.general.string = statsText + "\n\n" + debug.allLines()
                        copied = true
                        Task { try? await Task.sleep(for: .seconds(1.5)); copied = false }
                    }
                    ShareLink("Share log", item: debug.logURL)
                    Spacer()
                }
                .font(.caption2.weight(.semibold))
                .buttonStyle(.bordered)
                .controlSize(.mini)
            }
        }
        .font(.system(size: 10, design: .monospaced))
        .padding(8)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.black.opacity(0.65), in: RoundedRectangle(cornerRadius: 10))
    }

    private var statsText: String {
        let s = debug.snapshot()
        let since = Int(Date().timeIntervalSince(s.trackingSince))
        let lastFrame = s.lastFrameAt.map { String(format: "%.1f s ago", Date().timeIntervalSince($0)) } ?? "never"
        return """
        tracking  \(s.tracking) (\(since) s) · map \(s.mapping)
        frames    \(s.frames) · depth \(s.framesWithDepth) · \(s.fps) fps · last \(lastFrame)
        camera    \(s.imageSize) · light \(s.brightness)/255 · exp \(s.exposure)
        format    \(s.videoFormat) · depth \(s.depthSize)
        fusion    \(s.passes) passes · \(Int(s.passMs)) ms (avg \(Int(s.passMsAvg))) · skips \(s.skippedBusy) · \(s.blocks) blocks
        photos    \(s.keyframes) · \(Int(s.keyframeMs)) ms · \(s.keyframeReason)
        device    \(Int(Debug.memoryMB())) MB · \(Debug.thermal()) · int \(s.interruptions) · err \(s.errors) · \(Debug.buildDate)
        """
    }

    /// Drops the long timestamp prefix to fit the narrow panel.
    private static func trim(_ line: String) -> String {
        line.trimmingCharacters(in: .whitespaces)
    }
}
