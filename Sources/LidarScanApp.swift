import SwiftUI

@main
struct LidarScanApp: App {
    var body: some Scene {
        WindowGroup {
            ScanView()
                .preferredColorScheme(.dark)
        }
    }
}
