import SwiftUI

@main
struct LidarScanApp: App {
    var body: some Scene {
        WindowGroup {
            ScanView()
                .preferredColorScheme(.dark)
                .task {
                    if NamingSelfTest.requested { await NamingSelfTest.run() }
                    if let id = BuildSelfTest.scanID { await BuildSelfTest.run(id) }
                }
        }
    }
}
