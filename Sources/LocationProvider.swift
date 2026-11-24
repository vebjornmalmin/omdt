import CoreLocation

/// Rough GPS position, saved with each scan and used to offer continuing a scan made nearby.
/// Indoors it is only good to 5-50 m: enough to pick the right scan, not to line it up
/// (the ARKit world map does that).
final class LocationProvider: NSObject, ObservableObject, CLLocationManagerDelegate {
    @Published private(set) var current: GeoPoint?
    private let manager = CLLocationManager()

    override init() {
        super.init()
        manager.delegate = self
        manager.desiredAccuracy = kCLLocationAccuracyBest
        manager.distanceFilter = 5
    }

    func start() {
        switch manager.authorizationStatus {
        case .notDetermined: manager.requestWhenInUseAuthorization()
        case .authorizedWhenInUse, .authorizedAlways: manager.startUpdatingLocation()
        default: Debug.shared.log("location: not allowed")
        }
    }

    func stop() { manager.stopUpdatingLocation() }

    func locationManagerDidChangeAuthorization(_ m: CLLocationManager) {
        if m.authorizationStatus == .authorizedWhenInUse || m.authorizationStatus == .authorizedAlways {
            m.startUpdatingLocation()
        }
    }

    func locationManager(_ m: CLLocationManager, didUpdateLocations locations: [CLLocation]) {
        guard let l = locations.last, l.horizontalAccuracy >= 0 else { return }
        let first = current == nil
        current = GeoPoint(latitude: l.coordinate.latitude, longitude: l.coordinate.longitude, accuracy: l.horizontalAccuracy)
        if first { Debug.shared.log(String(format: "location: ±%.0f m", l.horizontalAccuracy)) }
    }

    func locationManager(_ m: CLLocationManager, didFailWithError error: Error) {
        Debug.shared.log("location failed: \(error.localizedDescription)")
    }

    /// Saved scans that can be continued and were made within reach of here, nearest first.
    func nearbyScans(_ scans: [ScanBundle]) -> [(scan: ScanBundle, meters: Double)] {
        guard let here = current else { return [] }
        return scans.compactMap { s -> (ScanBundle, Double)? in
            guard let loc = s.meta.location, s.canContinue else { return nil }
            let d = here.distance(to: loc)
            return d <= max(30, 1.5 * (here.accuracy + loc.accuracy)) ? (s, d) : nil
        }
        .sorted { $0.1 < $1.1 }
    }
}
