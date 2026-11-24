import CoreGraphics
import Foundation
import ImageIO
import simd

/// Picks the saved photos that show a box best, for showing it and for naming it again.
/// A photo counts when the box is in front of the camera, mostly in the picture (or filling a
/// good part of it, for big things seen up close), big enough, and not hidden behind something
/// nearer (from the photo's depth map). The best ones from different sides are kept.
enum BoxViews {
    struct View {
        var keyframe: Keyframe
        /// The box in the photo, in the photo's own (landscape sensor) pixels, clipped to it.
        var rect: CGRect
        var score: Float
    }

    /// Up to `count` views, best first, with camera directions at least `minAngle` apart.
    /// `depth` gives a keyframe's depth map (row-major, depthWidth × depthHeight) or nil.
    static func best(_ b: ObjectBox, keyframes: [Keyframe], count: Int = 3, minAngle: Float = 30 * .pi / 180,
                     depth: (Keyframe) -> [Float]?) -> [View] {
        var scored: [View] = []
        for k in keyframes {
            guard k.intrinsics.count == 4, k.camToWorld.count == 16, let v = project(b, k) else { continue }
            scored.append(v)
        }
        scored.sort { $0.score > $1.score }
        var out: [View] = []
        var dirs: [SIMD3<Float>] = []
        for v in scored where out.count < count {
            let cam = KeyframeSelector.position(v.keyframe.transform)
            let d = simd_normalize(cam - b.centerVec)
            guard dirs.allSatisfy({ acos(min(1, max(-1, simd_dot($0, d)))) >= minAngle }) else { continue }
            if let dm = depth(v.keyframe), occluded(b, v, dm) { continue }
            out.append(v); dirs.append(d)
        }
        return out
    }

    /// Where the box lands in a photo, with a score; nil when it doesn't show well.
    static func project(_ b: ObjectBox, _ k: Keyframe) -> View? {
        let toCam = k.transform.inverse
        let fx = k.intrinsics[0], fy = k.intrinsics[1], cx = k.intrinsics[2], cy = k.intrinsics[3]
        // The middle has to be in front; corners behind the camera (a bed seen up close) are
        // pulled up to just in front of it, which stretches the box across the picture.
        guard (toCam * SIMD4(b.centerVec, 1)).z < -0.3 else { return nil }
        var lo = CGPoint(x: CGFloat.infinity, y: .infinity), hi = CGPoint(x: -CGFloat.infinity, y: -.infinity)
        for p in b.corners {
            var c = toCam * SIMD4(p, 1)
            c.z = min(c.z, -0.15)
            let x = CGFloat(fx * c.x / -c.z + cx), y = CGFloat(fy * -c.y / -c.z + cy)
            lo = CGPoint(x: min(lo.x, x), y: min(lo.y, y)); hi = CGPoint(x: max(hi.x, x), y: max(hi.y, y))
        }
        let W = CGFloat(k.width), H = CGFloat(k.height)
        let full = CGRect(x: lo.x, y: lo.y, width: hi.x - lo.x, height: hi.y - lo.y)
        let clipped = full.intersection(CGRect(x: 0, y: 0, width: W, height: H))
        guard !clipped.isNull, min(clipped.width, clipped.height) >= 0.08 * min(W, H) else { return nil }
        let visible = Float(clipped.width * clipped.height / max(full.width * full.height, 1))
        let fills = Float(clipped.width * clipped.height / (W * H))
        guard visible >= 0.55 || fills >= 0.3 else { return nil }
        let size = min(1, Float(clipped.width * clipped.height / (0.12 * W * H)))
        let sharp = min(1, k.sharpness / 200)
        let score = max(visible, min(fills, 0.5)) * size * (0.7 + 0.3 * sharp)
        return View(keyframe: k, rect: clipped, score: score)
    }

    /// True when the depth in the middle of the box's picture is well in front of the box.
    static func occluded(_ b: ObjectBox, _ v: View, _ depth: [Float]) -> Bool {
        let k = v.keyframe
        guard depth.count == k.depthWidth * k.depthHeight, k.width > 0 else { return false }
        let s = CGFloat(k.depthWidth) / CGFloat(k.width)
        let x = min(k.depthWidth - 1, max(0, Int(v.rect.midX * s))), y = min(k.depthHeight - 1, max(0, Int(v.rect.midY * s)))
        let d = depth[y * k.depthWidth + x]
        let c = k.transform.inverse * SIMD4(b.centerVec, 1)
        let half = simd_length(SIMD3(b.size[0], b.size[1], b.size[2])) / 2
        return d > 0.05 && d < -c.z - half - 0.2
    }

    /// The view's crop from its photo, padded by 10 %, at most `maxSide` px, in sensor
    /// orientation (turn it right for portrait).
    static func crop(_ v: View, image url: URL, maxSide: CGFloat = 640) -> CGImage? {
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
              let img = CGImageSourceCreateImageAtIndex(src, 0, nil) else { return nil }
        // The photo may be stored smaller than the size the intrinsics are given at.
        let sx = CGFloat(img.width) / CGFloat(v.keyframe.width), sy = CGFloat(img.height) / CGFloat(v.keyframe.height)
        let r = v.rect.insetBy(dx: -v.rect.width * 0.1, dy: -v.rect.height * 0.1)
        let px = CGRect(x: r.minX * sx, y: r.minY * sy, width: r.width * sx, height: r.height * sy).integral
            .intersection(CGRect(x: 0, y: 0, width: img.width, height: img.height))
        guard !px.isNull, let cut = img.cropping(to: px) else { return nil }
        let scale = min(1, maxSide / max(px.width, px.height))
        guard scale < 1 else { return cut }
        let w = Int(px.width * scale), h = Int(px.height * scale)
        guard let ctx = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: 0,
                                  space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else { return cut }
        ctx.interpolationQuality = .high
        ctx.draw(cut, in: CGRect(x: 0, y: 0, width: w, height: h))
        return ctx.makeImage()
    }
}
