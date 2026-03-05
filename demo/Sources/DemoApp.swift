import SwiftUI
import Msplat
import MsplatCore
import AppKit
import QuartzCore
import Accelerate

// MARK: - Pixel conversion

/// Reusable RGBA buffer to avoid per-frame allocation
final class RGBABuffer {
    var bytes: UnsafeMutablePointer<UInt8>
    var capacity: Int

    init() { bytes = .allocate(capacity: 0); capacity = 0 }
    deinit { bytes.deallocate() }

    func ensure(_ count: Int) {
        guard count > capacity else { return }
        bytes.deallocate()
        bytes = .allocate(capacity: count)
        capacity = count
    }
}

nonisolated(unsafe) let rgbaBuffer = RGBABuffer()

/// Build CGImage directly from C pixel buffer. Reuses a persistent RGBA buffer.
func pixelBufferToCGImage(_ buf: MsplatPixelBuffer) -> CGImage {
    let w = Int(buf.width), h = Int(buf.height)
    let n = w * h
    let src = buf.data!
    let needed = n * 4
    rgbaBuffer.ensure(needed)
    let dst = rgbaBuffer.bytes

    // Single-pass RGB float → RGBA uint8
    for i in 0..<n {
        let r = src[i * 3], g = src[i * 3 + 1], b = src[i * 3 + 2]
        dst[i * 4]     = UInt8(min(max(r, 0), 1) * 255)
        dst[i * 4 + 1] = UInt8(min(max(g, 0), 1) * 255)
        dst[i * 4 + 2] = UInt8(min(max(b, 0), 1) * 255)
        dst[i * 4 + 3] = 255
    }

    let provider = CGDataProvider(dataInfo: nil, data: dst, size: needed,
                                   releaseData: { _, _, _ in })!
    return CGImage(width: w, height: h, bitsPerComponent: 8, bitsPerPixel: 32,
                   bytesPerRow: w * 4, space: CGColorSpaceCreateDeviceRGB(),
                   bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
                   provider: provider, decode: nil, shouldInterpolate: false,
                   intent: .defaultIntent)!
}

func pixelBufferToNSImage(_ buf: MsplatPixelBuffer) -> NSImage {
    let cg = pixelBufferToCGImage(buf)
    return NSImage(cgImage: cg, size: NSSize(width: cg.width, height: cg.height))
}

// MARK: - Vector helpers

func normalize(_ v: (Float, Float, Float)) -> (Float, Float, Float) {
    let len = sqrt(v.0*v.0 + v.1*v.1 + v.2*v.2)
    guard len > 1e-8 else { return (0, 1, 0) }
    return (v.0/len, v.1/len, v.2/len)
}

func cross(_ a: (Float, Float, Float), _ b: (Float, Float, Float)) -> (Float, Float, Float) {
    (a.1*b.2 - a.2*b.1, a.2*b.0 - a.0*b.2, a.0*b.1 - a.1*b.0)
}

// MARK: - Orbit camera

/// Build a cam-to-world matrix (4x4 row-major, OpenGL: Y-up, Z-back)
/// looking from `eye` toward `target` with a given world-up hint.
func lookAtCamToWorld(eye: (Float, Float, Float), target: (Float, Float, Float),
                      up: (Float, Float, Float)) -> [Float] {
    // Forward = normalize(eye - target)  (camera looks along -Z in OpenGL)
    let f = normalize((eye.0 - target.0, eye.1 - target.1, eye.2 - target.2))
    // Right = normalize(up × forward)
    let r = normalize(cross(up, f))
    // True up = forward × right
    let u = cross(f, r)

    // Row-major cam-to-world: columns are right, up, forward; last column is eye position
    return [
        r.0, u.0, f.0, eye.0,
        r.1, u.1, f.1, eye.1,
        r.2, u.2, f.2, eye.2,
          0,   0,   0,     1,
    ]
}

struct OrbitParams {
    var lookAt: (Float, Float, Float)    // where the camera looks (scene focus)
    var eyeCenter: (Float, Float, Float) // center of the orbit circle (above lookAt)
    var radius: Float
    var up: (Float, Float, Float)        // scene up direction (from camera average)
    var tangent1: (Float, Float, Float)  // ground-plane basis vector 1
    var tangent2: (Float, Float, Float)  // ground-plane basis vector 2
}

/// Compute orbit parameters from dataset camera poses.
/// Derives the ground plane from the average camera up-vector (column 1 of camToWorld).
/// The orbit center is where cameras are looking, not where they are.
func computeOrbitParams(_ poses: [[Float]]) -> OrbitParams {
    let n = Float(poses.count)

    // Camera positions and up vectors
    var cx: Float = 0, cy: Float = 0, cz: Float = 0
    var ux: Float = 0, uy: Float = 0, uz: Float = 0

    for p in poses {
        cx += p[3]; cy += p[7]; cz += p[11]
        ux += p[1]; uy += p[5]; uz += p[9]
    }
    cx /= n; cy /= n; cz /= n
    let up = normalize((ux, uy, uz))

    // Orbit radius = average distance from centroid projected onto ground plane
    var totalR: Float = 0
    for p in poses {
        let dx = p[3] - cx, dy = p[7] - cy, dz = p[11] - cz
        let dot = dx*up.0 + dy*up.1 + dz*up.2
        let gx = dx - dot*up.0, gy = dy - dot*up.1, gz = dz - dot*up.2
        totalR += sqrt(gx*gx + gy*gy + gz*gz)
    }
    let radius = totalR / n * 1.3

    // Look-at target: below camera centroid (where the scene centerpiece is)
    let lookAt = (
        cx - up.0 * radius * 0.3,
        cy - up.1 * radius * 0.3,
        cz - up.2 * radius * 0.3
    )
    // Eye orbits slightly above the camera centroid so it looks down at the scene
    let eyeCenter = (
        cx - up.0 * radius * 0.07,
        cy - up.1 * radius * 0.07,
        cz - up.2 * radius * 0.07
    )

    // Build two tangent vectors spanning the ground plane
    let absX = abs(up.0), absY = abs(up.1), absZ = abs(up.2)
    let seed: (Float, Float, Float)
    if absX <= absY && absX <= absZ { seed = (1, 0, 0) }
    else if absY <= absZ { seed = (0, 1, 0) }
    else { seed = (0, 0, 1) }

    let t1 = normalize(cross(up, seed))
    let t2 = cross(up, t1)

    return OrbitParams(lookAt: lookAt, eyeCenter: eyeCenter, radius: radius,
                       up: up, tangent1: t1, tangent2: t2)
}

// MARK: - Engine

@MainActor
final class Engine: ObservableObject {
    @Published var image: NSImage?
    @Published var iteration: Int = 0
    @Published var totalIterations: Int = 2_000
    @Published var splatCount: Int = 0
    @Published var msPerStep: Float = 0
    @Published var fps: Float = 0
    @Published var phase: Phase = .countdown(5)
    @Published var countdown: Int = 5

    enum Phase: Equatable {
        case countdown(Int), loading, training, orbiting
    }

    func start(datasetPath: String) {
        phase = .countdown(5)
        countdown = 5

        // 5-second countdown on main thread
        Task { @MainActor in
            for s in stride(from: 4, through: 0, by: -1) {
                try? await Task.sleep(for: .seconds(1))
                self.countdown = s
                self.phase = .countdown(s)
            }
            self.phase = .loading
            self.beginTraining(datasetPath: datasetPath)
        }
    }

    private func beginTraining(datasetPath: String) {
        Thread.detachNewThread { [weak self] in
            guard let self else { return }

            // Use C API directly to avoid extra copies in the hot path
            let ds = msplat_dataset_create(datasetPath, 1.0, false, 8)!
            let numCameras = Int(msplat_dataset_num_train(ds))

            var config = msplat_default_config()
            config.iterations = 2_000
            config.numDownscales = 0
            config.bgColor = (0, 0, 0)
            let trainer = msplat_trainer_create(ds, config)!

            DispatchQueue.main.async { self.phase = .training }

            // Phase 1: Training
            var batchStart = CACurrentMediaTime()
            var batchSteps = 0

            for i in 0..<2_000 {
                let stats = msplat_trainer_step(trainer)
                batchSteps += 1

                if i % 25 == 0 || i == 1_999 {
                    let batchEnd = CACurrentMediaTime()
                    let avgMs = Float((batchEnd - batchStart) / Double(batchSteps) * 1000.0)

                    let buf = msplat_trainer_render(trainer, 0, false)
                    let img = pixelBufferToNSImage(buf)
                    free(buf.data)
                    let iter = stats.iteration
                    let count = stats.splatCount
                    DispatchQueue.main.async {
                        self.image = img
                        self.iteration = Int(iter)
                        self.splatCount = Int(count)
                        self.msPerStep = avgMs
                    }

                    batchStart = CACurrentMediaTime()
                    batchSteps = 0
                }
            }

            let finalCount = Int(msplat_trainer_splat_count(trainer))
            DispatchQueue.main.async {
                self.splatCount = finalCount
                self.phase = .orbiting
            }

            // Phase 2: Smooth circular orbit — use C API for zero-copy render
            var poses = [[Float]]()
            for i in 0..<numCameras {
                var pose = [Float](repeating: 0, count: 16)
                msplat_dataset_camera_pose(ds, Int32(i), &pose)
                poses.append(pose)
            }
            let orbit = computeOrbitParams(poses)
            var frameCount = 0
            var fpsTimer = CACurrentMediaTime()

            // Query dimensions once to pre-allocate RGBA buffer
            var imgW: Int32 = 0, imgH: Int32 = 0
            var firstPose = lookAtCamToWorld(eye: orbit.eyeCenter, target: orbit.lookAt, up: orbit.up)
            msplat_trainer_render_pose_to_buffer(trainer, &firstPose, 0, nil, &imgW, &imgH)
            let rgbaBuf = UnsafeMutablePointer<UInt8>.allocate(capacity: Int(imgW * imgH) * 4)
            // Pre-fill alpha channel
            for i in 0..<Int(imgW * imgH) { rgbaBuf[i * 4 + 3] = 255 }

            let colorSpace = CGColorSpaceCreateDeviceRGB()
            let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)

            let orbitPeriod: Double = 7.5  // seconds per full revolution
            let orbitStart = CACurrentMediaTime()

            while true {
                    let elapsed = CACurrentMediaTime() - orbitStart
                    let angle = Float(elapsed / orbitPeriod) * 2.0 * .pi
                    let cosA = cos(angle), sinA = sin(angle)
                    let t1 = orbit.tangent1, t2 = orbit.tangent2
                    let eye = (
                        orbit.eyeCenter.0 + orbit.radius * (cosA * t1.0 + sinA * t2.0),
                        orbit.eyeCenter.1 + orbit.radius * (cosA * t1.1 + sinA * t2.1),
                        orbit.eyeCenter.2 + orbit.radius * (cosA * t1.2 + sinA * t2.2)
                    )

                    var pose = lookAtCamToWorld(eye: eye, target: orbit.lookAt, up: orbit.up)
                    msplat_trainer_render_pose_to_buffer(trainer, &pose, 0, rgbaBuf, &imgW, &imgH)

                    let provider = CGDataProvider(dataInfo: nil, data: rgbaBuf,
                                                   size: Int(imgW * imgH) * 4,
                                                   releaseData: { _, _, _ in })!
                    let cgImg = CGImage(width: Int(imgW), height: Int(imgH),
                                        bitsPerComponent: 8, bitsPerPixel: 32,
                                        bytesPerRow: Int(imgW) * 4, space: colorSpace,
                                        bitmapInfo: bitmapInfo, provider: provider,
                                        decode: nil, shouldInterpolate: false,
                                        intent: .defaultIntent)!
                    let img = NSImage(cgImage: cgImg, size: NSSize(width: Int(imgW), height: Int(imgH)))
                    frameCount += 1

                    var currentFps: Float = 0
                    let now = CACurrentMediaTime()
                    let dt = now - fpsTimer
                    if dt >= 0.5 {
                        currentFps = Float(frameCount) / Float(dt)
                        frameCount = 0
                        fpsTimer = now
                    }

                    let fpsVal = currentFps
                    DispatchQueue.main.async {
                        self.image = img
                        if fpsVal > 0 { self.fps = fpsVal }
                    }
            }
        }
    }
}

// MARK: - UI

struct ContentView: View {
    @StateObject private var engine = Engine()
    let datasetPath: String

    var body: some View {
        ZStack {
            Color.black

            if let img = engine.image {
                Image(nsImage: img)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
            }

            if case .countdown(let s) = engine.phase {
                Text("\(s)")
                    .font(.system(size: 120, weight: .bold, design: .monospaced))
                    .foregroundStyle(.white.opacity(0.8))
            }

            if engine.phase == .loading {
                VStack {
                    ProgressView()
                        .scaleEffect(1.5)
                        .tint(.white)
                    Text("Loading dataset...")
                        .foregroundStyle(.white)
                        .padding(.top, 8)
                }
            }

            if engine.phase == .training || engine.phase == .orbiting {
                VStack(spacing: 0) {
                    Spacer()

                    // Full-width progress bar during training
                    if engine.phase == .training {
                        progressBar
                    }

                    // Full-width stats bar
                    statsBar
                }
            }

            // Prominent FPS counter top-right during orbit
            if engine.phase == .orbiting {
                VStack {
                    HStack {
                        Spacer()
                        Text(String(format: "%.0f FPS", engine.fps))
                            .font(.system(size: 36, weight: .bold, design: .monospaced))
                            .foregroundStyle(.white)
                            .padding(.horizontal, 20)
                            .padding(.vertical, 10)
                            .background(.black.opacity(0.7))
                            .cornerRadius(12)
                            .padding(20)
                    }
                    Spacer()
                }
            }
        }
        .onAppear {
            engine.start(datasetPath: datasetPath)
        }
    }

    private var progressBar: some View {
        let progress = Double(engine.iteration) / Double(engine.totalIterations)
        return GeometryReader { geo in
            ZStack(alignment: .leading) {
                Rectangle()
                    .fill(.white.opacity(0.15))
                Rectangle()
                    .fill(
                        LinearGradient(
                            colors: [Color(red: 0.2, green: 0.5, blue: 1.0),
                                     Color(red: 0.0, green: 0.9, blue: 0.7)],
                            startPoint: .leading, endPoint: .trailing
                        )
                    )
                    .frame(width: geo.size.width * progress)
                    .animation(.linear(duration: 0.1), value: progress)
            }
        }
        .frame(height: 6)
    }

    private var statsBar: some View {
        HStack(spacing: 32) {
            switch engine.phase {
            case .countdown, .loading:
                EmptyView()
            case .training:
                Text("step \(engine.iteration) / \(engine.totalIterations)")
                Text("\(fmtCount(engine.splatCount)) splats")
                Text(String(format: "%.1f ms/step", engine.msPerStep))
            case .orbiting:
                Text("\(fmtCount(engine.splatCount)) splats")
                Text(String(format: "%.0f fps", engine.fps))
            }
        }
        .font(.system(size: 18, weight: .semibold, design: .monospaced))
        .foregroundStyle(.white)
        .frame(maxWidth: .infinity)
        .padding(.vertical, 14)
        .background(.black.opacity(0.85))
    }

    private func fmtCount(_ n: Int) -> String {
        if n >= 1_000_000 { return String(format: "%.2fM", Double(n) / 1_000_000) }
        if n >= 1_000 { return String(format: "%.0fK", Double(n) / 1_000) }
        return "\(n)"
    }
}

// MARK: - App entry

class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular)
        NSApp.activate(ignoringOtherApps: true)
    }
}

@main
struct DemoApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) var delegate
    let datasetPath: String

    init() {
        let args = CommandLine.arguments
        datasetPath = args.count > 1 ? args[1] : "datasets/mipnerf360/garden"
    }

    var body: some Scene {
        WindowGroup {
            ContentView(datasetPath: datasetPath)
                .frame(minWidth: 960, minHeight: 640)
        }
        .defaultSize(width: 1280, height: 720)
    }
}
