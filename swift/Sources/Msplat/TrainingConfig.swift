import MsplatCore

/// Configuration for Gaussian splatting training.
public struct TrainingConfig {
    public var iterations: Int32 = 30_000
    public var shDegree: Int32 = 3
    public var shDegreeInterval: Int32 = 1_000
    public var ssimWeight: Float = 0.2
    public var numDownscales: Int32 = 2
    public var resolutionSchedule: Int32 = 3_000
    public var refineEvery: Int32 = 100
    public var warmupLength: Int32 = 500
    public var resetAlphaEvery: Int32 = 30
    public var densifyGradThresh: Float = 0.0002
    public var densifySizeThresh: Float = 0.01
    public var stopScreenSizeAt: Int32 = 4_000
    public var splitScreenSize: Float = 0.05
    public var keepCrs: Bool = false
    public var downscaleFactor: Float = 1.0
    /// Background color as (R, G, B) in [0, 1]. Default magenta — high contrast
    /// against typical scenes, makes under-reconstructed regions obvious.
    public var bgColor: (Float, Float, Float) = (0.6130, 0.0101, 0.3984)

    public init() {}

    func toC() -> MsplatConfig {
        var c = msplat_default_config()
        c.iterations = iterations
        c.shDegree = shDegree
        c.shDegreeInterval = shDegreeInterval
        c.ssimWeight = ssimWeight
        c.numDownscales = numDownscales
        c.resolutionSchedule = resolutionSchedule
        c.refineEvery = refineEvery
        c.warmupLength = warmupLength
        c.resetAlphaEvery = resetAlphaEvery
        c.densifyGradThresh = densifyGradThresh
        c.densifySizeThresh = densifySizeThresh
        c.stopScreenSizeAt = stopScreenSizeAt
        c.splitScreenSize = splitScreenSize
        c.keepCrs = keepCrs
        c.downscaleFactor = downscaleFactor
        c.bgColor = (bgColor.0, bgColor.1, bgColor.2)
        return c
    }
}
