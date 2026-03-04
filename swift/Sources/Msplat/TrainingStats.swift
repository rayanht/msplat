import MsplatCore

/// Statistics from a single training step.
public struct TrainingStats {
    public let iteration: Int
    public let splatCount: Int
    public let msPerStep: Float

    init(from c: MsplatStats) {
        self.iteration = Int(c.iteration)
        self.splatCount = Int(c.splatCount)
        self.msPerStep = c.msPerStep
    }
}

/// Evaluation metrics from held-out test views.
public struct EvalMetrics {
    public let psnr: Float
    public let ssim: Float
    public let l1: Float
    public let numTest: Int
    public let numGaussians: Int

    init(from c: MsplatEvalMetrics) {
        self.psnr = c.psnr
        self.ssim = c.ssim
        self.l1 = c.l1
        self.numTest = Int(c.numTest)
        self.numGaussians = Int(c.numGaussians)
    }
}
