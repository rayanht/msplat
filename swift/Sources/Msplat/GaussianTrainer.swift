import MsplatCore
import Foundation

/// Trains a 3D Gaussian Splatting scene on a dataset.
public class GaussianTrainer {
    private let handle: MsplatTrainer

    /// Create a trainer.
    /// - Parameters:
    ///   - dataset: The loaded dataset. Must outlive the trainer.
    ///   - config: Training configuration.
    public init(dataset: GaussianDataset, config: TrainingConfig = TrainingConfig()) {
        handle = msplat_trainer_create(dataset.handle, config.toC())
    }

    deinit {
        msplat_trainer_destroy(handle)
        msplat_cleanup()
    }

    /// Run one training step.
    @discardableResult
    public func step() -> TrainingStats {
        TrainingStats(from: msplat_trainer_step(handle))
    }

    /// Train for all remaining iterations (blocking, no progress callbacks).
    /// For progress reporting, use `step()` in a loop instead.
    public func train() {
        msplat_trainer_train(handle)
    }

    /// Evaluate on held-out test views.
    public func evaluate() -> EvalMetrics {
        EvalMetrics(from: msplat_trainer_evaluate(handle))
    }

    /// Render a camera view as RGB float32 pixel data.
    public func render(cameraIndex: Int, useTest: Bool = false) -> PixelData {
        let buf = msplat_trainer_render(handle, Int32(cameraIndex), useTest)
        let count = Int(buf.width) * Int(buf.height) * 3
        let data = Array(UnsafeBufferPointer(start: buf.data, count: count))
        free(buf.data)
        return PixelData(pixels: data, width: Int(buf.width), height: Int(buf.height))
    }

    /// Render from an arbitrary camera-to-world pose (4x4 row-major, OpenGL convention).
    /// Uses intrinsics (focal length, resolution) from the given reference camera.
    public func renderFromPose(camToWorld: [Float], refCameraIndex: Int = 0) -> PixelData {
        precondition(camToWorld.count == 16)
        let buf = camToWorld.withUnsafeBufferPointer { ptr in
            msplat_trainer_render_pose(handle, ptr.baseAddress!, Int32(refCameraIndex))
        }
        let count = Int(buf.width) * Int(buf.height) * 3
        let data = Array(UnsafeBufferPointer(start: buf.data, count: count))
        free(buf.data)
        return PixelData(pixels: data, width: Int(buf.width), height: Int(buf.height))
    }

    /// Export scene as PLY.
    public func exportPly(to path: String) {
        msplat_trainer_export_ply(handle, path)
    }

    /// Export scene as .splat.
    public func exportSplat(to path: String) {
        msplat_trainer_export_splat(handle, path)
    }

    /// Save full training state for resume.
    public func saveCheckpoint(to path: String) {
        msplat_trainer_save_checkpoint(handle, path)
    }

    /// Load checkpoint and resume training. Returns the saved iteration.
    @discardableResult
    public func loadCheckpoint(from path: String) -> Int {
        Int(msplat_trainer_load_checkpoint(handle, path))
    }

    /// Current number of gaussians.
    public var splatCount: Int { Int(msplat_trainer_splat_count(handle)) }

    /// Current training iteration.
    public var iteration: Int { Int(msplat_trainer_iteration(handle)) }
}

/// RGB float32 pixel data from a render.
public struct PixelData {
    public let pixels: [Float]  // RGB, HWC layout
    public let width: Int
    public let height: Int
}

/// Synchronize the GPU (wait for all commands to complete).
public func msplatSync() {
    msplat_sync()
}

/// Release cached GPU resources. Called automatically when GaussianTrainer is deallocated.
/// Only needed if you want to free GPU memory early in a long-running process.
public func msplatCleanup() {
    msplat_cleanup()
}
