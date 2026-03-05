import MsplatCore
import Foundation

nonisolated(unsafe) private var _metallibConfigured = false

func ensureMetallibConfigured() {
    guard !_metallibConfigured else { return }
    _metallibConfigured = true
    if let path = Bundle.module.path(forResource: "default", ofType: "metallib") {
        msplat_set_metallib_path(path)
    }
}

/// A loaded dataset of camera views for training.
public class GaussianDataset {
    let handle: MsplatDataset

    /// Load a dataset from disk.
    /// - Parameters:
    ///   - path: Path to COLMAP, Nerfstudio, or other supported format.
    ///   - downscaleFactor: Image downscale factor (1.0 = full resolution).
    ///   - evalMode: If true, split cameras into train/test sets.
    ///   - testEvery: Hold out every Nth image for evaluation.
    public init(path: String, downscaleFactor: Float = 1.0,
                evalMode: Bool = false, testEvery: Int32 = 8) {
        ensureMetallibConfigured()
        handle = msplat_dataset_create(path, downscaleFactor, evalMode, testEvery)
    }

    deinit {
        msplat_dataset_destroy(handle)
    }

    /// Number of training cameras.
    public var numTrain: Int { Int(msplat_dataset_num_train(handle)) }

    /// Number of test cameras (0 if evalMode was false).
    public var numTest: Int { Int(msplat_dataset_num_test(handle)) }

    /// Get the camera-to-world pose (4x4 row-major, OpenGL convention) for a training camera.
    public func cameraPose(at index: Int) -> [Float] {
        var pose = [Float](repeating: 0, count: 16)
        pose.withUnsafeMutableBufferPointer { ptr in
            msplat_dataset_camera_pose(handle, Int32(index), ptr.baseAddress!)
        }
        return pose
    }
}
