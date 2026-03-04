import XCTest
import Msplat

final class MsplatTests: XCTestCase {

    static let gardenPath = "../datasets/mipnerf360/garden"

    func testConfigDefaults() {
        let config = TrainingConfig()
        XCTAssertEqual(config.iterations, 30_000)
        XCTAssertEqual(config.shDegree, 3)
        XCTAssertEqual(config.ssimWeight, 0.2, accuracy: 0.001)
    }

    func testLoadDataset() throws {
        let dataset = GaussianDataset(
            path: Self.gardenPath,
            downscaleFactor: 4.0,
            evalMode: true,
            testEvery: 8
        )
        XCTAssertGreaterThan(dataset.numTrain, 0)
        XCTAssertGreaterThan(dataset.numTest, 0)
    }

    func testTrainShort() throws {
        let dataset = GaussianDataset(
            path: Self.gardenPath,
            downscaleFactor: 4.0
        )
        var config = TrainingConfig()
        config.iterations = 10
        config.numDownscales = 0

        let trainer = GaussianTrainer(dataset: dataset, config: config)

        for _ in 0..<10 {
            let stats = trainer.step()
            XCTAssertGreaterThan(stats.splatCount, 0)
        }

        XCTAssertEqual(trainer.iteration, 10)
        XCTAssertGreaterThan(trainer.splatCount, 100_000)
    }

    func testRender() throws {
        let dataset = GaussianDataset(
            path: Self.gardenPath,
            downscaleFactor: 4.0
        )
        var config = TrainingConfig()
        config.iterations = 5
        config.numDownscales = 0

        let trainer = GaussianTrainer(dataset: dataset, config: config)
        for _ in 0..<5 { trainer.step() }

        let rendered = trainer.render(cameraIndex: 0)
        XCTAssertGreaterThan(rendered.width, 0)
        XCTAssertGreaterThan(rendered.height, 0)
        XCTAssertEqual(rendered.pixels.count, rendered.width * rendered.height * 3)
    }

    func testExportPly() throws {
        let dataset = GaussianDataset(
            path: Self.gardenPath,
            downscaleFactor: 4.0
        )
        var config = TrainingConfig()
        config.iterations = 5
        config.numDownscales = 0

        let trainer = GaussianTrainer(dataset: dataset, config: config)
        for _ in 0..<5 { trainer.step() }

        let tmpPath = NSTemporaryDirectory() + "msplat_test_export.ply"
        trainer.exportPly(to: tmpPath)
        XCTAssertTrue(FileManager.default.fileExists(atPath: tmpPath))

        let fileSize = try FileManager.default.attributesOfItem(atPath: tmpPath)[.size] as! Int
        XCTAssertGreaterThan(fileSize, 0)

        try FileManager.default.removeItem(atPath: tmpPath)
    }
}
