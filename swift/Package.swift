// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "Msplat",
    platforms: [.macOS(.v15)],
    products: [
        .library(name: "Msplat", targets: ["Msplat"]),
    ],
    dependencies: [
        .package(path: "../core"),
    ],
    targets: [

        // Swift API
        .target(
            name: "Msplat",
            dependencies: [
                .product(name: "MsplatCore", package: "core")
            ],
            path: "Sources/Msplat",
            resources: [.copy("Resources/default.metallib")],
            linkerSettings: [
                .linkedFramework("Metal"),
                .linkedFramework("MetalKit"),
                .linkedFramework("MetalPerformanceShaders"),
                .linkedFramework("Foundation"),
                .linkedFramework("ImageIO"),
                .linkedFramework("CoreGraphics"),
                .unsafeFlags(["-lc++"]),
            ]
        ),

        // Tests
        .testTarget(
            name: "MsplatTests",
            dependencies: ["Msplat"],
            path: "Tests"
        ),
    ]
)
