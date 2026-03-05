// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "DemoApp",
    platforms: [.macOS(.v15)],
    dependencies: [
        .package(path: "../swift"),
    ],
    targets: [
        .executableTarget(
            name: "DemoApp",
            dependencies: [
                .product(name: "Msplat", package: "swift"),
            ],
            path: "Sources"
        ),
    ]
)
