// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "core",
    platforms: [.macOS(.v15)],
    products: [
        .library(name: "MsplatCore", targets: ["MsplatCore"]),
    ],
    dependencies: [
        .package(url: "https://github.com/NewChromantics/nlohmann_json", branch: "graham/FixSwiftPackage")
    ],
    targets: [
        .target(
            name: "MsplatCore",
            dependencies: [
                .product(name: "json", package: "nlohmann_json")
            ],
            path: nil,
            exclude: ["include"],  // Exclude C++ headers from module
            sources: nil,
            resources: [
                //.copy("Resources/default.metallib")
            ],
            publicHeadersPath: "c_api",  // expose CAPI
            cxxSettings: [
                .define("_LIBCPP_ENABLE_CXX17_REMOVED_FEATURES", to: "1"),
                .headerSearchPath("include"),
                .headerSearchPath("metal"),
                .unsafeFlags(["-std=c++17", "-stdlib=libc++"]),
            ],
            linkerSettings: [
                .linkedFramework("Foundation"),
                .linkedFramework("Metal"),
                .linkedFramework("MetalKit"),
                .linkedFramework("MetalPerformanceShaders"),
                .linkedFramework("CoreGraphics"),
                .linkedFramework("ImageIO"),
                .unsafeFlags(["-lc++"]),
            ]
        ),
    ],
    cxxLanguageStandard: .cxx17
)
