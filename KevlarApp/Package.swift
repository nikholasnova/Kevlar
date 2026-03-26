// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "KevlarApp",
    platforms: [.macOS(.v14)],
    targets: [
        .executableTarget(
            name: "KevlarApp",
            path: "Sources"
        )
    ]
)
