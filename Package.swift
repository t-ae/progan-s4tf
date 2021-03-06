// swift-tools-version:5.1
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "ProGAN",
    dependencies: [
        .package(url: "https://github.com/t-ae/gan-utils-s4tf.git", from: "0.1.3"),
        .package(url: "https://github.com/t-ae/image-loader.git", from: "0.1.8"),
        .package(url: "https://github.com/t-ae/tensorboardx-s4tf.git", from: "0.0.11"),
    ],
    targets: [
        // Targets are the basic building blocks of a package. A target can define a module or a test suite.
        // Targets can depend on other targets in this package, and on products in packages which this package depends on.
        .target(
            name: "ProGANModel",
            dependencies: ["GANUtils"]),
        .target(name: "train", dependencies: ["ProGANModel", "ImageLoader", "TensorBoardX"]),
        .testTarget(
            name: "ProGANModelTests",
            dependencies: ["ProGANModel"]),
    ]
)
