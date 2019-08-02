import Foundation

public enum Config {
    public static let latentSize = 512
    
    // Level 1 generates 4x4 images.
    // Level 7 generates 256x256 images.
    public static let maxLevel = 7
    
    // Can't use fused scale currently since TransposedConv2D doesn't work correctly.
    // https://github.com/tensorflow/swift-apis/pull/288
    public static let useFusedScale = false
    
    public static let loss = LSGANLoss()
    
    // minibatch size for each level
    public static let minibatchSizeSchedule = [128, 64, 64, 32, 32, 16, 16]
    
    public static let numImagesPerPhase = 100 //800_000
    
    public static let imageDirectory = URL(fileURLWithPath: "./images")
    public static let tensorboardOutputDirectory = URL(fileURLWithPath: "./tensorboard")
    
    public static let numStepsToInfer = 10_000
    
    public static let debugPrint = true
}
