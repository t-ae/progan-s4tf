public enum Config {
    public static let latentSize = 512
    
    // Level 1 generates 8x8 images.
    // Level 6 generates 256x256 images.
    public static let initialLevel = 1
    public static let maxLevel = 6
    
    // minibatch size for each level
    public static let minibatchSizeSchedule = [128, 64, 64, 32, 32, 16]
    
    public static let numImagesPerPhase = 80_000
    
    public static let imageDirectory = "./images"
    public static let imageOutputDirectory = "./output"
    
    public static let numStepsToInfer = 10_000
    
    public static let debugPrint = true
}
