public enum Config {
    public static let latentSize = 512
    
    // Level 1 generates 4x4 images.
    // Level 7 generates 256x256 images.
    public static let maxLevel = 7
    
    // minibatch size for each level
    public static let minibatchSizeSchedule = [128, 64, 64, 32, 32, 16, 16]
    
    public static let numImagesPerPhase = 800_000
    
    public static let imageDirectory = "./images"
    public static let tensorboardOutputDirectory = "./tensorboard"
    
    public static let numStepsToInfer = 10_000
    
    public static let debugPrint = true
}
