import Foundation

public struct Config: Codable {
    // MARK: Model settings
    public var latentSize: Int
    
    public var normalizeLatent: Bool
    
    public var loss: GANLossType
    
    // MARK: Training settings
    public var learningRates: GDPair<Float>
    
    public var startSize: ImageSize
    public var endSize: ImageSize
    public var batchSizes: [ImageSize: Int]
    
    public var imagesPerPhase: Int
    
    public init(
        latentSize: Int,
        normalizeLatent: Bool,
        loss: GANLossType,
        learningRates: GDPair<Float>,
        startSize: ImageSize,
        endSize: ImageSize,
        batchSizes: [ImageSize: Int],
        imagesPerPhase: Int
    ) {
        self.latentSize = latentSize
        self.startSize = startSize
        self.endSize = endSize
        self.normalizeLatent = normalizeLatent
        self.loss = loss
        self.learningRates = learningRates
        self.batchSizes = batchSizes
        self.imagesPerPhase = imagesPerPhase
    }
}

public struct GDPair<T: Codable>: Codable {
    public var G: T
    public var D: T
    
    public init(G: T, D: T) {
        self.G = G
        self.D = D
    }
}
