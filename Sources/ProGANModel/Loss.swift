import Foundation
import TensorFlow

public enum LossType {
    case nonSaturating, lsgan, wgan
    
    public func createLoss() -> Loss {
        switch self {
        case .nonSaturating:
            return NonSaturatingLoss()
        case .lsgan:
            return LSGANLoss()
        case .wgan:
            return WGANLoss()
        }
    }
}

public protocol Loss {
    @differentiable
    func generatorLoss(fake: Tensor<Float>) -> Tensor<Float>
    
    @differentiable
    func discriminatorLoss(real: Tensor<Float>, fake: Tensor<Float>) -> Tensor<Float>
}

public struct NonSaturatingLoss: Loss {
    public init() {}
    
    @differentiable
    public func generatorLoss(fake: Tensor<Float>) -> Tensor<Float> {
        softplus(-fake).mean()
    }
    
    @differentiable
    public func discriminatorLoss(real: Tensor<Float>, fake: Tensor<Float>) -> Tensor<Float> {
        let realLoss = softplus(-real).mean()
        let fakeLoss = softplus(fake).mean()
        
        return realLoss + fakeLoss
    }
}

// https://github.com/tkarras/progressive_growing_of_gans/blob/original-theano-version/train.py#L345-L347
public struct LSGANLoss: Loss {
    public var fakeWeight: Float = 0.1
    
    public init() {}
    
    @differentiable
    public func generatorLoss(fake: Tensor<Float>) -> Tensor<Float> {
        meanSquaredError(predicted: fake, expected: Tensor<Float>(zeros: fake.shape)) / 2
    }
    
    @differentiable
    public func discriminatorLoss(real: Tensor<Float>, fake: Tensor<Float>) -> Tensor<Float> {
        let realLoss = meanSquaredError(predicted: real, expected: Tensor<Float>(zeros: real.shape))
        let fakeLoss = meanSquaredError(predicted: fake, expected: Tensor<Float>(ones: fake.shape))
        
        return realLoss + fakeLoss * fakeWeight
    }
}

public struct WGANLoss: Loss {
    public init() {}
    
    @differentiable
    public func generatorLoss(fake: Tensor<Float>) -> Tensor<Float> {
        return -fake.mean()
    }
    
    @differentiable
    public func discriminatorLoss(real: Tensor<Float>, fake: Tensor<Float>) -> Tensor<Float> {
        
        let wganLoss = fake.mean() - real.mean()
        
        let driftLoss = 1e-3 * real.squared().mean()
        
        return wganLoss + driftLoss
    }
}
