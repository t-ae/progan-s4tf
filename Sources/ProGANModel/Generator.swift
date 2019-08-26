import Foundation
import TensorFlow

public struct GeneratorFirstBlock: Layer {
    var dense: EqualizedDense
    var conv: EqualizedConv2D
    
    public init() {
        dense = EqualizedDense(inputSize: Config.latentSize,
                               outputSize: 256*4*4,
                               activation: lrelu,
                               gain: sqrt(2)/4)
        conv = EqualizedConv2D(inputChannels: 256,
                               outputChannels: 256,
                               kernelSize: (3, 3),
                               activation: lrelu)
    }
    
    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        let batchSize = input.shape[0]
        var x = dense(input)
        x = x.reshaped(to: [batchSize, 4, 4, 256])
        x = pixelNormalization(x)
        x = pixelNormalization(conv(x)) // [batchSize, 256, 4, 4]
        return x
    }
}

public struct GeneratorBlock: Layer {
    var conv1: EqualizedConv2D
    var conv2: EqualizedConv2D
    
    public init(inputChannels: Int, outputChannels: Int) {
        conv1 = EqualizedConv2D(inputChannels: inputChannels,
                                outputChannels: outputChannels,
                                kernelSize: (3, 3),
                                activation: lrelu)
        conv2 = EqualizedConv2D(inputChannels: outputChannels,
                                outputChannels: outputChannels,
                                kernelSize: (3, 3),
                                activation: lrelu)
    }
    
    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        var x = input
        x = resize2xBilinear(images: x)
        x = pixelNormalization(conv1(x))
        x = pixelNormalization(conv2(x))
        return x
    }
}

public struct Generator: Layer {
    public var firstBlock = GeneratorFirstBlock()
    
    public var blocks: [GeneratorBlock] = []
    
    public var toRGB1 = EqualizedConv2D(inputChannels: 1, outputChannels: 1, kernelSize: (1, 1), activation: identity, gain: 1) // dummy at first
    public var toRGB2 = EqualizedConv2D(inputChannels: 256, outputChannels: 3, kernelSize: (1, 1), activation: identity, gain: 1)
    
    @noDerivative
    public private(set) var level = 1
    @noDerivative
    public var alpha: Float = 1.0
    
    public init() {}
    
    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        var x = input
        if Config.normalizeLatent {
            x = pixelNormalization(x)
        }
        x = firstBlock(x)
        
        guard level > 1 else {
            // 常にalpha = 1
            return toRGB2(x)
        }
        
        for lv in 0..<level-2 {
            x = blocks[lv](x)
        }
        
        var x1 = x
        x1 = toRGB1(x1)
        x1 = resize2xBilinear(images: x1)
        
        var x2 = blocks[level-2](x)
        x2 = toRGB2(x2)
        
        return lerp(x1, x2, rate: alpha)
    }
    
    static let ioChannels = [
        (256, 256),
        (256, 256),
        (256, 128),
        (128, 64),
        (64, 32),
        (32, 16)
    ]
    
    public mutating func grow() {
        level += 1
        guard level <= Config.maxLevel else {
            fatalError("Generator.level exceeds Config.maxLevel")
        }
        
        let blockCount = blocks.count
        let io = Generator.ioChannels[blockCount]
        
        blocks.append(GeneratorBlock(inputChannels: io.0, outputChannels: io.1))
        toRGB1 = toRGB2
        toRGB2 = EqualizedConv2D(inputChannels: io.1, outputChannels: 3, kernelSize: (1, 1), activation: identity, gain: 1)
    }
    
    public func getHistogramWeights() -> [String: Tensor<Float>] {
        var dict = [
            "gen\(level)/first.dense": firstBlock.dense.weight,
            "gen\(level)/first.conv": firstBlock.conv.filter,
            "gen\(level)/toRGB1": toRGB1.filter,
            "gen\(level)/toRGB2": toRGB2.filter,
        ]
        
        for i in 0..<blocks.count {
            dict["gen\(level)/block\(i).conv1"] = blocks[i].conv1.filter
            dict["gen\(level)/block\(i).conv2"] = blocks[i].conv2.filter
        }
        
        return dict
    }
}
