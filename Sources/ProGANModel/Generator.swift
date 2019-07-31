import Foundation
import TensorFlow

public struct GeneratorFirstBlock: Layer {
    var conv1: WSConv2D
    var conv2: WSConv2D
    
    public init() {
        conv1 = WSConv2D(inputChannels: Config.latentSize,
                         outputChannels: 1024*4*4,
                         kernelSize: (1, 1),
                         activation: lrelu)
        conv2 = WSConv2D(inputChannels: 1024,
                         outputChannels: 1024,
                         kernelSize: (3, 3),
                         activation: lrelu)
    }
    
    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        let batchSize = input.shape[0]
        var x = input.reshaped(to: [batchSize, 1, 1, Config.latentSize])
        x = conv1(x) // [batchSize, 1, 1, 1024*4*4]
        x = x.reshaped(to: [batchSize, 4, 4, 1024])
        x = pixelNormalization(x)
        x = pixelNormalization(conv2(x)) // [batchSize, 1024, 4, 4]
        return x
    }
}

public struct GeneratorBlock: Layer {
    var conv1: WSConv2D
    var conv2: WSConv2D

    public init(inputChannels: Int, outputChannels: Int) {
        conv1 = WSConv2D(inputChannels: inputChannels,
                         outputChannels: outputChannels,
                         kernelSize: (3, 3),
                         activation: lrelu)
        conv2 = WSConv2D(inputChannels: outputChannels,
                         outputChannels: outputChannels,
                         kernelSize: (3, 3),
                         activation: lrelu)
    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        var x = input
        x = pixelNormalization(conv1(input))
        x = pixelNormalization(conv2(x))
        return x
    }
}

public struct Generator: Layer {
    public var firstBlock = GeneratorFirstBlock()

    public var blocks: [GeneratorBlock] = []

    public var upsample = UpSampling2D<Float>(size: 2)
    
    public var toRGB1 = WSConv2D(inputChannels: 1, outputChannels: 1, kernelSize: (1, 1), activation: tanh)
    public var toRGB2 = WSConv2D(inputChannels: 1024, outputChannels: 3, kernelSize: (1, 1), activation: tanh)
    
    @noDerivative
    public private(set) var level = 1

    public init() {}
    
    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        var x = firstBlock(input)

        guard level > 1 else {
            // 常にalpha = 1
            return toRGB2(x)
        }

        for lv in 0..<level-2 {
            x = upsample(x)
            x = blocks[lv](x)
        }
        
        var x1 = x
        x1 = toRGB1(x1)
        x1 = upsample(x1)

        var x2 = upsample(x)
        x2 = blocks[level-2](x2)
        x2 = toRGB2(x2)

        return lerp(x1, x2, rate: GlobalState.alpha)
    }
    
    static let ioChannels = [
        (1024, 512),
        (512, 512),
        (512, 256),
        (256, 256),
        (256, 128),
        (128, 128)
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
        toRGB2 = WSConv2D(inputChannels: io.1, outputChannels: 3, kernelSize: (1, 1), activation: tanh)
    }
}
