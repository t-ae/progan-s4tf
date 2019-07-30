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
    public var firstBlock: GeneratorFirstBlock

    public var blocks: [GeneratorBlock]
    public var toRGB: [WSConv2D]

    public var upsample: UpSampling2D<Float>

    public init() {
        firstBlock = GeneratorFirstBlock()
        upsample = UpSampling2D(size: 2)

        blocks = [
            GeneratorBlock(inputChannels: 1024, outputChannels: 512), // 8x8
            GeneratorBlock(inputChannels: 512, outputChannels: 512), // 16x16
            GeneratorBlock(inputChannels: 512, outputChannels: 256), // 32x32
            GeneratorBlock(inputChannels: 256, outputChannels: 256), // 64x64
            GeneratorBlock(inputChannels: 256, outputChannels: 128), // 128x128
            GeneratorBlock(inputChannels: 128, outputChannels: 128), // 256x256
        ]

        toRGB = [
            WSConv2D(inputChannels: 512, outputChannels: 3, kernelSize: (1, 1), activation: tanh),
            WSConv2D(inputChannels: 512, outputChannels: 3, kernelSize: (1, 1), activation: tanh),
            WSConv2D(inputChannels: 256, outputChannels: 3, kernelSize: (1, 1), activation: tanh),
            WSConv2D(inputChannels: 256, outputChannels: 3, kernelSize: (1, 1), activation: tanh),
            WSConv2D(inputChannels: 128, outputChannels: 3, kernelSize: (1, 1), activation: tanh),
            WSConv2D(inputChannels: 128, outputChannels: 3, kernelSize: (1, 1), activation: tanh),
        ]
    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        var x = firstBlock(input)
        let level = GlobalState.level

        guard level > 1 else {
            // 常にalpha = 1
            return toRGB[0](blocks[0](upsample(x)))
        }

        for lv in 0..<level-1 {
            x = upsample(x)
            x = blocks[lv](x)
        }
        
        var x1 = x
        x1 = toRGB[level-2](x1)
        x1 = upsample(x1)

        var x2 = upsample(x)
        x2 = blocks[level-1](x2)
        x2 = toRGB[level-1](x2)

        return lerp(x1, x2, rate: GlobalState.alpha)
    }
}
