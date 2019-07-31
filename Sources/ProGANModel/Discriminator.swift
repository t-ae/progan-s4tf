import Foundation
import TensorFlow

struct DiscriminatorBlock: Layer {
    var conv1: WSConv2D
    var conv2: WSConv2D
    
    init(inputChannels: Int, outputChannels: Int) {
        conv1 = WSConv2D(inputChannels: inputChannels,
                         outputChannels: outputChannels,
                         kernelSize: (3, 3),
                         activation: lrelu)
        conv2 = WSConv2D(inputChannels: outputChannels,
                         outputChannels: outputChannels,
                         kernelSize: (3, 3),
                         activation: identity)
    }
    
    @differentiable
    func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        var x = input
        x = conv1(x)
        x = conv2(x)
        return x
    }
}

struct DiscriminatorLastBlock: Layer {
    var conv1: WSConv2D
    var conv2: WSConv2D
    
    public init() {
        conv1 = WSConv2D(inputChannels: 1024,
                         outputChannels: 1024,
                         kernelSize: (3, 3),
                         activation: lrelu)
        conv2 = WSConv2D(inputChannels: 1025,
                         outputChannels: 1,
                         kernelSize: (4, 4),
                         padding: .valid,
                         activation: lrelu)
    }
    
    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        let batchSize = input.shape[0]
        var x = input
        x = conv1(x)
        x = minibatchStdConcat(x)
        x = conv2(x)
        x = x.reshaped(to: [batchSize, 1])
        return x
    }
}

public struct Discriminator: Layer {
    
    var lastBlock = DiscriminatorLastBlock()
    
    var blocks: [DiscriminatorBlock] = []
    
    var fromRGB1 = WSConv2D(inputChannels: 3, outputChannels: 1, kernelSize: (1, 1))
    var fromRGB2 = WSConv2D(inputChannels: 3, outputChannels: 1024, kernelSize: (1, 1))
    
    var downsample = AvgPool2D<Float>(poolSize: (2, 2), strides: (2, 2))
        
    @noDerivative
    public private(set) var level = 1
    
    public init() {}
    
    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        guard level > 1 else {
            // alpha = 1
            return lastBlock(fromRGB2(input))
        }
        
        let x1 = fromRGB1(downsample(input))
        var x2 = fromRGB2(input)
    
        let lastIndex = level-2
        x2 = blocks[lastIndex](x2)
        x2 = downsample(x2)
        
        var x = lerp(x1, x2, rate: GlobalState.alpha)
        
        for l in (0..<lastIndex).reversed() {
            x = blocks[l](x)
            x = downsample(x)
        }
        
        return lastBlock(x)
    }
    
    static let ioChannels = [
        (512, 1024),
        (512, 512),
        (256, 512),
        (256, 256),
        (128, 256),
        (128, 128),
    ]
    
    public mutating func grow() {
        level += 1
        guard level <= Config.maxLevel else {
            fatalError("Generator.level exceeds Config.maxLevel")
        }
        
        let blockCount = blocks.count
        let io = Discriminator.ioChannels[blockCount]
        
        blocks.append(DiscriminatorBlock(inputChannels: io.0,outputChannels: io.1))
        
        fromRGB1 = fromRGB2
        fromRGB2 = WSConv2D(inputChannels: 3, outputChannels: io.0, kernelSize: (1, 1))
    }
}
