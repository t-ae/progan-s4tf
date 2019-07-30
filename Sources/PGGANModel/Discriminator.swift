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
        conv2 = WSConv2D(inputChannels: 1024,
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
        x = conv2(x)
        x = x.reshaped(to: [batchSize, 1])
        return x
    }
}

public struct Discriminator: Layer {
    
    var lastBlock: DiscriminatorLastBlock
    
    var blocks: [DiscriminatorBlock]
    
    var fromRGB: [WSConv2D]
    
    var downsample: AvgPool2D<Float>
    
    public init() {
        lastBlock = DiscriminatorLastBlock()
        
        blocks = [
            DiscriminatorBlock(inputChannels: 128,outputChannels: 256),
            DiscriminatorBlock(inputChannels: 256,outputChannels: 256),
            DiscriminatorBlock(inputChannels: 256,outputChannels: 512),
            DiscriminatorBlock(inputChannels: 512,outputChannels: 512),
            DiscriminatorBlock(inputChannels: 512,outputChannels: 1024),
            DiscriminatorBlock(inputChannels: 1024,outputChannels: 1024),
        ]
        
        fromRGB = [
            WSConv2D(inputChannels: 3, outputChannels: 128, kernelSize: (1, 1)), // lv6
            WSConv2D(inputChannels: 3, outputChannels: 256, kernelSize: (1, 1)),
            WSConv2D(inputChannels: 3, outputChannels: 256, kernelSize: (1, 1)),
            WSConv2D(inputChannels: 3, outputChannels: 512, kernelSize: (1, 1)),
            WSConv2D(inputChannels: 3, outputChannels: 512, kernelSize: (1, 1)),
            WSConv2D(inputChannels: 3, outputChannels: 1024, kernelSize: (1, 1)), // lv1
        ]
        
        downsample = AvgPool2D.init(poolSize: (2, 2), strides: (2, 2))
    }
    
    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        let level = GlobalState.level
        let startLayer = Config.maxLevel - level
        
        var x1 = fromRGB[startLayer](input)
        x1 = downsample(blocks[startLayer](x1))
        
        guard level > 1 else {
            x1 = lastBlock(x1)
            return x1
        }
        
        var x2 = downsample(input)
        x2 = fromRGB[startLayer+1](x2)
        
        var x = lerp(x1, x2, rate: GlobalState.alpha)
        
        for l in startLayer+1..<Config.maxLevel {
            x = blocks[l](x)
            x = downsample(x)
        }
        
        return lastBlock(x)
    }
}
