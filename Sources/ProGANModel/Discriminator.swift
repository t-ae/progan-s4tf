import Foundation
import TensorFlow

struct DiscriminatorBlock: Layer {
    var conv1: EqualizedConv2D
    var conv2: EqualizedConv2D
    
    var avgPool = AvgPool2D<Float>(poolSize: (2, 2), strides: (2, 2))
    
    init(inputChannels: Int, outputChannels: Int) {
        let stride = Config.useFusedScale ? 2 : 1
        
        conv1 = EqualizedConv2D(inputChannels: inputChannels,
                                outputChannels: outputChannels,
                                kernelSize: (3, 3),
                                activation: lrelu)
        conv2 = EqualizedConv2D(inputChannels: outputChannels,
                                outputChannels: outputChannels,
                                kernelSize: (3, 3),
                                strides: (stride, stride),
                                activation: lrelu)
    }
    
    @differentiable
    func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        var x = input
        x = conv1(x)
        x = conv2(x)
        if !Config.useFusedScale {
            x = avgPool(x)
        }
        return x
    }
}

struct DiscriminatorLastBlock: Layer {
    var conv: EqualizedConv2D
    var dense1: EqualizedDense
    var dense2: EqualizedDense
    
    public init() {
        conv = EqualizedConv2D(inputChannels: 256,
                               outputChannels: 256,
                               kernelSize: (4, 4),
                               padding: .valid,
                               activation: lrelu)
        dense1 = EqualizedDense(inputSize: 256,
                                outputSize: 128,
                                activation: lrelu)
        dense2 = EqualizedDense(inputSize: 128,
                                outputSize: 1,
                                activation: identity)
    }
    
    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        let batchSize = input.shape[0]
        var x = input
        // FIXME: Temporary disabled due to crash in Ubuntu/CUDA
        //x = minibatchStdConcat(x)
        x = conv(x)
        x = x.reshaped(to: [batchSize, -1])
        x = dense1(x)
        x = dense2(x)
        return x
    }
}

public struct Discriminator: Layer {
    
    var lastBlock = DiscriminatorLastBlock()
    
    var blocks: [DiscriminatorBlock] = []
    
    var fromRGB1 = EqualizedConv2D(inputChannels: 3, outputChannels: 1, kernelSize: (1, 1))
    var fromRGB2 = EqualizedConv2D(inputChannels: 3, outputChannels: 256, kernelSize: (1, 1))
    
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
        
        var x = lerp(x1, x2, rate: GlobalState.alpha)
        
        for l in (0..<lastIndex).reversed() {
            x = blocks[l](x)
        }
        
        return lastBlock(x)
    }
    
    static let ioChannels = [
        (256, 256),
        (256, 256),
        (128, 256),
        (64, 128),
        (32, 64),
        (16, 32),
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
        fromRGB2 = EqualizedConv2D(inputChannels: 3, outputChannels: io.0, kernelSize: (1, 1))
    }
}
