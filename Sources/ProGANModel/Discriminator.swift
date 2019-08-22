import Foundation
import TensorFlow

// Add per channel/batch noise
// https://github.com/tkarras/progressive_growing_of_gans/blob/original-theano-version/network.py#L360-L400
@differentiable
func addNoise(_ x: Tensor<Float>, noiseScale: Float) -> Tensor<Float> {
    let noiseShape: TensorShape = [x.shape[0], 1, 1, x.shape[3]]
    let scale = noiseScale * sqrt(Float(x.shape[3]))
    let noise = Tensor<Float>(randomNormal: noiseShape) * scale + 1
    return x * noise
}

struct DiscriminatorBlockInput: Differentiable {
    var x: Tensor<Float>
    @noDerivative
    var noiseScale: Float
}

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
    func callAsFunction(_ input: DiscriminatorBlockInput) -> Tensor<Float> {
        var x = input.x
        x = addNoise(x, noiseScale: input.noiseScale)
        x = conv1(x)
        x = addNoise(x, noiseScale: input.noiseScale)
        x = conv2(x)
        if !Config.useFusedScale {
            x = avgPool(x)
        }
        return x
    }
}

struct DiscriminatorLastBlock: Layer {
    var conv1: EqualizedConv2D
    var conv2: EqualizedConv2D
    var dense: EqualizedDense
    
    public init() {
        conv1 = EqualizedConv2D(inputChannels: 257,
                                outputChannels: 256,
                                kernelSize: (3, 3),
                                activation: lrelu)
        conv2 = EqualizedConv2D(inputChannels: 256,
                                outputChannels: 256,
                                kernelSize: (4, 4),
                                padding: .valid,
                                activation: lrelu)
        dense = EqualizedDense(inputSize: 256,
                               outputSize: 1,
                               activation: identity,
                               gain: 1)
    }
    
    @differentiable
    public func callAsFunction(_ input: DiscriminatorBlockInput) -> Tensor<Float> {
        var x = input.x
        let batchSize = x.shape[0]
        
        x = minibatchStdConcat(x)
        x = addNoise(x, noiseScale: input.noiseScale)
        x = conv1(x)
        x = addNoise(x, noiseScale: input.noiseScale)
        x = conv2(x)
        
        x = x.reshaped(to: [batchSize, -1])
        x = dense(x)
        return x
    }
}

public struct Discriminator: Layer {
    
    var lastBlock = DiscriminatorLastBlock()
    
    var blocks: [DiscriminatorBlock] = []
    
    var fromRGB1 = EqualizedConv2D(inputChannels: 3, outputChannels: 1, kernelSize: (1, 1)) // dummy at first
    var fromRGB2 = EqualizedConv2D(inputChannels: 3, outputChannels: 256, kernelSize: (1, 1))
    
    var downsample = AvgPool2D<Float>(poolSize: (2, 2), strides: (2, 2))
    
    @noDerivative
    public private(set) var level = 1
    
    // Mean of output for fake images
    @noDerivative
    public let outputMean: Parameter<Float> = Parameter(Tensor(0))
    
    public init() {}
    
    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        
        // Described in Appendix B
        let noiseScale: Float
        if Config.loss == .lsgan {
            noiseScale = 0.2 * pow(max(outputMean.value.scalar! - 0.5, 0), 2)
        } else {
            noiseScale = 0
        }
        
        guard level > 1 else {
            // alpha = 1
            return lastBlock(DiscriminatorBlockInput(x: fromRGB2(input), noiseScale: noiseScale))
        }
        
        let x1 = fromRGB1(downsample(input))
        var x2 = fromRGB2(input)
        
        let lastIndex = level-2
        x2 = blocks[lastIndex](DiscriminatorBlockInput(x: x2, noiseScale: noiseScale))
        
        var x = lerp(x1, x2, rate: GlobalState.alpha)
        
        for l in (0..<lastIndex).reversed() {
            x = blocks[l](DiscriminatorBlockInput(x: x, noiseScale: noiseScale))
        }
        
        return lastBlock(DiscriminatorBlockInput(x: x, noiseScale: noiseScale))
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
    
    public func getHistogramWeights() -> [String: Tensor<Float>] {
        return [
            "disc/last_conv1": lastBlock.conv1.conv.filter,
            "disc/last_conv2": lastBlock.conv2.conv.filter,
            "disc/last_dense": lastBlock.dense.dense.weight,
        ]
    }
}
