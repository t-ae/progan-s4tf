import Foundation
import TensorFlow

// Add per channel noise
// https://github.com/tkarras/progressive_growing_of_gans/blob/original-theano-version/network.py#L360-L400
@differentiable(wrt: x)
func addNoise(_ x: Tensor<Float>, noiseScale: Float) -> Tensor<Float> {
    let noiseShape: TensorShape = [1, 1, 1, x.shape[3]]
    let scale = noiseScale * sqrt(Float(x.shape[3]))
    let noise = Tensor<Float>(randomNormal: noiseShape) * scale + 1
    return x * noise
}

@differentiable
func avgPool(_ x: Tensor<Float>) -> Tensor<Float> {
    avgPool2D(x, filterSize: (1, 2, 2, 1), strides: (1, 2, 2, 1), padding: .valid)
}

struct DBlock: Layer {
    struct Input: Differentiable {
        var x: Tensor<Float>
        @noDerivative
        var noiseScale: Float
    }
    
    var conv1: EqualizedConv2D
    var conv2: EqualizedConv2D
    
    @noDerivative
    let lastBlock: Bool
    
    @differentiable
    func callAsFunction(_ input: Input) -> Tensor<Float> {
        var x = input.x
        if lastBlock {
            x = minibatchStdConcat(x)
        }
        x = addNoise(x, noiseScale: input.noiseScale)
        x = conv1(x)
        x = addNoise(x, noiseScale: input.noiseScale)
        x = conv2(x)
        if !lastBlock {
            x = avgPool(x)
        }
        return x
    }
}

public struct Discriminator: Layer {
    static let channels = [
        256, 256, 256, 256, 128, 64, 32, 16
    ]
    
    var blocks: [DBlock] = []
    var fromRGBs: [EqualizedConv2D] = []
    
    var lastDense: EqualizedDense
    
    @noDerivative
    public private(set) var level = 1
    @noDerivative
    public var alpha: Float = 1.0
    
    // Mean of output for fake images
    @noDerivative
    public let outputMean: Parameter<Float> = Parameter(Tensor(0))
    
    public init() {
        let channels = Self.channels
        
        // first block
        fromRGBs.append(.init(inputChannels: 3, outputChannels: channels[1],
                              kernelSize: (1, 1), padding: .valid, activation: lrelu))
        blocks.append(.init(
            conv1: .init(inputChannels: channels[1]+1, outputChannels: channels[0],
                         kernelSize: (3, 3), padding: .same, activation: lrelu),
            conv2: .init(inputChannels: channels[0], outputChannels: channels[0],
                         kernelSize: (4, 4), padding: .valid, activation: lrelu),
            lastBlock: true
        ))
        
        for lv in 2...Config.maxLevel {
            fromRGBs.append(.init(inputChannels: 3, outputChannels: channels[lv],
                                  kernelSize: (1, 1), padding: .valid, activation: lrelu))
            blocks.append(.init(
                conv1: .init(inputChannels: channels[lv], outputChannels: channels[lv-1],
                             kernelSize: (3, 3), padding: .same, activation: lrelu),
                conv2: .init(inputChannels: channels[lv-1], outputChannels: channels[lv-1],
                             kernelSize: (3, 3), padding: .same, activation: lrelu),
                lastBlock: false
            ))
        }
        
        lastDense = EqualizedDense(inputSize: channels[0], outputSize: 1)
    }
    
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
            var x = fromRGBs[0](input)
            x = blocks[0](.init(x: x, noiseScale: noiseScale))
            x = x.squeezingShape(at: 1, 2)
            return lastDense(x)
        }
        
        let x1 = fromRGBs[level-2](avgPool(input))
        var x2 = fromRGBs[level-1](input)
        x2 = blocks[level-1](.init(x: x2, noiseScale: noiseScale))
        
        var x = lerp(x1, x2, rate: alpha)
        
        for l in (0...level-2).reversed() {
            x = blocks[l](.init(x: x, noiseScale: noiseScale))
        }
        x = x.squeezingShape(at: 1, 2)
        return lastDense(x)
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
            fatalError("Discriminator.level exceeds Config.maxLevel")
        }
    }
    
    public func getHistogramWeights() -> [String: Tensor<Float>] {
        var dict = [
            "disc\(level)/last.conv1": lastDense.weight,
        ]
        
        for i in 0..<level {
            dict["disc\(level)/block\(i).conv1"] = blocks[i].conv1.filter
            dict["disc\(level)/block\(i).conv2"] = blocks[i].conv2.filter
            dict["disc\(level)/block\(i).fromRGB"] = fromRGBs[i].filter
        }
        
        return dict
    }
}
