import Foundation
import TensorFlow

struct GBlock: Layer {
    var conv1: EqualizedConv2D
    var conv2: EqualizedConv2D
    
    @noDerivative
    let firstBlock: Bool
    
    @differentiable
    func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        var x = input
        if !firstBlock {
            x = resize2xBilinear(images: x)
        }
        x = conv1(x)
        if firstBlock {
            x = x.reshaped(to: [x.shape[0], 4, 4, -1])
        }
        x = pixelNormalization(x)
        x = pixelNormalization(conv2(x))
        
        return x
    }
}

public struct Generator: Layer {
    static let channels = [
        Config.latentSize, 256, 256, 256, 128, 64, 32, 16
    ]
    
    var blocks: [GBlock] = []
    var toRGBs: [EqualizedConv2D] = []
    
    @noDerivative
    public private(set) var level = 1
    @noDerivative
    public var alpha: Float = 1.0
    
    public init() {
        let channels = Self.channels
        let firstBlock = GBlock(
            conv1: .init(inputChannels: channels[0], outputChannels: channels[1] * 4*4,
                         kernelSize: (1, 1), padding: .valid, activation: lrelu), // Dense
            conv2: .init(inputChannels: channels[1], outputChannels: channels[1],
                         kernelSize: (3, 3), padding: .same, activation: lrelu),
            firstBlock: true
        )
        blocks.append(firstBlock)
        toRGBs.append(.init(inputChannels: channels[1], outputChannels: 3,
                            kernelSize: (1, 1), padding: .valid))
        
        for i in 1..<Config.maxLevel-1 {
            let block = GBlock(
                conv1: .init(inputChannels: channels[i], outputChannels: channels[i+1],
                             kernelSize: (3, 3), padding: .same, activation: lrelu),
                conv2: .init(inputChannels: channels[i+1], outputChannels: channels[i+1],
                             kernelSize: (3, 3), padding: .same, activation: lrelu),
                firstBlock: false
            )
            blocks.append(block)
            toRGBs.append(.init(inputChannels: channels[i+1], outputChannels: 3,
                                kernelSize: (1, 1), padding: .valid))
        }
    }
    
    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        var x = input
        if Config.normalizeLatent {
            x = pixelNormalization(x)
        }
        x = x.expandingShape(at: 1, 2)
        
        guard level > 1 else {
            // alpha = 1
            return toRGBs[0](blocks[0](x))
        }
        
        for lv in 0..<level-2 {
            x = blocks[lv](x)
        }
        
        x = blocks[level-2](x)
        let rgb1 = resize2xBilinear(images: toRGBs[level-2](x))
        
        x = blocks[level-1](x)
        let rgb2 = toRGBs[level-1](x)
        
        return lerp(rgb1, rgb2, rate: alpha)
    }
    
    public mutating func grow() {
        level += 1
        guard level <= Config.maxLevel else {
            fatalError("Generator.level exceeds Config.maxLevel")
        }
    }
    
    public func getHistogramWeights() -> [String: Tensor<Float>] {
        var dict: [String: Tensor<Float>] = [:]
        
        for i in 0..<level {
            dict["gen/block\(i).conv1"] = blocks[i].conv1.filter
            dict["gen/block\(i).conv2"] = blocks[i].conv2.filter
            dict["gen/block\(i).toRGB"] = toRGBs[i].filter
        }
        
        return dict
    }
}
