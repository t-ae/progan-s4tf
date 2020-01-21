import Foundation
import TensorFlow

struct GBlock: Layer {
    var conv1: SNConv2D<Float>
    var conv2: SNConv2D<Float>
    
    init(inputChannels: Int, outputChannels: Int, enableSN: Bool) {
        conv1 = SNConv2D(Conv2D(filterShape: (3, 3, inputChannels, outputChannels),
                                padding: .same,
                                activation: lrelu),
                         enabled: enableSN)
        conv2 = SNConv2D(Conv2D(filterShape: (3, 3, outputChannels, outputChannels),
                                padding: .same,
                                activation: lrelu),
                         enabled: enableSN)
    }
    
    @differentiable
    func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        var x = input
        x = conv1(x)
        x = pixelNormalization(x)
        x = conv2(x)
        x = pixelNormalization(x)
        return x
    }
}

public struct Generator: Layer {
    var head: SNDense<Float>
    var blocks: [GBlock] = []
    var toRGBs: [SNConv2D<Float>] = []
    
    @noDerivative
    public private(set) var level = 1
    
    @noDerivative
    public var imageSize: ImageSize = .x4
    
    @noDerivative
    public var alpha: Float = 1.0
    
    @noDerivative
    public let config: Config
    
    public init(config: Config) {
        self.config = config
        let enableSN = config.enableSpectralNorm.G
        
        head = SNDense(Dense(inputSize: config.latentSize, outputSize: 4*4*256), enabled: enableSN)
        blocks = [
            GBlock(inputChannels: 256, outputChannels: 256, enableSN: enableSN), // 4x4
            GBlock(inputChannels: 256, outputChannels: 256, enableSN: enableSN), // 8x8
            GBlock(inputChannels: 256, outputChannels: 128, enableSN: enableSN), // 16x16
            GBlock(inputChannels: 128, outputChannels: 64, enableSN: enableSN), // 32x32
            GBlock(inputChannels: 64, outputChannels: 32, enableSN: enableSN), // 64x64
            GBlock(inputChannels: 32, outputChannels: 16, enableSN: enableSN), // 128x128
            GBlock(inputChannels: 16, outputChannels: 8, enableSN: enableSN), // 256x256
        ]
        toRGBs = [
            SNConv2D(Conv2D(filterShape: (1, 1, 256, 3), activation: identity), enabled: enableSN),
            SNConv2D(Conv2D(filterShape: (1, 1, 256, 3), activation: identity), enabled: enableSN),
            SNConv2D(Conv2D(filterShape: (1, 1, 128, 3), activation: identity), enabled: enableSN),
            SNConv2D(Conv2D(filterShape: (1, 1, 64, 3), activation: identity), enabled: enableSN),
            SNConv2D(Conv2D(filterShape: (1, 1, 32, 3), activation: identity), enabled: enableSN),
            SNConv2D(Conv2D(filterShape: (1, 1, 16, 3), activation: identity), enabled: enableSN),
            SNConv2D(Conv2D(filterShape: (1, 1, 8, 3), activation: identity), enabled: enableSN),
        ]
    }
    
    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        var x = input
        if config.normalizeLatent {
            x = pixelNormalization(x)
        }
        x = head(x) // 4x4x256
        x = x.reshaped(to: [-1, 4, 4, 256])
        
        x = blocks[0](x)
        
        guard imageSize > .x4 else {
            x = toRGBs[0](x)
            return x
        }
        
        let endIndex = imageSize.log2 - 2
        
        for i in 0...endIndex-1 {
            x = resize2xBilinear(images: x)
            x = blocks[i](x)
        }
        
        var x2 = x // half size
        
        x = resize2xBilinear(images: x)
        x = blocks[endIndex](x)
        
        if alpha < 1 {
            x2 = toRGBs[endIndex-1](x2)
            x2 = resize2xBilinear(images: x2)
            x = lerp(x2, x, rate: alpha)
        }
        return x
    }
    
}
