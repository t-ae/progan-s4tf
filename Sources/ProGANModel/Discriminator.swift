import Foundation
import TensorFlow

public struct DBlock: Layer {
    public var conv1: SNConv2D<Float>
    public var conv2: SNConv2D<Float>
    
    public init(inputChannels: Int, outputChannels: Int, enableSN: Bool) {
        conv1 = SNConv2D(Conv2D(filterShape: (3, 3, inputChannels, outputChannels),
                                padding: .same,
                                activation: lrelu,
                                filterInitializer: heNormal()),
                         enabled: enableSN)
        conv2 = SNConv2D(Conv2D(filterShape: (3, 3, outputChannels, outputChannels),
                                padding: .same,
                                activation: lrelu,
                                filterInitializer: heNormal()),
                         enabled: enableSN)
    }
    
    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        var x = input
        x = conv1(x)
        x = conv2(x)
        return x
    }
}

public struct Discriminator: Layer {
    public var fromRGBs: [SNConv2D<Float>] = []
    
    public var blocks: [DBlock] = []
    
    public var minibatchStdConcat: MinibatchStdConcat<Float>
    
    public var lastConv: SNConv2D<Float>
    public var lastDense: SNDense<Float>
    
    public var avgPool = AvgPool2D<Float>(poolSize: (2, 2), strides: (2, 2))
    
    @noDerivative
    public var imageSize: ImageSize = .x4
    
    @noDerivative
    public var alpha: Float = 1.0
    
    public init(config: Config) {
        let enableSN = config.enableSpectralNorm.D
        fromRGBs = [
            SNConv2D(Conv2D(filterShape: (1, 1, 3, 8),
                            activation: lrelu,
                            filterInitializer: heNormal()),
                     enabled: enableSN), // 256x256
            SNConv2D(Conv2D(filterShape: (1, 1, 3, 16),
                            activation: lrelu,
                            filterInitializer: heNormal()),
                     enabled: enableSN), // 128x128
            SNConv2D(Conv2D(filterShape: (1, 1, 3, 32),
                            activation: lrelu,
                            filterInitializer: heNormal()),
                     enabled: enableSN), // 64x64
            SNConv2D(Conv2D(filterShape: (1, 1, 3, 64),
                            activation: lrelu,
                            filterInitializer: heNormal()),
                     enabled: enableSN), // 32x32
            SNConv2D(Conv2D(filterShape: (1, 1, 3, 128),
                            activation: lrelu,
                            filterInitializer: heNormal()),
                     enabled: enableSN), // 16x16
            SNConv2D(Conv2D(filterShape: (1, 1, 3, 256),
                            activation: lrelu,
                            filterInitializer: heNormal()),
                     enabled: enableSN), // 8x8
            SNConv2D(Conv2D(filterShape: (1, 1, 3, 256),
                            activation: lrelu,
                            filterInitializer: heNormal()),
                     enabled: enableSN), // 4x4
        ]
        blocks = [
            DBlock(inputChannels: 8, outputChannels: 16, enableSN: enableSN), // 256x256
            DBlock(inputChannels: 16, outputChannels: 32, enableSN: enableSN), // 128x128
            DBlock(inputChannels: 32, outputChannels: 64, enableSN: enableSN), // 64x64
            DBlock(inputChannels: 64, outputChannels: 128, enableSN: enableSN), // 32x32
            DBlock(inputChannels: 128, outputChannels: 256, enableSN: enableSN), // 16x16
            DBlock(inputChannels: 256, outputChannels: 256, enableSN: enableSN), // 8x8
            DBlock(inputChannels: 256, outputChannels: 256, enableSN: enableSN), // 4x4
        ]
        
        minibatchStdConcat = MinibatchStdConcat(groupSize: 4)
        lastConv = SNConv2D(Conv2D(filterShape: (3, 3, 257, 64),
                                   padding: .same,
                                   activation: lrelu,
                                   filterInitializer: heNormal()),
                            enabled: enableSN)
        lastDense = SNDense(Dense(inputSize: 4*4*64, outputSize: 1, weightInitializer: heNormal()), enabled: enableSN)
    }
    
    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        var x = input
        guard imageSize > .x4 else {
            x = fromRGBs[6](x)
            x = blocks[6](x)
            x = minibatchStdConcat(x)
            x = lastConv(x)
            x = x.reshaped(to: [x.shape[0], 4*4*64])
            x = lastDense(x)
            return x
        }
        
        let blockCount = withoutDerivative(at: blocks.count)
        let startIndex = blockCount + 1 - imageSize.log2
        
        var x2 = x
        x = fromRGBs[startIndex](x)
        x = blocks[startIndex](x)
        x = avgPool(x)
        
        if alpha < 1 {
            x2 = avgPool(x2)
            x2 = fromRGBs[startIndex+1](x2)
            x = lerp(x2, x, rate: alpha)
        }
        
        x = blocks[startIndex+1](x)
        
        for i in startIndex+2..<blockCount {
            x = avgPool(x)
            x = blocks[i](x)
        }
        
        x = minibatchStdConcat(x)
        x = lastConv(x)
        x = x.reshaped(to: [x.shape[0], 4*4*64])
        x = lastDense(x)
        return x
    }
}
