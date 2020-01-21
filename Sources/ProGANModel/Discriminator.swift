import Foundation
import TensorFlow

struct DBlock: Layer {
    var conv1: SNConv2D<Float>
    var conv2: SNConv2D<Float>
    
    init(inputChannels: Int, outputChannels: Int) {
        conv1 = SNConv2D(Conv2D(filterShape: (3, 3, inputChannels, outputChannels),
                                padding: .same,
                                activation: lrelu))
        conv2 = SNConv2D(Conv2D(filterShape: (3, 3, outputChannels, outputChannels),
                                padding: .same,
                                activation: lrelu))
    }
    
    @differentiable
    func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        var x = input
        x = conv1(x)
        x = conv2(x)
        return x
    }
}

public struct Discriminator: Layer {
    var fromRGBs: [SNConv2D<Float>] = []
    
    var blocks: [DBlock] = []
    
    
    var minibatchStdConcat: MinibatchStdConcat<Float>
    
    var lastConv: SNConv2D<Float>
    var lastDense: SNDense<Float>
    
    var avgPool = AvgPool2D<Float>(poolSize: (2, 2), strides: (2, 2))
    
    @noDerivative
    public private(set) var level = 1
    
    @noDerivative
    public var imageSize: ImageSize = .x4
    
    @noDerivative
    public var alpha: Float = 1.0
    
    public init(config: Config) {
        fromRGBs = [
            SNConv2D(Conv2D(filterShape: (1, 1, 3, 8), activation: lrelu)), // 256x256
            SNConv2D(Conv2D(filterShape: (1, 1, 3, 16), activation: lrelu)), // 128x128
            SNConv2D(Conv2D(filterShape: (1, 1, 3, 32), activation: lrelu)), // 64x64
            SNConv2D(Conv2D(filterShape: (1, 1, 3, 64), activation: lrelu)), // 32x32
            SNConv2D(Conv2D(filterShape: (1, 1, 3, 128), activation: lrelu)), // 16x16
            SNConv2D(Conv2D(filterShape: (1, 1, 3, 256), activation: lrelu)), // 8x8
            SNConv2D(Conv2D(filterShape: (1, 1, 3, 256), activation: lrelu)), // 4x4
        ]
        blocks = [
            DBlock(inputChannels: 8, outputChannels: 16), // 256x256
            DBlock(inputChannels: 16, outputChannels: 32), // 128x128
            DBlock(inputChannels: 32, outputChannels: 64), // 64x64
            DBlock(inputChannels: 64, outputChannels: 128), // 32x32
            DBlock(inputChannels: 128, outputChannels: 256), // 16x16
            DBlock(inputChannels: 256, outputChannels: 256), // 8x8
            DBlock(inputChannels: 256, outputChannels: 256), // 4x4
        ]
        
        minibatchStdConcat = MinibatchStdConcat(groupSize: 4)
        lastConv = SNConv2D(Conv2D(filterShape: (3, 3, 257, 64), activation: lrelu))
        lastDense = SNDense(Dense(inputSize: 4*4*64, outputSize: 1 ))
    }
    
    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        var x = input
        guard imageSize > .x4 else {
            x = fromRGBs[6](x)
            x = blocks[6](x)
            x = minibatchStdConcat(x)
            x = lastConv(x)
            x = x.reshaped(to: [-1, 4*4*64])
            return lastDense(x)
        }
        
        let startIndex = 8 - imageSize.log2
        
        let x2 = fromRGBs[startIndex+1].callAsFunction(avgPool(x))
        x = fromRGBs[startIndex](x)
        x = blocks[startIndex](x)
        x = avgPool(x)
        
        x = lerp(x2, x, rate: alpha)
        
        x = blocks[startIndex+1](x)
        
        let blockCount = withoutDerivative(at: blocks.count)
        for i in startIndex+2..<blockCount {
            x = avgPool(x)
            x = blocks[i](x)
        }
        
        x = minibatchStdConcat(x)
        x = lastConv(x)
        x = x.reshaped(to: [-1, 4*4*64])
        return x
    }
}
