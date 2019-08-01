import Foundation
import TensorFlow

@differentiable
public func lrelu(_ x: Tensor<Float>) -> Tensor<Float> {
    leakyRelu(x)
}

@differentiable(wrt: x)
public func pixelNormalization(_ x: Tensor<Float>, epsilon: Float = 1e-8) -> Tensor<Float> {
    let x2 = x * x
    let mean = x2.mean(alongAxes: 3)
    return x / (sqrt(mean + epsilon))
}

@differentiable(wrt: (a, b))
public func lerp(_ a: Tensor<Float>, _ b: Tensor<Float>, rate: Float) -> Tensor<Float> {
    let rate = min(max(rate, 0), 1)
    return a + rate * (b - a)
}

@differentiable
public func minibatchStdConcat(_ x: Tensor<Float>) -> Tensor<Float> {
    let groupSize = 4
    let batchSize = x.shape[0]
    let height = x.shape[1]
    let width = x.shape[2]
    let M = batchSize / groupSize
    
    // Compute stddev of each pixel in group
    var y = x.reshaped(to: [groupSize, M, -1])
    let mean = y.mean(alongAxes: 0) // [1, M, -1]
    y = (y - mean).squared().mean(alongAxes: 0) // [1, M, -1]
    y = sqrt(y + 1e-8)
    
    y = y.mean(alongAxes: 2) // [1, M, 1]
    y = y.reshaped(to: [M, 1, 1, 1])
    y = y.tiled(multiples: Tensor([Int32(groupSize), Int32(height), Int32(width), 1]))
    return x.concatenated(with: y, alongAxis: 3)
}

public struct WSConv2D: Layer {
    
    public var filter: Tensor<Float>
    public var bias: Tensor<Float>
    
    @noDerivative public let scale: Tensor<Float>
    
    @noDerivative public let padding: Padding
    
    @noDerivative public let activation: Activation
    
    public typealias Activation = @differentiable (Tensor<Float>) -> Tensor<Float>

    public init(inputChannels: Int,
                outputChannels: Int,
                kernelSize: (Int, Int),
                padding: Padding = .same,
                activation: @escaping Activation = identity,
                gain: Float = sqrt(2)) {
        self.filter = Tensor(randomNormal: [kernelSize.0,
                                            kernelSize.1,
                                            inputChannels,
                                            outputChannels])
        self.bias = Tensor(zeros: [outputChannels])
        self.padding = padding
        self.activation = activation
        
        self.scale = Tensor(gain) / sqrt(Float(inputChannels*kernelSize.0*kernelSize.1))
    }
    
    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        return conv2D(input, filter: scale * filter, padding: padding) + bias
    }
}
