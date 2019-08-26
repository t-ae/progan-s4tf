import Foundation
import TensorFlow

@differentiable
public func lrelu(_ x: Tensor<Float>) -> Tensor<Float> {
    leakyRelu(x)
}

@differentiable(wrt: x)
public func pixelNormalization(_ x: Tensor<Float>, epsilon: Float = 1e-8) -> Tensor<Float> {
    let mean = x.squared().mean(alongAxes: x.shape.count-1)
    return x * rsqrt(mean + epsilon)
}

@differentiable(wrt: (a, b))
public func lerp(_ a: Tensor<Float>, _ b: Tensor<Float>, rate: Float) -> Tensor<Float> {
    let rate = min(max(rate, 0), 1)
    return a + rate * (b - a)
}

@differentiable
public func minibatchStdConcat(_ x: Tensor<Float>) -> Tensor<Float> {
    let batchSize = x.shape[0]
    let height = x.shape[1]
    let width = x.shape[2]
//    let channels = x.shape[3]
    
    // All images
    let mean = x.mean(alongAxes: 0)
    let variance = squaredDifference(x, mean).mean(alongAxes: 0)
    let std = sqrt(variance + 1e-8)
    
    var y = std.mean(alongAxes: 1, 2, 3) // [1, 1, 1, 1]
    y = y.tiled(multiples: Tensor([Int32(batchSize), Int32(height), Int32(width), 1]))
    
    // group version
//    let groupSize = 4
//    let M = batchSize / groupSize
//
//    // Compute stddev of each pixel in group
//    var y = x.reshaped(to: [groupSize, M, height, width, channels])
//    let mean = y.mean(alongAxes: 0) // [1, M, height, width, channels]
//    y = squaredDifference(x, mean).mean(squeezingAxes: 0) // [M, height, width, channels]
//    y = sqrt(y + 1e-8) // stddev
//
//    y = y.mean(alongAxes: 1, 2, 3) // [M, 1, 1, 1]
//    y = y.tiled(multiples: Tensor([Int32(groupSize), Int32(height), Int32(width), 1]))
//    y = y.reshaped(to: [batchSize, height, width, 1])
//
  
    // Concatenation
    // https://bugs.swift.org/browse/TF-705
    //return x.concatenated(with: y, alongAxis: 3)
    // https://bugs.swift.org/browse/TF-706
    // return Tensor(concatenating: [x, y], alongAxis: 3)

    // Dirty hack to avoid the bugs above
    let xs = x.unstacked(alongAxis: 3)
    y = y.squeezingShape(at: 3)
    return Tensor(stacking: xs + [y], alongAxis: 3)
}

@differentiable(vjp: vjpResize2xBilinear)
public func resize2xBilinear(images: Tensor<Float>) -> Tensor<Float> {
    let newHeight = images.shape[1] * 2
    let newWidth = images.shape[2] * 2
    return Raw.resizeBilinear(images: images,
                              size: Tensor([Int32(newHeight), Int32(newWidth)]),
                              alignCorners: true)
}

public func vjpResize2xBilinear(images: Tensor<Float>) -> (Tensor<Float>, (Tensor<Float>)->Tensor<Float>) {
    let resized = resize2xBilinear(images: images)
    return (resized, { v in
        Raw.resizeBilinearGrad(grads: v, originalImage: images, alignCorners: true)
    })
}

public struct EqualizedDense: Layer {
    public typealias Activation = @differentiable (Tensor<Float>) -> Tensor<Float>
    
    public var weight: Tensor<Float>
    public var bias: Tensor<Float>
    
    @noDerivative public let scale: Float
    
    @noDerivative public let activation: Activation
    
    public init(inputSize: Int,
                outputSize: Int,
                activation: @escaping Activation = identity,
                gain: Float = sqrt(2)) {
        self.weight = Tensor<Float>(randomNormal: [inputSize, outputSize])
        self.bias = Tensor<Float>(zeros: [outputSize])
        
        self.scale = gain / sqrt(Float(inputSize))
        
        self.activation = activation
    }
    
    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        activation(matmul(input, weight * scale) + bias)
    }
}

public struct EqualizedConv2D: Layer {
    public typealias Activation = @differentiable (Tensor<Float>) -> Tensor<Float>
    
    public var filter: Tensor<Float>
    public var bias: Tensor<Float>
    @noDerivative public let scale: Float
    
    @noDerivative public let strides: (Int, Int)
    @noDerivative public let padding: Padding
    
    @noDerivative public let activation: Activation
    
    public init(inputChannels: Int,
                outputChannels: Int,
                kernelSize: (Int, Int),
                strides: (Int, Int) = (1, 1),
                padding: Padding = .same,
                activation: @escaping Activation = identity,
                gain: Float = sqrt(2)) {
        self.filter = Tensor<Float>(randomNormal: [kernelSize.0,
                                                   kernelSize.1,
                                                   inputChannels,
                                                   outputChannels])
        self.bias = Tensor<Float>(zeros: [outputChannels])
        
        self.scale = gain / sqrt(Float(inputChannels*kernelSize.0*kernelSize.1))
        
        self.strides = strides
        self.padding = padding
        
        self.activation = activation
    }
    
    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        activation(conv2D(input,
                          filter: filter * scale,
                          strides: (1, strides.0, strides.1, 1),
                          padding: padding,
                          dilations: (1, 1, 1, 1)) + bias)
    }
}
