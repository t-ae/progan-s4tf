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

@differentiable(vjp: vjpResize2xBilinear)
public func resize2xBilinear(images: Tensor<Float>) -> Tensor<Float> {
    let newHeight = images.shape[1] * 2
    let newWidth = images.shape[2] * 2
    return _Raw.resizeBilinear(images: images,
                              size: Tensor([Int32(newHeight), Int32(newWidth)]),
                              alignCorners: true)
}

public func vjpResize2xBilinear(images: Tensor<Float>) -> (Tensor<Float>, (Tensor<Float>)->Tensor<Float>) {
    let resized = resize2xBilinear(images: images)
    return (resized, { v in
        _Raw.resizeBilinearGrad(grads: v, originalImage: images, alignCorners: true)
    })
}

public struct MinibatchStdConcat<Scalar: TensorFlowFloatingPoint>: ParameterlessLayer {
    @noDerivative
    public let groupSize: Int
    
    public init(groupSize: Int) {
        self.groupSize = groupSize
    }
    
    @differentiable
    public func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
        precondition(input.rank == 4)
        
        let (b, h, w, c) = (input.shape[0], input.shape[1], input.shape[2], input.shape[3])
        precondition(b.isMultiple(of: groupSize), "Not divisible by `groupSize`: \(b) / \(groupSize)")
        
        var x = input.reshaped(to: [groupSize, b/groupSize, h, w, c])
        let mean = x.mean(alongAxes: 0)
        let variance = squaredDifference(x, mean).mean(alongAxes: 0)
        let std = sqrt(variance + 1e-8) // [1, b/groupSize, h, w, c]
        x = std.mean(alongAxes: 2, 3, 4) // [1, b/groupSize, 1, 1, 1]
        x = x.tiled(multiples: Tensor<Int32>([Int32(groupSize), 1, Int32(h), Int32(w), 1]))
        x = x.reshaped(to: [b, h, w, 1])
        
        return input.concatenated(with: x, alongAxis: 3)
    }
}
