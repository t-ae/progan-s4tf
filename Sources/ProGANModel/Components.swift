import Foundation
import TensorFlow
import GANUtils

public func heNormal<Scalar: TensorFlowFloatingPoint>() -> ParameterInitializer<Scalar> {
    return { shape in
        let out = shape.dimensions.dropLast().reduce(1, *)
        return Tensor(randomNormal: shape) * sqrt(2 / Scalar(out))
    }
}

@differentiable
func lrelu(_ x: Tensor<Float>) -> Tensor<Float> {
    leakyRelu(x)
}

@differentiable
public func resize2xBilinear(images: Tensor<Float>) -> Tensor<Float> {
    let newHeight = images.shape[1] * 2
    let newWidth = images.shape[2] * 2
    
    return resizeBilinear(images: images,
                          width: newWidth,
                          height: newHeight,
                          alignCorners: true)
}

public struct ActivationSelector: ParameterlessLayer {
    public enum Activation {
        case identity, tanh
    }
    public var activation: Activation
    public init(_ activation: Activation) {
        self.activation = activation
    }
    
    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        switch activation {
        case .identity:
            return input
        case .tanh:
            return tanh(input)
        }
    }
}
