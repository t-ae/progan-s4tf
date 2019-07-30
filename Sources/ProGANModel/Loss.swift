import Foundation
import TensorFlow

@differentiable
public func generatorLoss(fakeLogits: Tensor<Float>) -> Tensor<Float> {
    sigmoidCrossEntropy(logits: fakeLogits,
                        labels: Tensor(ones: fakeLogits.shape))
}

@differentiable
public func discriminatorLoss(realLogits: Tensor<Float>, fakeLogits: Tensor<Float>) -> Tensor<Float> {
    let realLoss = sigmoidCrossEntropy(logits: realLogits,
                                       labels: Tensor(ones: realLogits.shape))
    let fakeLoss = sigmoidCrossEntropy(logits: fakeLogits,
                                       labels: Tensor(zeros: fakeLogits.shape))
    return realLoss + fakeLoss
}
