import Foundation
import TensorFlow

@differentiable
public func generatorLoss(fakeLogits: Tensor<Float>) -> Tensor<Float> {
//    sigmoidCrossEntropy(logits: fakeLogits,
//                        labels: Tensor(ones: fakeLogits.shape))
    return softplus(-fakeLogits).mean()
}

@differentiable
public func discriminatorLoss(realLogits: Tensor<Float>, fakeLogits: Tensor<Float>) -> Tensor<Float> {
//    let realLoss = sigmoidCrossEntropy(logits: realLogits,
//                                       labels: Tensor(ones: realLogits.shape))
//    let fakeLoss = sigmoidCrossEntropy(logits: fakeLogits,
//                                       labels: Tensor(zeros: fakeLogits.shape))
    
    let realLoss = softplus(-realLogits).mean()
    let fakeLoss = softplus(fakeLogits).mean()
    
    return realLoss + fakeLoss
}
