import Foundation
import Python
import TensorFlow
import TensorBoardX
import ProGANModel

var generator = Generator()
var discriminator = Discriminator()

var optG = Adam(for: generator, learningRate: 1e-3, beta1: 0)
var optD = Adam(for: discriminator, learningRate: 1e-3, beta1: 0)

func grow() {
    generator.grow()
    discriminator.grow()
    
    optG = Adam(for: generator, learningRate: Config.generatorLearningRate, beta1: 0)
    optD = Adam(for: discriminator, learningRate: Config.discriminatorLearningRate, beta1: 0)
}

let imageLoader = try ImageLoader(imageDirectory: Config.imageDirectory)

func train(minibatch: Tensor<Float>) -> (lossG: Tensor<Float>, lossD: Tensor<Float>){
    Context.local.learningPhase = .training
    let minibatchSize = minibatch.shape[0]
    // Update generator
    let noise1 = sampleNoise(size: minibatchSize)
    
    let (lossG, 𝛁generator) = generator.valueWithGradient { generator ->Tensor<Float> in
        let images = generator(noise1)
        let logits = discriminator(images)
        return Config.loss.generatorLoss(fake: logits)
    }
    optG.update(&generator.allDifferentiableVariables, along: 𝛁generator)
    
    // Update discriminator
    let noise2 = sampleNoise(size: minibatchSize)
    let fakeImages = generator(noise2)
    let (lossD, 𝛁discriminator) = discriminator.valueWithGradient { discriminator -> Tensor<Float> in
        let realLogits = discriminator(minibatch)
        let fakeLogits = discriminator(fakeImages)
        return Config.loss.discriminatorLoss(real: realLogits, fake: fakeLogits)
    }
    optD.update(&discriminator.allDifferentiableVariables, along: 𝛁discriminator)
    
    return (lossG, lossD)
}

// Plot
let writer = SummaryWriter(logdir: Config.tensorboardOutputDirectory, flushSecs: 10)

// Test
let testNoise = sampleNoise(size: 64)
func infer(level: Int, step: Int) {
    print("infer...")
    Context.local.learningPhase = .inference
    
    var images = generator(testNoise)
    images = images.padded(forSizes: [(0, 0), (1, 1), (1, 1), (0, 0)], with: 0)
    let (height, width) = (images.shape[1], images.shape[2])
    images = images.reshaped(to: [8, 8, height, width, 3])
    images = images.transposed(withPermutations: [0, 2, 1, 3, 4])
    images = images.reshaped(to: [8*height, 8*width, 3])
    
    // [0, 1] range
    images = (images + 1) / 2
    images = images.clipped(min: Tensor(0), max: Tensor(1))
    
    writer.addImage(tag: "lv\(level)", image: images, globalStep: step)
}

enum Phase {
    case fading, stabilizing
}

var phase: Phase = .stabilizing
var imageCount = 0

for step in 1... {
    if phase == .fading {
        GlobalState.alpha = Float(imageCount) / Float(Config.numImagesPerPhase)
    }
    print("step: \(step), alpha: \(GlobalState.alpha)")
    
    let level = generator.level
    
    let minibatchSize = Config.minibatchSizeSchedule[level - 1]
    let imageSize = 2 * Int(powf(2, Float(level)))

    let minibatch = measureTime(label: "minibatch load") {
        imageLoader.minibatch(size: minibatchSize, imageSize: (imageSize, imageSize))
    }
    
    let (lossG, lossD) = measureTime(label: "train") {
        train(minibatch: minibatch)
    }
    
    writer.addScalar(tag: "lv\(level)/lossG", scalar: lossG.scalar!, globalStep: step)
    writer.addScalar(tag: "lv\(level)/lossD", scalar: lossD.scalar!, globalStep: step)
    
    imageCount += minibatchSize
    
    if imageCount >= Config.numImagesPerPhase {
        imageCount = 0
        
        switch (phase, level) {
        case (.fading, _):
            phase = .stabilizing
            GlobalState.alpha = 1
            print("Start stabilizing lv: \(generator.level)")
        case (.stabilizing, Config.maxLevel):
            break
        case (.stabilizing, _):
            phase = .fading
            GlobalState.alpha = 0
            grow()
            print("Start fading lv: \(generator.level)")
        }
    }
    
    if step.isMultiple(of: Config.numStepsToInfer) {
        infer(level: level, step: step)
    }
}
