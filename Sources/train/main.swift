import Foundation
import Python
import TensorFlow
import ProGANModel

var generator = Generator()
var discriminator = Discriminator()

var optG = Adam(for: generator, learningRate: 1e-3, beta1: 0)
var optD = Adam(for: discriminator, learningRate: 1e-3, beta1: 0)

func grow() {
    generator.grow()
    discriminator.grow()
    
    optG = Adam(for: generator, learningRate: 1e-3, beta1: 0)
    optD = Adam(for: discriminator, learningRate: 1e-3, beta1: 0)
}

func train(minibatch: Tensor<Float>) {
    Context.local.learningPhase = .training
    let minibatchSize = minibatch.shape[0]
    // Update generator
    let noise1 = sampleNoise(size: minibatchSize)
    let 𝛁generator = generator.gradient { generator ->Tensor<Float> in
        let images = generator(noise1)
        let logits = discriminator(images)
        return generatorLoss(fakeLogits: logits)
    }
    optG.update(&generator.allDifferentiableVariables, along: 𝛁generator)
    
    // Update discriminator
    let noise2 = sampleNoise(size: minibatchSize)
    let fakeImages = generator(noise2)
    let 𝛁discriminator = discriminator.gradient { discriminator -> Tensor<Float> in
        let realLogits = discriminator(minibatch)
        let fakeLogits = discriminator(fakeImages)
        return discriminatorLoss(realLogits: realLogits, fakeLogits: fakeLogits)
    }
    optD.update(&discriminator.allDifferentiableVariables, along: 𝛁discriminator)
}

// Test
let testNoise = sampleNoise(size: 64)
let plot = Plot(outputFolder: Config.imageOutputDirectory)
func infer(imageName: String) {
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
    
    plot.plotImage(images.clipped(min: Tensor(0), max: Tensor(1)), name: imageName)
}

let imageLoader = try ImageLoader(imageDirectory: Config.imageDirectory)

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
    
    measureTime(label: "train") {
        train(minibatch: minibatch)
    }
    
    imageCount += minibatchSize
    
    if imageCount >= Config.numImagesPerPhase {
        imageCount = 0
        
        switch (phase, level) {
        case (.fading, _):
            phase = .stabilizing
            GlobalState.alpha = 1
            print("Start stabilizing lv: \(level)")
        case (.stabilizing, Config.maxLevel):
            break
        case (.stabilizing, _):
            phase = .fading
            GlobalState.alpha = 0
            grow()
            print("Start fading lv: \(level)")
        }
    }
    
    if step.isMultiple(of: Config.numStepsToInfer) {
        let imageName = String(format: "%09d.png", step)
        infer(imageName: imageName)
    }
}
