import Foundation
import Python
import TensorFlow
import ImageLoader
import TensorBoardX
import ProGANModel

let config = Config(
    latentSize: 256,
    normalizeLatent: true,
    enableSpectralNorm: GDPair(G: false, D: true),
    useTanhOutput: false,
    loss: .hinge,
    learningRates: GDPair(G: 1e-3, D: 1e-3),
    startSize: .x4,
    endSize: .x256,
    batchSizes: [
        .x4: 16,
        .x8: 16,
        .x16: 16,
        .x32: 16,
        .x64: 16,
        .x128: 16,
        .x256: 16,
    ],
    imagesPerPhase: 800_000
)

var generator = Generator(config: config)
var discriminator = Discriminator(config: config)

var optG = Adam(for: generator, learningRate: config.learningRates.G, beta1: 0, beta2: 0.99)
var optD = Adam(for: discriminator, learningRate: config.learningRates.D, beta1: 0, beta2: 0.99)

// MARK: - Dataset
let args = ProcessInfo.processInfo.arguments
guard args.count == 2 else {
    print("Image directory is not specified.")
    exit(1)
}
print("Seaerch images...")
let imageDir = URL(fileURLWithPath: args[1])
let entries = [Entry](directory: imageDir)
print("\(entries.count) images found")

let criterion = GANLoss(type: config.loss)

// Plot
let logdir = URL(fileURLWithPath: "./logdir")
let writer = SummaryWriter(logdir: logdir)
func plotImages(tag: String, images: Tensor<Float>,
                colSize: Int = 8,  globalStep: Int) {
    var images = images
    images = (images + 1) / 2
    images = images.clipped(min: 0, max: 1)
    writer.addImages(tag: tag,
                     images: images,
                     colSize: colSize,
                     globalStep: globalStep)
}
try writer.addJSONText(tag: "config", encodable: config)

// Test
let testRandomNoises = (0..<4).map { _ in sampleNoise(size: 64, latentSize: config.latentSize) }
let testIntplNoises = (0..<4).map { _ in sampleGridNoise(gridSize: 8, latentSize: config.latentSize) }

enum Phase {
    case fading, stabilizing
}

func train(imageSize: ImageSize, phase: Phase) {
    let tag = "\(imageSize.name)_\(phase)"
    let batchSize = config.batchSizes[imageSize]!
    let numberOfSteps = config.imagesPerPhase / batchSize
    
    generator.imageSize = imageSize
    discriminator.imageSize = imageSize
    if phase == .stabilizing {
        generator.alpha = 1
        discriminator.alpha = 1
    }
    
    let loader = ImageLoader(
        entries: entries,
        transforms: [
            Transforms.paddingToSquare(with: 1),
            Transforms.resize(.area, width: imageSize.rawValue, height: imageSize.rawValue),
            Transforms.randomFlipHorizontally()
        ]
    )
    
    print("Start training.")
    Context.local.learningPhase = .training
    var step = 0
    loop: while true {
        loader.shuffle()
        
        for batch in loader.iterator(batchSize: batchSize) {
            if step % 10 == 0 {
                print("imageSize:\(imageSize), phase:\(phase), step:\(step) / \(numberOfSteps)")
            }
            
            let reals = 2 * batch.images - 1
            
            if phase == .fading {
                let alpha = Float(step) / Float(numberOfSteps)
                generator.alpha = alpha
                discriminator.alpha = alpha
            }
            
            let noise = sampleNoise(size: batchSize, latentSize: config.latentSize)
            
            // Update Discriminator
            let 𝛁discriminator = gradient(at: discriminator) { discriminator -> Tensor<Float> in
                let fakes = generator(noise)
                let realScores = discriminator(reals)
                let fakeScores = discriminator(fakes)
                
                let loss = criterion.lossD(real: realScores, fake: fakeScores)
                
                writer.addScalar(tag: "\(tag)_D/loss", scalar: loss.scalarized(), globalStep: step)
                if step % 100 == 0 {
                    plotImages(tag: "\(tag)/reals", images: reals, globalStep: step)
                    plotImages(tag: "\(tag)/fakes", images: fakes, globalStep: step)
                    writer.flush()
                }
                
                return loss
            }
            optD.update(&discriminator, along: 𝛁discriminator)
            
            // Update Generator
            let 𝛁generator = gradient(at: generator) { generator ->Tensor<Float> in
                let fakes = generator(noise)
                let scores = discriminator(fakes)
                
                let loss = criterion.lossG(scores)
                
                writer.addScalar(tag: "\(tag)_G/loss", scalar: loss.scalarized(), globalStep: step)
                
                return loss
            }
            optG.update(&generator, along: 𝛁generator)
            
            if step % 1000 == 0 {
                generator.writeHistograms(writer: writer, globalStep: step)
                discriminator.writeHistograms(writer: writer, globalStep: step)
            }
            
            step += 1
            guard step < numberOfSteps else {
                break loop
            }
        }
    }
    
    // Inference
    print("Training end. Start inference.")
    Context.local.learningPhase = .inference
    
    for (i, noise) in testRandomNoises.enumerated() {
        let fakes = generator(noise)
        plotImages(tag: "\(tag)/result_fakes", images: fakes, globalStep: i)
    }
    for (i, noise) in testIntplNoises.enumerated() {
        let fakes = generator(noise)
        plotImages(tag: "\(tag)/result_fakes", images: fakes, globalStep: i)
    }
}

train(imageSize: config.startSize, phase: .stabilizing)
for size in ImageSize.allCases {
    guard size > config.startSize else {
        continue
    }
    guard size <= config.endSize else {
        break
    }
    train(imageSize: size, phase: .fading)
    train(imageSize: size, phase: .stabilizing)
}
