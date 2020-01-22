import TensorFlow
import XCTest
import ProGANModel

final class ProGANModelTests: XCTestCase {
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
        imagesPerPhase: 80_000
    )
    
    func testGeneratorSize() {
        var gen = Generator(config: config)
        
        let noise = sampleNoise(size: 8, latentSize: config.latentSize)
        
        for size in ImageSize.allCases {
            gen.imageSize = size
            let image = gen(noise)
            XCTAssertEqual(image.shape, [8, size.rawValue, size.rawValue, 3])
        }
    }
    
    func testDiscriminatorSize() {
        var dis = Discriminator(config: config)
                
        for size in ImageSize.allCases {
            dis.imageSize = size
            let image = Tensor<Float>(zeros: [8, size.rawValue, size.rawValue, 3])
            let logits = dis(image)
            XCTAssertEqual(logits.shape, [8, 1])
        }
    }
    
    func testGeneratorGrow() {
        var generator = Generator(config: config)
        let noise = sampleNoise(size: 10, latentSize: config.latentSize)
        
        var small = generator(noise)
        
        for size in ImageSize.allCases.dropFirst() {
            generator.imageSize = size
            generator.alpha = 0
            let large = generator(noise)
            
            XCTAssertTrue(resize2xBilinear(images: small).isAlmostEqual(to: large))
            
            generator.alpha = 1
            small = generator(noise)
        }
    }
    
    func testDiscriminatorGrow() {
        // If .training, SN will change.
        Context.local.learningPhase = .inference
        
        var discriminator = Discriminator(config: config)
        
        var score1 = discriminator(Tensor<Float>(ones: [8, 4, 4, 3]))
        
        for size in ImageSize.allCases.dropFirst() {
            discriminator.imageSize = size
            discriminator.alpha = 0
            
            let image = Tensor<Float>(ones: [8, size.rawValue, size.rawValue, 3])
            
            let score2 = discriminator(image)
            
            XCTAssertTrue(score1.isAlmostEqual(to: score2))
            
            discriminator.alpha = 1
            score1 = discriminator(image)
        }
    }
    
    func testGeneratorDifferentiability() {
        var gen = Generator(config: config)
        gen.imageSize = .x16
        
        let df: (Tensor<Float>)->Tensor<Float> = gradient { x in
            gen(x).sum()
        }
        
        let noise = sampleNoise(size: 1, latentSize: config.latentSize)
        let grad = df(noise)
        XCTAssertEqual(grad.shape, noise.shape)
        
        let dgen = gradient(at: gen) { gen in
            gen(noise).sum()
        }
        print(dgen.allKeyPaths)
    }
    
    func testDiscriminatorDifferentiability() {
        var dis = Discriminator(config: config)
        
        let size = ImageSize.x16
        dis.imageSize = size
        
        let df: (Tensor<Float>)->Tensor<Float> = gradient { x in
            dis(x).sum()
        }
        
        let image = Tensor<Float>(zeros: [8, size.rawValue, size.rawValue, 3])
        let grad = df(image)
        XCTAssertEqual(grad.shape, image.shape)
        
        let ddis = gradient(at: dis) { dis in
            dis(image).sum()
        }
        print(ddis.allKeyPaths)
    }
    
    func testGeneratorTrainability() {
        var gen = Generator(config: config)
        var dis = Discriminator(config: config)
        
        gen.imageSize = .x16
        dis.imageSize = .x16
        
        let opt = Adam(for: gen)
        let noise = sampleNoise(size: 8, latentSize: config.latentSize)
        
        let loss = GANLoss(type: config.loss)
        
        let dgen = gradient(at: gen) { gen -> Tensor<Float> in
            let images = gen(noise)
            let logits = dis(images)
            return loss.lossG(logits)
        }
        opt.update(&gen, along: dgen)
    }
    
    func testDiscriminatorTrainability() {
        var gen = Generator(config: config)
        var dis = Discriminator(config: config)
        
        gen.imageSize = .x16
        dis.imageSize = .x16
        
        let opt = Adam(for: dis)
        let noise = sampleNoise(size: 8, latentSize: config.latentSize)
        let fakeImages = gen(noise)
        let realImages = Tensor<Float>(zeros: fakeImages.shape)
        
        let loss = GANLoss(type: config.loss)
        
        let ddis = gradient(at: dis) { dis -> Tensor<Float> in
            let realLogits = dis(realImages)
            let fakeLogits = dis(fakeImages)
            return loss.lossD(real: realLogits, fake: fakeLogits)
        }
        opt.update(&dis, along: ddis)
    }
    
    static let allTests = [
        ("testGeneratorSize", testGeneratorSize),
        ("testDiscriminatorSize", testDiscriminatorSize),
        ("testGeneratorGrow", testGeneratorGrow),
        ("testDiscriminatorGrow", testDiscriminatorGrow),
        ("testGeneratorDifferentiability", testGeneratorDifferentiability),
        ("testDiscriminatorDifferentiability", testDiscriminatorDifferentiability),
        ("testGeneratorTrainability", testGeneratorTrainability),
        ("testDiscriminatorTrainability", testDiscriminatorTrainability)
    ]
}
