import TensorFlow
import XCTest
import ProGANModel

final class ProGANModelTests: XCTestCase {
    func testGeneratorSize() {
        var gen = Generator()
        
        let noise = sampleNoise(size: 10)
        
        var size = 4
        
        for _ in 0..<7 {
            let image = gen(noise)
            XCTAssertEqual(image.shape, [10, size, size, 3])
            
            if gen.level < 7 {
                gen.grow()
                size *= 2
            }
        }
    }
    
    func testDiscriminatorSize() {
        var dis = Discriminator()
        
        var size = 4
                
        for _ in 0..<7 {
            let image = Tensor<Float>(zeros: [8, size, size, 3])
            let logits = dis(image)
            XCTAssertEqual(logits.shape, [8, 1])
            
            if dis.level < 7 {
                dis.grow()
                size *= 2
            }
        }
    }
    
    func testGeneratorGrow() {
        var generator = Generator()
        let noise = sampleNoise(size: 10)
        
        var small = generator(noise)
        
        for _ in 0..<6 {
            generator.grow()
            generator.alpha = 0
            let large = generator(noise)
            
            XCTAssertTrue(resize2xBilinear(images: small).isAlmostEqual(to: large))
            
            generator.alpha = 1
            small = generator(noise)
        }
    }
    
    func testDiscriminatorGrow() {
        var discriminator = Discriminator()
        
        var score1 = discriminator(Tensor<Float>(ones: [4, 4, 4, 3]))
        
        for _ in 0..<6 {
            discriminator.grow()
            discriminator.alpha = 0
            let size = Int(pow(2.0, Double(discriminator.level)) * 2)
            
            let image = Tensor<Float>(ones: [4, size, size, 3])
            
            let score2 = discriminator(image)
            
            XCTAssertTrue(score1.isAlmostEqual(to: score2))
            
            discriminator.alpha = 1
            score1 = discriminator(image)
        }
    }
    
    func testGeneratorDifferentiability() {
        var gen = Generator()
        for _ in 0..<3 {
            gen.grow()
        }
        
        let df: (Tensor<Float>)->Tensor<Float> = gradient { x in
            gen(x).sum()
        }
        
        let noise = sampleNoise(size: 10)
        let grad = df(noise)
        XCTAssertEqual(grad.shape, noise.shape)
    }
    
    func testDiscriminatorDifferentiability() {
        var dis = Discriminator()
        var size = 4
        for _ in 0..<3 {
            dis.grow()
            size *= 2
        }
        
        let df: (Tensor<Float>)->Tensor<Float> = gradient { x in
            dis(x).sum()
        }
        
        let image = Tensor<Float>(zeros: [8, size, size, 3])
        let grad = df(image)
        XCTAssertEqual(grad.shape, image.shape)
    }
    
    func testGeneratorTrainability() {
        var gen = Generator()
        var dis = Discriminator()
        
        for _ in 0..<3 {
            gen.grow()
            dis.grow()
        }
        
        let opt = Adam(for: gen)
        let noise = sampleNoise(size: 8)
        
        let loss = Config.loss.createLoss()
        
        let dgen = gen.gradient { gen -> Tensor<Float> in
            let images = gen(noise)
            let logits = dis(images)
            return loss.generatorLoss(fake: logits)
        }
        opt.update(&gen, along: dgen)
    }
    
    func testDiscriminatorTrainability() {
        var gen = Generator()
        var dis = Discriminator()
        
        for _ in 0..<3 {
            gen.grow()
            dis.grow()
        }
        
        let opt = Adam(for: dis)
        let noise = sampleNoise(size: 8)
        let fakeImages = gen(noise)
        let realImages = Tensor<Float>(zeros: fakeImages.shape)
        
        let loss = Config.loss.createLoss()
        
        let ddis = dis.gradient { dis -> Tensor<Float> in
            let realLogits = dis(realImages)
            let fakeLogits = dis(fakeImages)
            return loss.discriminatorLoss(real: realLogits, fake: fakeLogits)
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
