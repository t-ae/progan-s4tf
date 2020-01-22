import XCTest
import TensorFlow
import ProGANModel

class ComponentTests: XCTestCase {
    func testResize2xBilinear() {
        let tensor = Tensor((0..<36).map { Float($0) })
            .reshaped(to: [1, 6, 6, 1])
        let grad = gradient(at: tensor) { tensor in
            resize2xBilinear(images: tensor).sum()
        }
        let grad4x4 = grad.reshaped(to: [6, 6])
        // TODO: assert
        print(grad4x4)
    }
    
    func testPixelNorm() {
        let length = 32
        let tensor = Tensor<Float>(randomNormal: [1, length*length])
        
        let norm = pixelNormalization(tensor)
        
        let len = sqrt(norm.squared().sum())
        
        XCTAssert(len.isAlmostEqual(to: Tensor(Float(length))))
    }
    
    static let allTests = [
        ("testPixelNorm", testPixelNorm)
    ]
}
