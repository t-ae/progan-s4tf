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
    
    static let allTests = [
        ("testResize2xBilinear", testResize2xBilinear)
    ]
}
