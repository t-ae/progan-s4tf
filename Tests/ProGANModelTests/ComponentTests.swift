import XCTest
import TensorFlow
import ProGANModel

class ComponentTests: XCTestCase {
    func testMinibatchStdConcatDifferentiability() {
        let df: (Tensor<Float>)->Tensor<Float> = gradient { x in
            minibatchStdConcat(x).sum()
        }
        
        let tensor = Tensor<Float>(randomNormal: [16, 32, 32, 128])
        XCTAssertEqual(df(tensor).shape, tensor.shape)
    }
    
    func testResize2xBilinear() {
        let tensor = Tensor((0..<36).map { Float($0) })
            .reshaped(to: [1, 6, 6, 1])
        let grad = tensor.gradient { tensor in
            resize2xBilinear(images: tensor).sum()
        }
        let grad4x4 = grad.reshaped(to: [6, 6])
        // TODO: assert
        print(grad4x4)
    }
    
    static let allTests = [
        ("testMinibatchStdConcatDifferentiability", testMinibatchStdConcatDifferentiability)
    ]
}
