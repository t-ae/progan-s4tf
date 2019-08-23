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
    
    static let allTests = [
        ("testMinibatchStdConcatDifferentiability", testMinibatchStdConcatDifferentiability)
    ]
}
