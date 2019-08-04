import XCTest
import TensorFlow
import ProGANModel

class ComponentTests: XCTestCase {
    
    func testMinibatchStdConcat() {
        let group1 = 0.001 * Tensor<Float>(randomNormal: [4, 4, 4, 3])
        let group2 = 1 * Tensor<Float>(randomNormal: [4, 4, 4, 3])
        let tensor = Tensor(stacking: [group1, group2])
            .transposed(withPermutations: 1, 0, 2, 3, 4)
            .reshaped(to: [8, 4, 4, 3])
        let concat = minibatchStdConcat(tensor)
        
        XCTAssertEqual(concat[0, 0, 0, 3], concat[0, 1, 1, 3])
        XCTAssertEqual(concat[0, 0, 0, 3], concat[2, 1, 1, 3])
        XCTAssertNotEqual(concat[0, 0, 0, 3], concat[1, 1, 1, 3])
    }

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
