import Foundation
import TensorFlow
import GANUtils

public func sampleNoise(size: Int, latentSize: Int) -> Tensor<Float> {
    Tensor(randomNormal: [size, latentSize])
}

public func sampleGridNoise(gridSize: Int, latentSize: Int) -> Tensor<Float> {
    let noises = sampleNoise(size: 4, latentSize: latentSize)
    
    let (z0, z1, z2, z3) = (noises[0], noises[1], noises[2], noises[3])
    
    var zs: [Tensor<Float>] = []
    for y in 0..<gridSize {
        let rate = Float(y) / Float(gridSize-1)
        let z02 = lerp(z0, z2, rate: rate)
        let z13 = lerp(z1, z3, rate: rate)
        for x in 0..<gridSize {
            let rate = Float(x) / Float(gridSize-1)
            zs.append(lerp(z02, z13, rate: rate))
        }
    }
    
    
    return Tensor(stacking: zs)
}
