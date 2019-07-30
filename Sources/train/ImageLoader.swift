import Foundation
import TensorFlow
import Swim

class ImageLoader {
    var imageDirectory: String
    var fileNames: [String]
    
    var index = 0
    
    var multiThread = true
    
    let appendQueue = DispatchQueue(label: "ImageLoader.appendQueue")
    
    init(imageDirectory: String) throws {
        self.imageDirectory = imageDirectory
        fileNames = try FileManager.default.contentsOfDirectory(atPath: imageDirectory)
    }
    
    func shuffle() {
        fileNames.shuffle()
    }
    
    func resetIndex() {
        index = 0
    }
    
    func minibatch(size: Int, imageSize: (height: Int, width: Int)) -> Tensor<Float> {
        let start = Date()
        defer { debugPrint("minibatch load: \(Date().timeIntervalSince(start))sec") }
        
        if fileNames.count >= index+size {
            resetIndex()
            shuffle()
        }
        
        let imageDir = URL(fileURLWithPath: imageDirectory)
        
        var tensors: [Tensor<Float>]
        let fileNames = self.fileNames[index..<index+size]
        
        if multiThread {
            tensors = []
            DispatchQueue.concurrentPerform(iterations: size) { i in
                let url = imageDir.appendingPathComponent(fileNames[i])
                let image = try! Image<RGB, Float>(contentsOf: url)
                let resized = image.resize(width: imageSize.width, height: imageSize.height)
                
                let tensor = resized.withUnsafeBufferPointer { bp in
                    Tensor<Float>(Array(bp))
                }
                appendQueue.sync {
                    tensors.append(tensor)
                }
            }
        } else {
            let images = fileNames.map { fileName -> Image<RGB, Float> in
                let url = imageDir.appendingPathComponent(fileName)
                let image = try! Image<RGB, Float>(contentsOf: url)
                return image.resize(width: imageSize.width, height: imageSize.height)
            }
            tensors = images.map { image in
                image.withUnsafeBufferPointer { bp in
                    Tensor<Float>(Array(bp))
                }
            }
        }
        
        let tensor = Tensor<Float>(stacking: tensors)
        
        // [-1, 1] range
        return tensor.reshaped(to: [size, imageSize.height, imageSize.width, 3]) * 2 - 1
    }
}
