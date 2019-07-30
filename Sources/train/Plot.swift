import Foundation
import Python
import TensorFlow

class Plot {
    var outputFolder: String
    
    let np = Python.import("numpy")
    let matplotlib = Python.import("matplotlib")
    let plt = Python.import("matplotlib.pyplot")
    
    init(outputFolder: String) {
        self.outputFolder = outputFolder
        matplotlib.use("Agg")
    }
    
    func plotImage(_ image: Tensor<Float>, name: String) {
        // Create figure.
        let ax = plt.gca()
        let array = np.array([image.scalars])
        let pixels = array.reshape(image.shape)
        if !FileManager.default.fileExists(atPath: outputFolder) {
            try! FileManager.default.createDirectory(
                atPath: outputFolder,
                withIntermediateDirectories: false,
                attributes: nil)
        }
        ax.imshow(pixels)
        plt.savefig("\(outputFolder)/\(name).png", dpi: 300)
        plt.close()
    }
}
