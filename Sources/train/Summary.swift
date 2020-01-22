import Foundation
import TensorBoardX
import ProGANModel

extension GBlock: HistogramWritable {
    public func writeHistograms(tag: String, writer: SummaryWriter, globalStep: Int?) {
        conv1.conv.writeHistograms(tag: "\(tag).conv1", writer: writer, globalStep: globalStep)
        conv2.conv.writeHistograms(tag: "\(tag).conv1", writer: writer, globalStep: globalStep)
    }
}

extension Generator {
    public func writeHistograms(writer: SummaryWriter, globalStep: Int) {
        let tag = "G_\(imageSize.name)"
        head.dense.writeHistograms(tag: "\(tag)/head",
            writer: writer, globalStep: globalStep)
        let endIndex = imageSize.log2 - 2
        for i in 0...endIndex {
            blocks[i].writeHistograms(tag: "\(tag)/blocks[\(i)]",
                writer: writer, globalStep: globalStep)
        }
        toRGBs[endIndex].conv.writeHistograms(tag: "\(tag)/toRGBs[\(endIndex)]",
            writer: writer, globalStep: globalStep)
        if endIndex > 0 {
            toRGBs[endIndex-1].conv.writeHistograms(tag: "\(tag)/toRGBs[\(endIndex-1)]",
                writer: writer, globalStep: globalStep)
        }
    }
}

extension DBlock: HistogramWritable {
    public func writeHistograms(tag: String, writer: SummaryWriter, globalStep: Int?) {
        conv1.conv.writeHistograms(tag: "\(tag).conv1", writer: writer, globalStep: globalStep)
        conv2.conv.writeHistograms(tag: "\(tag).conv1", writer: writer, globalStep: globalStep)
    }
}

extension Discriminator {
    public func writeHistograms(writer: SummaryWriter, globalStep: Int) {
        let tag = "D_\(imageSize.name)"
        let startIndex = 8 - imageSize.log2
        
        fromRGBs[startIndex].conv.writeHistograms(tag: "\(tag)/fromRGBs[\(startIndex)]",
            writer: writer, globalStep: globalStep)
        if startIndex < fromRGBs.count-1 {
            fromRGBs[startIndex+1].conv.writeHistograms(tag: "\(tag)/fromRGBs[\(startIndex+1)]",
                writer: writer, globalStep: globalStep)
        }
        
        for i in startIndex..<blocks.count {
            blocks[i].writeHistograms(tag: "\(tag)/blocks[\(i)]",
                writer: writer, globalStep: globalStep)
        }
        
        lastConv.conv.writeHistograms(tag: "\(tag)/lastConv",
            writer: writer, globalStep: globalStep)
        lastDense.dense.writeHistograms(tag: "\(tag)/lastDense",
            writer: writer, globalStep: globalStep)
    }
}
