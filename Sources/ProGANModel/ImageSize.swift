import Foundation

public enum ImageSize: Int, Codable, CaseIterable {
    case x4 = 4
    case x8 = 8
    case x16 = 16
    case x32 = 32
    case x64 = 64
    case x128 = 128
    case x256 = 256
    
    public var log2: Int {
        return Int(Foundation.log2(Float(rawValue)))
    }
    
    public var name: String {
        "\(rawValue)x\(rawValue)"
    }
}

extension ImageSize: Comparable {
    public static func < (lhs: ImageSize, rhs: ImageSize) -> Bool {
        switch (lhs, rhs) {
        case (.x4, .x8), (.x4, .x16), (.x4, .x32), (.x4, .x64), (.x4, .x128), (.x4, .x256):
            return true
        case (.x8, .x16), (.x8, .x32), (.x8, .x64), (.x8, .x128), (.x8, .x256):
            return true
        case (.x16, .x32), (.x16, .x64), (.x16, .x128), (.x16, .x256):
            return true
        case (.x32, .x64), (.x32, .x128), (.x32, .x256):
            return true
        case (.x64, .x128), (.x64, x256):
            return true
        case (.x128, x256):
            return true
        default:
            return false
        }
    }
}
