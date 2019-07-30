import Foundation
import PGGANModel

func debugPrint(_ text: String) {
    guard Config.debugPrint else {
        return
    }
    print(text)
}

func measureTime(label: String, f: ()->Void) {
    let start = Date()
    
    f()
    
    debugPrint("\(label): \(Date().timeIntervalSince(start))sec")
}
