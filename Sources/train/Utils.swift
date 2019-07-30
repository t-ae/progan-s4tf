import Foundation
import PGGANModel

func debugPrint(_ text: String) {
    guard Config.debugPrint else {
        return
    }
    print(text)
}
