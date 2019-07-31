import XCTest

#if !canImport(ObjectiveC)
public func allTests() -> [XCTestCaseEntry] {
    return [
        testCase(ProGANModelTests.allTests),
        testCase(ComponentTests.allTests)
    ]
}
#endif
