public enum GlobalState {
    public static var alpha: Float = 1
    public static var level: Int = 1 {
        didSet {
            precondition(1 <= level && level <= Config.maxLevel)
        }
    }
    
    public static var batchSize = 32
}
