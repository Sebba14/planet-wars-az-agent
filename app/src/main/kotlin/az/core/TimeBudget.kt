package az.core

/** Simple wall-clock deadline manager for the per-move search budget. */
class TimeBudget(private val totalMs: Long) {
    private val startNs = System.nanoTime()
    fun elapsedMs(): Double = (System.nanoTime() - startNs) / 1e6
    fun remainingMs(): Double = totalMs - elapsedMs()
    /** True once we are within [marginMs] of the deadline (stop launching new work). */
    fun expired(marginMs: Double = 0.0): Boolean = remainingMs() <= marginMs
}
