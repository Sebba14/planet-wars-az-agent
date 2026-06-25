package az

import az.net.OnnxNet
import games.planetwars.core.GameParamGenerator
import games.planetwars.core.GameStateFactory
import games.planetwars.core.Player

/**
 * Smoke test for onnxruntime-java inference of the bundled AlphaZero net.
 * Verifies the model loads from resources, runs, and measures per-inference latency
 * (critical: net eval must be cheap enough to call many times within the 50ms move budget).
 *
 * Run: gradlew :app:runOnnxSmoke
 */
fun main() {
    val net = OnnxNet.fromResource()
    if (net == null) { println("[onnx-smoke] NO MODEL at /az/model.onnx"); return }
    val params = GameParamGenerator.randomParams(5L)
    val state = GameStateFactory(params).createGame()

    repeat(30) { net.infer(state, Player.Player1, params) }  // warmup
    val n = 300
    val t0 = System.nanoTime()
    var acc = 0f
    repeat(n) { acc += net.infer(state, Player.Player1, params).value }
    val msPer = (System.nanoTime() - t0) / 1e6 / n

    val o = net.infer(state, Player.Player1, params)
    println("[onnx-smoke] %.3f ms/infer (steady) | value=%.4f srcLogits=%d tgtLogits=%d".format(msPer, o.value, o.srcLogits.size, o.tgtLogits.size))
    println("[onnx-smoke] src[0..3]=%.3f,%.3f,%.3f,%.3f stop=%.3f".format(o.srcLogits[0], o.srcLogits[1], o.srcLogits[2], o.srcLogits[3], o.srcLogits[o.srcLogits.size - 1]))
    println("[onnx-smoke] OK (acc=$acc)")
    net.close()
}
