package az

import az.agents.MctsAgent
import az.agents.RolloutSearchAgent
import az.net.OnnxNet
import games.planetwars.agents.PlanetWarsAgent
import games.planetwars.core.GameParamGenerator
import games.planetwars.core.Player
import games.planetwars.runners.GameRunner

/**
 * Promotion gate for the AlphaZero loop: candidate = net-guided MCTS loaded from an ONNX FILE,
 * baseline = the current best (SFP rollout search). Plays N randomized-param games, both seats,
 * and prints a parseable winrate line for the orchestrator.
 *
 * Run: gradlew :app:runGate -Pargs="<onnxPath> <games> [budgetMs]"
 * Or:  java -cp client-server.jar az.RunGateKt "runs/gen1/model.onnx 30 20"
 */
fun main(args: Array<String>) {
    val parts = (args.getOrNull(0) ?: "").trim().split(Regex("\\s+")).filter { it.isNotEmpty() }
    val onnxPath = parts.getOrNull(0) ?: run { println("GATE_ERROR usage: <onnxPath> <games> [budgetMs]"); return }
    val games = parts.getOrNull(1)?.toIntOrNull() ?: 20
    val budgetMs = parts.getOrNull(2)?.toLongOrNull() ?: 20L

    val net = OnnxNet.fromFile(onnxPath) ?: run { println("GATE_ERROR no onnx at $onnxPath"); return }

    var wins = 0; var losses = 0; var draws = 0
    for (g in 0 until games) {
        val params = GameParamGenerator.randomParams(7000L + g)
        val meFirst = g % 2 == 0
        val cand: PlanetWarsAgent = MctsAgent(budgetMs = budgetMs, maxCandidates = 16, preloadedNet = net)
        val base: PlanetWarsAgent = RolloutSearchAgent(budgetMs = budgetMs, rolloutTicks = 120, maxCandidates = 16)
        val a1 = if (meFirst) cand else base
        val a2 = if (meFirst) base else cand
        val fm = GameRunner(a1, a2, params).runGame()
        val mySeat = if (meFirst) Player.Player1 else Player.Player2
        when (fm.getLeader()) {
            mySeat -> wins++
            Player.Neutral -> draws++
            else -> losses++
        }
    }
    val wr = wins.toDouble() / games
    println("GATE_RESULT onnx=$onnxPath games=$games wins=$wins losses=$losses draws=$draws winrate=%.4f".format(wr))
}
