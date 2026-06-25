package az

import az.agents.MctsAgent
import az.agents.RolloutSearchAgent
import az.net.OnnxNet
import games.planetwars.agents.PlanetWarsAgent
import games.planetwars.core.GameParamGenerator
import games.planetwars.core.Player
import games.planetwars.runners.GameRunner
import kotlinx.coroutines.*
import kotlinx.coroutines.sync.Semaphore
import kotlinx.coroutines.sync.withPermit
import java.util.concurrent.atomic.AtomicInteger

/**
 * Gate for the AlphaZero loop. Candidate = net-guided MCTS (ONNX file). Opponent is configurable:
 *   sfp | rollout | <path-to-onnx>  (promotion gate vs best/rollout, ship gate vs SFP).
 * Games run CONCURRENTLY (parallelism arg) — the candidate/opponent ONNX sessions are shared and
 * ONNX Runtime is thread-safe for concurrent inference. Designed to run on GPU2's otherwise-idle
 * CPU cores during GPU training (off the Main PC).
 *
 * Run: gradlew :app:runGate -Pargs="<candidateOnnx> <games> <budgetMs> <opponent> <parallelism>"
 * Prints: GATE_RESULT ... winrate=<x>
 */
fun main(args: Array<String>) {
    val parts = (args.getOrNull(0) ?: "").trim().split(Regex("\\s+")).filter { it.isNotEmpty() }
    val onnxPath = parts.getOrNull(0) ?: run { println("GATE_ERROR usage: <candidateOnnx> <games> <budgetMs> <opponent> [parallelism]"); return }
    val games = parts.getOrNull(1)?.toIntOrNull() ?: 20
    val budgetMs = parts.getOrNull(2)?.toLongOrNull() ?: 18L
    val opponent = parts.getOrNull(3) ?: "sfp"
    val parallelism = parts.getOrNull(4)?.toIntOrNull() ?: 4

    val candNet = OnnxNet.fromFile(onnxPath) ?: run { println("GATE_ERROR no onnx at $onnxPath"); return }
    val oppNet = if (opponent != "sfp" && opponent != "rollout") OnnxNet.fromFile(opponent) else null
    if (opponent != "sfp" && opponent != "rollout" && oppNet == null) { println("GATE_ERROR no opponent onnx at $opponent"); return }

    fun mkCandidate(): PlanetWarsAgent = MctsAgent(budgetMs = budgetMs, maxCandidates = 16, preloadedNet = candNet)
    fun mkOpponent(): PlanetWarsAgent = when (opponent) {
        "sfp" -> RolloutSearchAgent(budgetMs = budgetMs, rolloutTicks = 120, maxCandidates = 16)
        "rollout" -> MctsAgent(budgetMs = budgetMs, maxCandidates = 16)
        else -> MctsAgent(budgetMs = budgetMs, maxCandidates = 16, preloadedNet = oppNet)
    }

    val wins = AtomicInteger(0); val losses = AtomicInteger(0); val draws = AtomicInteger(0)
    runBlocking {
        val sem = Semaphore(parallelism.coerceAtLeast(1))
        (0 until games).map { g ->
            async(Dispatchers.Default) {
                sem.withPermit {
                    val params = GameParamGenerator.randomParams(7000L + g)
                    val meFirst = g % 2 == 0
                    val a1 = if (meFirst) mkCandidate() else mkOpponent()
                    val a2 = if (meFirst) mkOpponent() else mkCandidate()
                    val fm = GameRunner(a1, a2, params).runGame()
                    val mySeat = if (meFirst) Player.Player1 else Player.Player2
                    when (fm.getLeader()) {
                        mySeat -> wins.incrementAndGet()
                        Player.Neutral -> draws.incrementAndGet()
                        else -> losses.incrementAndGet()
                    }
                }
            }
        }.awaitAll()
    }
    val wr = wins.get().toDouble() / games
    println("GATE_RESULT onnx=$onnxPath opp=$opponent games=$games par=$parallelism wins=${wins.get()} losses=${losses.get()} draws=${draws.get()} winrate=%.4f".format(wr))
}
