package az

import az.agents.ChampionAgent
import az.agents.HeuristicAgent
import az.agents.MctsAgent
import az.agents.RheaAgent
import az.agents.RolloutSearchAgent
import competition_entry.GreedyHeuristicAgent
import games.planetwars.agents.evo.SimpleEvoAgent
import games.planetwars.agents.PlanetWarsAgent
import games.planetwars.core.GameParamGenerator
import games.planetwars.core.Player
import games.planetwars.runners.GameRunner
import kotlinx.coroutines.*
import kotlinx.coroutines.sync.Semaphore
import kotlinx.coroutines.sync.withPermit
import java.util.concurrent.atomic.AtomicInteger

/**
 * Head-to-head A/B between two search-agent configs (no net) — the only meaningful metric now that
 * the sample pool is saturated. Games run concurrently across cores.
 *
 * Agent spec: name[:budgetMs[:rolloutTicks[:maxCandidates]]]
 *   name = sfp (RolloutSearchAgent) | mcts (rollout-MCTS) | rhea | heuristic | greedy
 *   for rhea, fields 3/4 are repurposed: rhea:budgetMs:planTicks:tailTicks (e.g. rhea:28:30:120)
 *
 * Run: gradlew :app:runDuel -Pargs="<specA> <specB> <games> <parallelism>"
 *   e.g. "mcts:28:150:20 sfp:28:150:20 40 4"
 * Prints: DUEL_RESULT A=<specA> B=<specB> ... winrateA=<x>
 */
private fun mk(spec: String): PlanetWarsAgent {
    val name = spec.substringBefore(":")
    val rest = spec.substringAfter(":", "")

    // Flexible key=value spec for mcts experiments, e.g.
    //   mcts:b=28,roll=120,cand=28,cpuct=1.4,depth=6,eval=dyn,fpu=0.2,greedy=true
    // (any subset; omitted keys fall back to the known-good current-best defaults).
    if (name == "mcts" && rest.contains("=")) {
        val kv = rest.split(",").filter { it.contains("=") }
            .associate { val (k, v) = it.split("="); k.trim() to v.trim() }
        return MctsAgent(
            budgetMs = kv["b"]?.toLongOrNull() ?: 28L,
            cPuct = kv["cpuct"]?.toDoubleOrNull() ?: 1.4,
            maxDepth = kv["depth"]?.toIntOrNull() ?: 6,
            rolloutTicks = kv["roll"]?.toIntOrNull() ?: 120,
            maxCandidates = kv["cand"]?.toIntOrNull() ?: 28,
            evalPreset = kv["eval"] ?: "base",
            fpuReduction = kv["fpu"]?.toDoubleOrNull() ?: Double.NaN,
            ensureGreedy = kv["greedy"]?.toBooleanStrictOrNull() ?: false,
        )
    }

    val p = spec.split(":")
    val budget = p.getOrNull(1)?.toLongOrNull() ?: 28L
    val rt = p.getOrNull(2)?.toIntOrNull() ?: 150
    val cand = p.getOrNull(3)?.toIntOrNull() ?: 20
    val cpuct = p.getOrNull(4)?.toDoubleOrNull() ?: 1.4   // mcts only
    val depth = p.getOrNull(5)?.toIntOrNull() ?: 6        // mcts only
    val evalPreset = p.getOrNull(6) ?: "base"             // mcts only
    return when (name) {
        "sfp" -> RolloutSearchAgent(budgetMs = budget, rolloutTicks = rt, maxCandidates = cand)
        "mcts" -> MctsAgent(budgetMs = budget, cPuct = cpuct, maxDepth = depth, rolloutTicks = rt, maxCandidates = cand, evalPreset = evalPreset)
        "rhea" -> RheaAgent(budgetMs = budget, planTicks = rt, tailTicks = cand)
        "champion" -> ChampionAgent()
        "heuristic" -> HeuristicAgent()
        "greedy" -> GreedyHeuristicAgent()
        "simpleevo" -> SimpleEvoAgent()
        else -> error("unknown agent spec: $spec")
    }
}

fun main(args: Array<String>) {
    val parts = (args.getOrNull(0) ?: "").trim().split(Regex("\\s+")).filter { it.isNotEmpty() }
    val specA = parts.getOrNull(0) ?: run { println("usage: <specA> <specB> <games> [parallelism]"); return }
    val specB = parts.getOrNull(1) ?: run { println("usage: <specA> <specB> <games> [parallelism]"); return }
    val games = parts.getOrNull(2)?.toIntOrNull() ?: 40
    val par = parts.getOrNull(3)?.toIntOrNull() ?: 4

    val winsA = AtomicInteger(0); val winsB = AtomicInteger(0); val draws = AtomicInteger(0)
    runBlocking {
        val sem = Semaphore(par.coerceAtLeast(1))
        (0 until games).map { g ->
            async(Dispatchers.Default) {
                sem.withPermit {
                    val params = GameParamGenerator.randomParams(9000L + g)
                    val aFirst = g % 2 == 0
                    val p1 = if (aFirst) mk(specA) else mk(specB)
                    val p2 = if (aFirst) mk(specB) else mk(specA)
                    val fm = GameRunner(p1, p2, params).runGame()
                    val aSeat = if (aFirst) Player.Player1 else Player.Player2
                    when (fm.getLeader()) {
                        aSeat -> winsA.incrementAndGet()
                        Player.Neutral -> draws.incrementAndGet()
                        else -> winsB.incrementAndGet()
                    }
                }
            }
        }.awaitAll()
    }
    val wrA = winsA.get().toDouble() / games
    println("DUEL_RESULT A=$specA B=$specB games=$games winsA=${winsA.get()} winsB=${winsB.get()} draws=${draws.get()} winrateA=%.4f".format(wrA))
}
