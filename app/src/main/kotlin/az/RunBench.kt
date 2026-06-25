package az

import az.agents.HeuristicAgent
import games.planetwars.agents.Action
import games.planetwars.core.ForwardModel
import games.planetwars.core.GameParamGenerator
import games.planetwars.core.GameStateFactory
import games.planetwars.core.Player
import kotlin.system.measureNanoTime

/**
 * Forward-model speed benchmark — gates the SFP search budget (how many rollout ticks
 * fit in the ~30 ms of compute we have after WebSocket/JSON overhead within the 50 ms tick).
 *
 * Measures: (1) raw step (do-nothing), (2) realistic rollout with HeuristicAgent on both
 * sides WITHOUT deepCopy (search internals don't need copies since our policies don't mutate).
 *
 * Run: gradlew :app:runBench
 */
fun main() {
    // 1) Raw step speed (empty actions)
    run {
        val params = GameParamGenerator.randomParams(1L)
        val state = GameStateFactory(params).createGame()
        val fm = ForwardModel(state, params)
        val empty = emptyMap<Player, Action>()
        repeat(50_000) { fm.step(empty) } // warmup/JIT
        val n = 2_000_000
        val ns = measureNanoTime { repeat(n) { fm.step(empty) } }
        println("[BENCH] raw step (do-nothing): %.3f us/step".format(ns.toDouble() / n / 1000.0))
    }

    // 2) Realistic rollout: HeuristicAgent both sides, no deepCopy (agents are read-only)
    run {
        val h1 = HeuristicAgent(); val h2 = HeuristicAgent()
        var totalTicks = 0L; var games = 0
        // warmup
        repeat(3) { g ->
            val params = GameParamGenerator.randomParams(900L + g)
            val st = GameStateFactory(params).createGame()
            val fm = ForwardModel(st, params)
            h1.prepareToPlayAs(Player.Player1, params); h2.prepareToPlayAs(Player.Player2, params)
            while (!fm.isTerminal()) {
                fm.step(mapOf(Player.Player1 to h1.getAction(fm.state), Player.Player2 to h2.getAction(fm.state)))
            }
        }
        val ns = measureNanoTime {
            repeat(30) { g ->
                val params = GameParamGenerator.randomParams(100L + g)
                val st = GameStateFactory(params).createGame()
                val fm = ForwardModel(st, params)
                h1.prepareToPlayAs(Player.Player1, params); h2.prepareToPlayAs(Player.Player2, params)
                while (!fm.isTerminal()) {
                    fm.step(mapOf(Player.Player1 to h1.getAction(fm.state), Player.Player2 to h2.getAction(fm.state)))
                    totalTicks++
                }
                games++
            }
        }
        val msPerGame = ns.toDouble() / 1e6 / games
        val usPerTick = ns.toDouble() / 1000.0 / totalTicks
        println("[BENCH] heuristic-rollout: %.2f ms/game, %.2f us/tick (avg %.0f ticks/game)".format(msPerGame, usPerTick, totalTicks.toDouble() / games))
        val budgetUs = 30_000.0
        println("[BENCH] within 30ms: ~%.0f rollout-ticks, or ~%.1f full 100-tick rollouts, or ~%.2f full games"
            .format(budgetUs / usPerTick, budgetUs / (usPerTick * 100), budgetUs / (usPerTick * (totalTicks.toDouble() / games))))
    }
}
