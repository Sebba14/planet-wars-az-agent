package az

import az.agents.HeuristicAgent
import az.agents.MctsAgent
import az.agents.RolloutSearchAgent
import az.agents.SimpleDispatchAgent
import competition_entry.GreedyHeuristicAgent
import games.planetwars.agents.DoNothingAgent
import games.planetwars.agents.PlanetWarsAgent
import games.planetwars.agents.random.BetterRandomAgent
import games.planetwars.agents.random.CarefulRandomAgent
import games.planetwars.agents.random.PureRandomAgent
import games.planetwars.core.GameParamGenerator
import games.planetwars.core.Player
import games.planetwars.runners.GameRunner

/**
 * Local, in-process gauntlet vs the repo sample agents, across the competition parameter
 * distribution (GameParamGenerator), both seats. GameRunner = no 50ms timeout (fast benching).
 *
 * Run: gradlew :app:runLocalLeague -Pargs="<gamesPerPair> <agent>"
 *   <agent> = heuristic (default) | simple
 */
fun main(args: Array<String>) {
    val parts = (args.getOrNull(0) ?: "").trim().split(Regex("\\s+")).filter { it.isNotEmpty() }
    val gamesPerPair = parts.getOrNull(0)?.toIntOrNull() ?: 50
    val which = parts.getOrNull(1) ?: "heuristic"

    val me: () -> PlanetWarsAgent = when (which) {
        "simple" -> { { SimpleDispatchAgent() } }
        "sfp" -> { { RolloutSearchAgent(budgetMs = 30, rolloutTicks = 120, maxCandidates = 16) } }
        "mcts" -> { { MctsAgent(budgetMs = 14, rolloutTicks = 60, maxDepth = 6, maxCandidates = 16) } }
        "netmcts" -> { { MctsAgent(budgetMs = 20, maxDepth = 6, maxCandidates = 16, useNet = true) } }
        else -> { { HeuristicAgent() } }
    }
    val opponents: List<Pair<String, () -> PlanetWarsAgent>> = listOf(
        "PureRandom" to { PureRandomAgent() },
        "BetterRandom" to { BetterRandomAgent() },
        "CarefulRandom" to { CarefulRandomAgent() },
        "GreedyHeuristic" to { GreedyHeuristicAgent() },
        "Heuristic" to { HeuristicAgent() },
        "SimpleDispatch" to { SimpleDispatchAgent() },
        "DoNothing" to { DoNothingAgent() },
    ).filter { it.first.lowercase() != which.lowercase() }

    println("=== AZ gauntlet: ${me().getAgentType()} vs samples, $gamesPerPair games/pair, both seats ===")
    var totalWins = 0; var totalGames = 0
    for ((name, mk) in opponents) {
        var wins = 0; var losses = 0; var draws = 0
        for (g in 0 until gamesPerPair) {
            val params = GameParamGenerator.randomParams(1000L + g)
            val meFirst = g % 2 == 0
            val a1 = if (meFirst) me() else mk()
            val a2 = if (meFirst) mk() else me()
            val finalModel = GameRunner(a1, a2, params).runGame()
            val mySeat = if (meFirst) Player.Player1 else Player.Player2
            when (finalModel.getLeader()) {
                mySeat -> wins++
                Player.Neutral -> draws++
                else -> losses++
            }
        }
        totalWins += wins; totalGames += gamesPerPair
        println("vs %-16s  W=%-4d L=%-4d D=%-4d  winrate=%.1f%%".format(name, wins, losses, draws, 100.0 * wins / gamesPerPair))
    }
    println("OVERALL winrate: %.1f%% (%d/%d)".format(100.0 * totalWins / totalGames, totalWins, totalGames))
}
