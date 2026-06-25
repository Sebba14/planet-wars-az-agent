package az

import competition_entry.GreedyHeuristicAgent
import games.planetwars.agents.PlanetWarsAgent
import games.planetwars.agents.RemoteAgent
import games.planetwars.agents.random.CarefulRandomAgent
import games.planetwars.core.GameParamGenerator
import games.planetwars.runners.GameRunnerCoRoutines

/**
 * Connects to an ALREADY-RUNNING agent server (e.g. our Docker container or a bare
 * `java -jar` process) at ws://localhost:PORT/ws and plays one game under the 50 ms
 * timeout. Does NOT start a server.
 *
 * Run: gradlew :app:runRemoteClient -Pargs="8080 greedy"
 *   arg0 = port (default 8080); arg1 = opponent: careful (default) | greedy
 */
fun main(args: Array<String>) {
    val parts = (args.getOrNull(0) ?: "").trim().split(Regex("\\s+")).filter { it.isNotEmpty() }
    val port = parts.getOrNull(0)?.toIntOrNull() ?: 8080
    val oppName = parts.getOrNull(1) ?: "careful"
    val opponent: PlanetWarsAgent = if (oppName == "greedy") GreedyHeuristicAgent() else CarefulRandomAgent()
    val params = GameParamGenerator.randomParams(11L)
    val remote = RemoteAgent("AZ-Server", port = port)
    val runner = GameRunnerCoRoutines(remote, opponent, params, timeoutMillis = 50)
    val fm = runner.runGame()
    println("[REMOTE-CLIENT] port=$port ${fm.statusString()} leader=${fm.getLeader()}")
    println("[REMOTE-CLIENT] avgActionTimesMs=${runner.getAverageActionTimes()} timeouts=${runner.getTimeoutCount()}")
    try { remote.processGameOver(fm.state) } catch (_: Exception) {}
    println("[REMOTE-CLIENT] DONE")
}
