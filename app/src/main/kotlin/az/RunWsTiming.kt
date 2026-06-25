package az

import az.agents.ChampionAgent
import games.planetwars.agents.RemoteAgent
import games.planetwars.agents.random.CarefulRandomAgent
import games.planetwars.core.GameParamGenerator
import games.planetwars.core.Player
import games.planetwars.runners.GameRunnerCoRoutines
import json_rmi.GameAgentServer
import kotlin.concurrent.thread

/**
 * v2 WebSocket budget / timeout verification (the real competition transport).
 *
 * Unlike v1's RolloutSearchAgent (which finished in ~2.5ms), the MCTS ChampionAgent uses the FULL
 * ~28ms search budget every move, so it sits much closer to the 50ms hard limit — this gates v2.
 *
 * Serves ChampionAgent via GameAgentServer (the exact path Docker `java -jar` would use), connects
 * over ws://.../ws with the framework RemoteAgent, plays several full games vs CarefulRandom under a
 * realistic 50ms timeout, and reports per-game timeout counts + average action times. PASS = 0 timeouts.
 *
 * Run: java -cp client-server.jar az.RunWsTimingKt
 */
fun main() {
    val port = 9097
    val server = GameAgentServer(port = port, agentClass = ChampionAgent::class)
    thread(isDaemon = true) { server.start(wait = true) }
    Thread.sleep(3000) // let Netty bind

    var totalTimeouts = 0
    val seeds = longArrayOf(1L, 7L, 13L, 21L, 42L)
    for (seed in seeds) {
        val params = GameParamGenerator.randomParams(seed)
        val remote = RemoteAgent("ChampionV2", port = port)
        val runner = GameRunnerCoRoutines(remote, CarefulRandomAgent(), params, timeoutMillis = 50)
        val fm = runner.runGame()
        val to = runner.getTimeoutCount()[Player.Player1] ?: 0      // our champion is agent1 = Player1
        val avg = runner.getAverageActionTimes()[Player.Player1]
        totalTimeouts += to
        println("[WS] seed=$seed planets=${params.numPlanets} endTick=${fm.state.gameTick} " +
            "ourAvgMs=$avg ourTimeouts=$to leader=${fm.getLeader()}")
        try { remote.processGameOver(fm.state) } catch (_: Exception) {}
    }
    println("[WS] TOTAL_TIMEOUTS=$totalTimeouts over ${seeds.size} games  => ${if (totalTimeouts == 0) "PASS" else "FAIL"}")
    server.stop()
    println("[WS] DONE")
}
