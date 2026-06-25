package az

import az.agents.SimpleDispatchAgent
import games.planetwars.agents.RemoteAgent
import games.planetwars.agents.random.CarefulRandomAgent
import games.planetwars.core.GameParamGenerator
import games.planetwars.runners.GameRunnerCoRoutines
import json_rmi.GameAgentServer
import kotlin.concurrent.thread

/**
 * End-to-end WebSocket smoke test (the real competition transport):
 *  - starts our GameAgentServer (SimpleDispatchAgent) on a local port,
 *  - connects via the framework's RemoteAgent (JSON-RMI over ws://.../ws),
 *  - plays a full game vs a local CarefulRandomAgent under the realistic 50 ms timeout,
 *  - reports the result, average action times, and timeout count.
 *
 * Proves: server reflection/RMI, GameState/Action JSON (de)serialization, and that our
 * agent answers within budget over the wire. Run: gradlew :app:runRemoteSmoke
 */
fun main() {
    val port = 9099
    val server = GameAgentServer(port = port, agentClass = SimpleDispatchAgent::class)
    thread(isDaemon = true) { server.start(wait = true) }
    Thread.sleep(2500) // let Netty bind the port

    val params = GameParamGenerator.randomParams(7L)
    val remote = RemoteAgent("SimpleDispatch", port = port)
    val opponent = CarefulRandomAgent()
    val runner = GameRunnerCoRoutines(remote, opponent, params, timeoutMillis = 50)

    val finalModel = runner.runGame()
    println("[REMOTE-SMOKE] params=$params")
    println("[REMOTE-SMOKE] ${finalModel.statusString()}")
    println("[REMOTE-SMOKE] leader=${finalModel.getLeader()}")
    println("[REMOTE-SMOKE] avgActionTimesMs=${runner.getAverageActionTimes()}")
    println("[REMOTE-SMOKE] timeouts=${runner.getTimeoutCount()}")

    try { remote.processGameOver(finalModel.state) } catch (_: Exception) {}
    server.stop()
    println("[REMOTE-SMOKE] DONE")
}
