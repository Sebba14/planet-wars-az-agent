package az

import az.agents.ChampionAgent
import json_rmi.GameAgentServer

/**
 * Competition entry point: serve our agent over WebSocket on port 8080 using the
 * framework's JSON-RMI [GameAgentServer] wrapper (the same wrapper the organizers'
 * RemoteAgent client connects to at ws://host:PORT/ws).
 *
 * This is set as the JAR Main-Class in build.gradle.kts so that the Docker
 * `java -jar app.jar` command launches THIS, not the upstream stub MultiRTSServer.
 *
 * The served agent is [ChampionAgent] (v2: rollout-MCTS with FPU + vulnerability eval, ~71% vs the
 * v1 SFP config, 0 WebSocket timeouts verified). It has a no-arg constructor, so GameAgentServer's
 * reflective createInstance() builds it with the tuned champion config each session.
 */
fun main() {
    val port = System.getenv("PORT")?.toIntOrNull() ?: 8080
    println("[AZ] Starting GameAgentServer on port $port (agent=ChampionAgent / MCTS-v2)")
    val server = GameAgentServer(port = port, agentClass = ChampionAgent::class)
    server.start(wait = true)
}
