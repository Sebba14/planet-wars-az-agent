package az

import az.agents.RolloutSearchAgent
import json_rmi.GameAgentServer

/**
 * Competition entry point: serve our agent over WebSocket on port 8080 using the
 * framework's JSON-RMI [GameAgentServer] wrapper (the same wrapper the organizers'
 * RemoteAgent client connects to at ws://host:PORT/ws).
 *
 * This is set as the JAR Main-Class in build.gradle.kts so that the Docker
 * `java -jar app.jar` command launches THIS, not the upstream stub MultiRTSServer.
 *
 * The served agent is [RolloutSearchAgent] (SFP: 1-ply + heuristic rollout). The agent
 * class must have a no-arg constructor — RolloutSearchAgent's params all have defaults, so
 * GameAgentServer's reflective createInstance() uses those tuned defaults each session.
 */
fun main() {
    val port = System.getenv("PORT")?.toIntOrNull() ?: 8080
    println("[AZ] Starting GameAgentServer on port $port (agent=RolloutSearchAgent / SFP)")
    val server = GameAgentServer(port = port, agentClass = RolloutSearchAgent::class)
    server.start(wait = true)
}
