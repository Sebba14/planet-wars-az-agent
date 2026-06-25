package az.agents

import games.planetwars.agents.Action
import games.planetwars.core.GameParams
import games.planetwars.core.GameState
import games.planetwars.core.GameStateFactory
import games.planetwars.core.Player
import kotlinx.serialization.json.Json

/**
 * The shipped v2 agent: rollout-MCTS with the tuning that emerged from the overnight ablation
 * campaign (all validated vs the v1 config and held-out non-self-model opponents):
 *   - FPU (fpuReduction=0.2) replacing the forced "visit every child once" sweep — the dominant
 *     lever; lets the budget deepen the few moves that matter instead of force-sampling all 28.
 *   - eval=vuln: forward-looking vulnerability term (credit/debit planets about to change hands
 *     per in-flight fleets) on top of material + growth*horizon + territory.
 *   - rolloutTicks=80 (shorter rollouts -> more sims), maxCandidates=28 (wider), cPuct=1.4.
 * Measured ~71% vs the v1 config (n=200, Docker) and 99% vs SimpleEvo/Greedy.
 *
 * No-arg constructor so json_rmi.GameAgentServer can reflectively instantiate it (the served agent).
 */
class ChampionAgent : MctsAgent(
    budgetMs = 28,
    cPuct = 1.4,
    maxDepth = 6,
    rolloutTicks = 80,
    maxCandidates = 28,
    fpuReduction = 0.2,
    ensureGreedy = false,
    evalPreset = "vuln",
    firstMoveBudgetMs = 6,  // tiny first-move budget — leave headroom for any residual cold-start cost
) {
    override fun getAgentType(): String = "AZ-MCTS-v2"

    override fun prepareToPlayAs(player: Player, params: GameParams, opponent: String?): String {
        super.prepareToPlayAs(player, params, opponent)  // search-path JIT warmup + move counter reset
        warmSerialization()                              // JIT the JSON-RMI (de)serialization path
        return getAgentType()
    }

    /**
     * The lone WS-test timeout was always move 1 of a cold JVM — the first-ever JSON-RMI
     * (de)serialization of a GameState JIT-compiling. The server runs in THIS JVM, so round-tripping
     * a full-size GameState/Action here (prepareToPlayAs is not under the per-move timeout) compiles
     * those exact serializer code paths in advance, so the first timed move is already warm.
     */
    private fun warmSerialization() {
        try {
            val j = Json { ignoreUnknownKeys = true }
            val s: GameState = GameStateFactory(GameParams(numPlanets = 30)).createGame()
            repeat(15) {
                val str = j.encodeToString(GameState.serializer(), s)
                j.decodeFromString(GameState.serializer(), str)
                j.decodeFromString(Action.serializer(), j.encodeToString(Action.serializer(), Action.doNothing()))
            }
        } catch (_: Exception) { /* warmup must never fail the game */ }
    }
}
