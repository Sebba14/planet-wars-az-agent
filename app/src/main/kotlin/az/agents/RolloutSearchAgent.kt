package az.agents

import az.core.ActionGen
import az.core.Heuristics
import az.core.TimeBudget
import games.planetwars.agents.Action
import games.planetwars.agents.PlanetWarsPlayer
import games.planetwars.core.ForwardModel
import games.planetwars.core.GameState
import games.planetwars.core.GameStateFactory
import games.planetwars.core.Player

/**
 * Phase-1 strong agent: 1-ply lookahead with deterministic heuristic rollout (SFP).
 *
 * For each candidate first-move (from ActionGen), apply it together with the opponent's
 * modeled move this tick, then roll BOTH sides forward with HeuristicAgent for up to
 * [rolloutTicks] (or to terminal), and score the resulting state with Heuristics.evaluate
 * (which projects compounding economy over the remaining horizon). Pick the best candidate.
 *
 * The forward model + heuristic are deterministic, so one rollout per candidate suffices.
 * Bounded by [budgetMs] via TimeBudget; falls back to the greedy heuristic move if the
 * budget is too tight to evaluate any candidate.
 */
class RolloutSearchAgent(
    // Shipped defaults (used by GameAgentServer.createInstance): tuned so server compute
    // (~10ms with these) + WebSocket/JSON round-trip stays well under the 50ms tick budget.
    private val budgetMs: Long = 28,
    private val rolloutTicks: Int = 150,
    private val maxCandidates: Int = 20,
) : PlanetWarsPlayer() {

    private val fallback = HeuristicAgent()
    private val rollMe = HeuristicAgent()
    private val rollOpp = HeuristicAgent()

    override fun getAgentType(): String = "AZ-SFP-Rollout"

    override fun prepareToPlayAs(player: Player, params: games.planetwars.core.GameParams, opponent: String?): String {
        super.prepareToPlayAs(player, params, opponent)
        fallback.prepareToPlayAs(player, params, opponent)
        rollMe.prepareToPlayAs(player, params, opponent)
        rollOpp.prepareToPlayAs(player.opponent(), params, opponent)
        warmUp()
        return getAgentType()
    }

    /**
     * JIT-warm the search path before the timed game so the first real getAction doesn't
     * blow the 50ms budget (the observed single cold-start timeout). prepareToPlayAs is NOT
     * under the per-move timeout, so this is free. Warms on a representative full-size board.
     */
    private fun warmUp() {
        try {
            repeat(4) {
                val g = GameStateFactory(params).createGame()
                getAction(g)
            }
        } catch (_: Exception) { /* warmup must never fail the game */ }
    }

    override fun getAction(gameState: GameState): Action {
        val budget = TimeBudget(budgetMs)
        val me = player
        val opp = me.opponent()

        val candidates = ActionGen.candidates(gameState, me, params, maxCandidates)
        if (candidates.size <= 1) return candidates.firstOrNull() ?: Action.doNothing()

        // Opponent's modeled action for THIS tick (self-model).
        val oppNow = rollOpp.getAction(gameState)

        var best: Action = fallback.getAction(gameState) // safe default if budget is tiny
        var bestVal = Double.NEGATIVE_INFINITY
        var evaluated = 0

        for (cand in candidates) {
            if (budget.expired(3.0)) break
            val sim = gameState.deepCopy()
            val fm = ForwardModel(sim, params)
            fm.step(mapOf(me to cand, opp to oppNow))
            var t = 0
            while (t < rolloutTicks && !fm.isTerminal()) {
                fm.step(mapOf(me to rollMe.getAction(fm.state), opp to rollOpp.getAction(fm.state)))
                t++
            }
            val v = Heuristics.evaluate(fm.state, me, params)
            if (v > bestVal) { bestVal = v; best = cand }
            evaluated++
        }
        return best
    }
}
