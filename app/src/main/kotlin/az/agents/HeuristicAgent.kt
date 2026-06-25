package az.agents

import az.core.Heuristics
import games.planetwars.agents.Action
import games.planetwars.agents.PlanetWarsPlayer
import games.planetwars.core.GameState
import games.planetwars.core.Planet
import games.planetwars.core.Player

/**
 * Strong-ish greedy economic agent — the Phase-1 floor and the search rollout policy.
 *
 * Per tick it considers EVERY idle owned planet as a source and EVERY non-owned planet
 * as a target, and launches the single best capture:
 *   - sizes the send to exactly outnumber predicted defenders-at-arrival (+ buffer),
 *     instead of GreedyHeuristic's blind "send half" (which wastes ships / under-commits);
 *   - skips targets it is already capturing with an inbound fleet (no double-commit);
 *   - values economy (growthRate gained), with extra weight for taking from the enemy;
 *   - keeps a defensive reserve equal to the net enemy ships already inbound to the source.
 *
 * Deterministic, sub-millisecond, robust across the parameter ranges.
 */
class HeuristicAgent(
    private val captureBuffer: Double = 2.0,
    private val growthWeight: Double = 80.0,
    private val enemyDenyWeight: Double = 1.5,
    private val distanceWeight: Double = 0.6,
    private val minSourceShips: Double = 2.0,
) : PlanetWarsPlayer() {

    override fun getAgentType(): String = "AZ-Heuristic"

    override fun getAction(gameState: GameState): Action {
        val me = player
        val idle = gameState.planets.filter {
            it.owner == me && it.transporter == null && it.nShips > minSourceShips
        }
        if (idle.isEmpty()) return Action.doNothing()

        var bestSource: Planet? = null
        var bestTarget: Planet? = null
        var bestShips = 0.0
        var bestScore = Double.NEGATIVE_INFINITY

        for (source in idle) {
            // Defend first: reserve the net enemy ships already inbound to this planet.
            val threat = Heuristics.enemyIncoming(gameState, me, source.id)
            val friendlyIn = Heuristics.myIncoming(gameState, me, source.id)
            val reserve = (threat - friendlyIn).coerceAtLeast(0.0)
            val available = source.nShips - reserve
            if (available <= 1.0) continue

            for (target in gameState.planets) {
                if (target.owner == me) continue
                val arr = Heuristics.travelTicks(source, target, params)
                val defenders = Heuristics.defendersAtArrival(target, arr)
                val alreadyComing = Heuristics.myIncoming(gameState, me, target.id)
                val need = defenders + captureBuffer - alreadyComing
                if (need <= 0.0) continue        // already covered by an inbound fleet
                if (need > available) continue   // can't capture from this source now

                val deny = if (target.owner == Player.Neutral) 1.0 else enemyDenyWeight
                val value = growthWeight * target.growthRate * deny
                val score = value - need - distanceWeight * arr
                if (score > bestScore) {
                    bestScore = score
                    bestSource = source
                    bestTarget = target
                    bestShips = need
                }
            }
        }

        val src = bestSource ?: return Action.doNothing()
        val tgt = bestTarget ?: return Action.doNothing()
        val ships = minOf(bestShips, src.nShips)
        if (ships < 1.0) return Action.doNothing()
        return Action(me, src.id, tgt.id, ships)
    }
}
