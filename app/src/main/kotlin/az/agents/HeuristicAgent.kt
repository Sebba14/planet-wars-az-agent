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

    override fun getAction(gameState: GameState): Action =
        Heuristics.greedyAction(
            gameState, player, params,
            captureBuffer = captureBuffer, growthWeight = growthWeight,
            enemyDenyWeight = enemyDenyWeight, distanceWeight = distanceWeight,
            minSourceShips = minSourceShips,
        )
}
