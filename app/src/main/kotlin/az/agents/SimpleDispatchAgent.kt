package az.agents

import games.planetwars.agents.Action
import games.planetwars.agents.PlanetWarsPlayer
import games.planetwars.core.GameState
import games.planetwars.core.Player

/**
 * Phase 0 placeholder agent — minimal, obviously-correct, and non-passive.
 *
 * Strategy: from the owned, idle (no in-flight transporter) planet with the most
 * ships, send half its garrison to the cheapest-to-take non-owned planet
 * (ranked by defenders + a small distance penalty; enemy planets weighted higher).
 *
 * Its ONLY purpose is to de-risk the build / Docker / WebSocket / submission
 * pipeline end-to-end before any real strategy. The strong SFP agent (Phase 1)
 * and the AlphaZero net (Phase 2+) replace this. Do not invest strategy here.
 */
class SimpleDispatchAgent : PlanetWarsPlayer() {
    override fun getAgentType(): String = "AZ-SimpleDispatch-Phase0"

    override fun getAction(gameState: GameState): Action {
        val mine = gameState.planets.filter { it.owner == player && it.transporter == null }
        if (mine.isEmpty()) return Action.doNothing()
        val source = mine.maxByOrNull { it.nShips } ?: return Action.doNothing()
        if (source.nShips < 2.0) return Action.doNothing()

        val targets = gameState.planets.filter { it.owner != player }
        if (targets.isEmpty()) return Action.doNothing()
        val target = targets.minByOrNull { t ->
            val dist = source.position.distance(t.position)
            val strength = if (t.owner == Player.Neutral) t.nShips else t.nShips * 1.5
            strength + dist / 100.0
        } ?: return Action.doNothing()

        return Action(player, source.id, target.id, source.nShips / 2.0)
    }
}
