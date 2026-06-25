package az.core

import games.planetwars.core.GameParams
import games.planetwars.core.GameState
import games.planetwars.core.Planet
import games.planetwars.core.Player
import kotlin.math.ceil
import kotlin.math.max

/**
 * Shared game-logic helpers used by the heuristic agent, the SFP search, and the
 * (later) net featurizer. All derived from the verified ForwardModel mechanics:
 *  - fleets travel at constant `transporterSpeed`; arrive when within target radius
 *  - owned planets grow by growthRate each tick; neutral planets do not grow
 *  - a planet holds at most one in-flight transporter
 *  - capture requires arriving ships to strictly outnumber defenders
 */
object Heuristics {

    /** Whole ticks for a fleet from [source] to reach [target] (arrival = within target radius). */
    fun travelTicks(source: Planet, target: Planet, params: GameParams): Int {
        val d = source.position.distance(target.position) - target.radius
        return max(1, ceil(d / params.transporterSpeed).toInt())
    }

    /**
     * Defenders present when our fleet arrives (approximation ignoring third-party fleets):
     * neutral planets don't grow; enemy planets grow by growthRate per tick of travel.
     */
    fun defendersAtArrival(target: Planet, arrivalTicks: Int): Double =
        if (target.owner == Player.Neutral) target.nShips
        else target.nShips + target.growthRate * arrivalTicks

    /** Sum of our in-flight ships already heading to [targetId] (avoid double-committing). */
    fun myIncoming(state: GameState, me: Player, targetId: Int): Double {
        var sum = 0.0
        for (p in state.planets) {
            val t = p.transporter ?: continue
            if (t.owner == me && t.destinationIndex == targetId) sum += t.nShips
        }
        return sum
    }

    /** Enemy in-flight ships heading to [targetId] (threat to our planets). */
    fun enemyIncoming(state: GameState, me: Player, targetId: Int): Double {
        var sum = 0.0
        for (p in state.planets) {
            val t = p.transporter ?: continue
            if (t.owner == me.opponent() && t.destinationIndex == targetId) sum += t.nShips
        }
        return sum
    }

    /**
     * Static evaluation of [state] from [me]'s perspective. Win is decided by end-game
     * garrison totals, but growth compounds over remaining ticks, so we value controlled
     * growth scaled by the remaining horizon (capped). In-flight ships count for their owner.
     * Returns a zero-sum-ish scalar (positive = good for me); not bounded.
     */
    fun evaluate(state: GameState, me: Player, params: GameParams): Double {
        val opp = me.opponent()
        var myShips = 0.0; var oppShips = 0.0
        var myGrow = 0.0; var oppGrow = 0.0
        var myPlanets = 0; var oppPlanets = 0
        for (p in state.planets) {
            when (p.owner) {
                me -> { myShips += p.nShips; myGrow += p.growthRate; myPlanets++ }
                opp -> { oppShips += p.nShips; oppGrow += p.growthRate; oppPlanets++ }
                else -> {}
            }
            val t = p.transporter ?: continue
            if (t.owner == me) myShips += t.nShips else if (t.owner == opp) oppShips += t.nShips
        }
        val remaining = max(0, params.maxTicks - state.gameTick)
        // Each unit of growth yields ~`remaining` future ships; cap horizon so early game
        // doesn't overweight economy to the point of ignoring material safety.
        val horizon = remaining.coerceAtMost(400).toDouble()
        val material = myShips - oppShips
        val economy = (myGrow - oppGrow) * horizon
        val territory = (myPlanets - oppPlanets) * 5.0
        return material + economy + territory
    }

    /** Normalized terminal/eval value in [-1, 1] for value targets (tanh-like squash). */
    fun normalizedValue(rawEval: Double, scale: Double = 500.0): Double {
        val x = rawEval / scale
        return x / (1.0 + kotlin.math.abs(x))
    }
}
