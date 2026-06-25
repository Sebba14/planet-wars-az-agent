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
     * Tunable weights for [evaluate]. Defaults reproduce the original eval exactly
     * (material + (myGrow-oppGrow)*min(remaining,400) + 5*planetDiff), so the shipped
     * config's behaviour is unchanged unless a non-default preset is selected.
     *
     *  - [material]      weight on the immediate garrison + in-flight ship difference.
     *  - [economyScale]  weight on growth-rate difference projected over the horizon.
     *  - [economyCap]    horizon cap (ticks) for projecting compounding growth.
     *  - [territory]     bonus per net owned planet (count tiebreaker).
     *  - [vulnerability] forward-looking term: credit/debit imminent ownership changes implied
     *                    by in-flight fleets that the truncated rollout hasn't resolved yet
     *                    (0 = off; skipped entirely so the default path keeps zero overhead).
     */
    data class EvalWeights(
        val material: Double = 1.0,
        val economyScale: Double = 1.0,
        val economyCap: Int = 400,
        val territory: Double = 5.0,
        val vulnerability: Double = 0.0,
        // in-flight transporter ships are counted at this multiple (1.0 = current; <1 = "ships in the
        // air are worth less than landed ships", since at maxTicks unlanded fleets may not count).
        val inflightDiscount: Double = 1.0,
        // leaf-value scale: when false use the fixed [fixedScale] (current behaviour); when true derive
        // scale from the contested "pie" (total ships + projected growth) so the squash doesn't saturate
        // on big maps nor stay linear on small ones -> PUCT keeps Q resolution across the param ranges.
        val dynamicScale: Boolean = false,
        val fixedScale: Double = 500.0,
        val scaleFloor: Double = 50.0,
    ) { companion object { val DEFAULT = EvalWeights() } }

    /** Named eval variants for A/B sweeps (selected by [az.agents.MctsAgent] / RunDuel spec). */
    val EVAL_PRESETS: Map<String, EvalWeights> = mapOf(
        "base" to EvalWeights(),
        "shorthorizon" to EvalWeights(economyCap = 200),
        "longhorizon" to EvalWeights(economyCap = 800),
        "fullhorizon" to EvalWeights(economyCap = 2000),
        "econ" to EvalWeights(economyScale = 1.5),
        "material" to EvalWeights(material = 2.0),
        "territory" to EvalWeights(territory = 15.0),
        "vuln" to EvalWeights(vulnerability = 1.0),
        "vulnlong" to EvalWeights(economyCap = 800, vulnerability = 1.0),
        "balanced" to EvalWeights(economyCap = 600, territory = 8.0, vulnerability = 0.5),
        // scale-normalization family (rank-3 idea) — the precondition for other eval terms to register.
        "dyn" to EvalWeights(dynamicScale = true),
        "dynlong" to EvalWeights(dynamicScale = true, economyCap = 800),
        "dynvuln" to EvalWeights(dynamicScale = true, vulnerability = 1.0),
        "dyninflight" to EvalWeights(dynamicScale = true, inflightDiscount = 0.85),
        "inflight" to EvalWeights(inflightDiscount = 0.85),
        // best-guess combination to confirm head-to-head once ablations are in.
        "combo" to EvalWeights(dynamicScale = true, economyCap = 800, vulnerability = 0.75, inflightDiscount = 0.85),
    )

    /**
     * Static evaluation of [state] from [me]'s perspective. Win is decided by end-game
     * garrison totals, but growth compounds over remaining ticks, so we value controlled
     * growth scaled by the remaining horizon (capped). In-flight ships count for their owner.
     * Returns a zero-sum-ish scalar (positive = good for me); not bounded.
     */
    fun evaluate(state: GameState, me: Player, params: GameParams, w: EvalWeights = EvalWeights.DEFAULT): Double {
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
            if (t.owner == me) myShips += t.nShips * w.inflightDiscount
            else if (t.owner == opp) oppShips += t.nShips * w.inflightDiscount
        }
        val remaining = max(0, params.maxTicks - state.gameTick)
        // Each unit of growth yields ~`remaining` future ships; cap horizon so early game
        // doesn't overweight economy to the point of ignoring material safety.
        val horizon = remaining.coerceAtMost(w.economyCap).toDouble()
        val material = w.material * (myShips - oppShips)
        val economy = w.economyScale * (myGrow - oppGrow) * horizon
        val territory = w.territory * (myPlanets - oppPlanets)
        var total = material + economy + territory
        if (w.vulnerability != 0.0) total += w.vulnerability * vulnerabilitySwing(state, me, opp, horizon)
        return total
    }

    /**
     * Net economic swing (growth*horizon) from planets whose ownership is about to flip per the
     * net in-flight balance: a planet I'll lose is debited its future growth, one I'll capture is
     * credited. Computes incoming sums in one pass to stay O(planets). Positive = good for me.
     */
    private fun vulnerabilitySwing(state: GameState, me: Player, opp: Player, horizon: Double): Double {
        val myInc = HashMap<Int, Double>(); val oppInc = HashMap<Int, Double>()
        for (p in state.planets) {
            val t = p.transporter ?: continue
            when (t.owner) {
                me -> myInc[t.destinationIndex] = (myInc[t.destinationIndex] ?: 0.0) + t.nShips
                opp -> oppInc[t.destinationIndex] = (oppInc[t.destinationIndex] ?: 0.0) + t.nShips
                else -> {}
            }
        }
        var swing = 0.0
        for (p in state.planets) {
            val mine = myInc[p.id] ?: 0.0; val theirs = oppInc[p.id] ?: 0.0
            val value = p.growthRate * horizon
            when (p.owner) {
                me -> if (theirs - mine - p.nShips > 0.0) swing -= value          // about to lose it
                opp -> if (mine - theirs - p.nShips > 0.0) swing += value          // about to take it
                else -> { // neutral: whoever's net inbound exceeds its garrison takes the new economy
                    if (mine - p.nShips > 0.0 && mine > theirs) swing += value
                    else if (theirs - p.nShips > 0.0 && theirs > mine) swing -= value
                }
            }
        }
        return swing
    }

    /** Normalized terminal/eval value in [-1, 1] for value targets (tanh-like squash). */
    fun normalizedValue(rawEval: Double, scale: Double = 500.0): Double {
        val x = rawEval / scale
        return x / (1.0 + kotlin.math.abs(x))
    }

    /**
     * Squashed leaf value in [-1,1] for search. Uses [EvalWeights.dynamicScale] to pick the squash
     * scale: a state-derived "contested pie" (total ships + projected owned growth) keeps the value
     * from saturating on big maps / staying linear on small ones, preserving PUCT's Q resolution.
     */
    fun evaluateNormalized(state: GameState, me: Player, params: GameParams, w: EvalWeights = EvalWeights.DEFAULT): Double {
        val raw = evaluate(state, me, params, w)
        val scale = if (!w.dynamicScale) w.fixedScale else {
            var totalShips = 0.0; var totalGrow = 0.0
            for (p in state.planets) {
                totalShips += p.nShips
                p.transporter?.let { totalShips += it.nShips }
                if (p.owner != Player.Neutral) totalGrow += p.growthRate
            }
            val horizon = max(0, params.maxTicks - state.gameTick).coerceAtMost(w.economyCap).toDouble()
            max(w.scaleFloor, totalShips + totalGrow * horizon)
        }
        return normalizedValue(raw, scale)
    }

    /**
     * The single best greedy capture move from [me]'s view — the shared core of [az.agents.HeuristicAgent]
     * (the rollout policy + SFP fallback). Exposed so [ActionGen] can guarantee it is never pruned from
     * the candidate set (otherwise the move driving the rollouts can be unplayable at the root).
     */
    fun greedyAction(
        state: GameState, me: Player, params: GameParams,
        captureBuffer: Double = 2.0, growthWeight: Double = 80.0, enemyDenyWeight: Double = 1.5,
        distanceWeight: Double = 0.6, minSourceShips: Double = 2.0,
    ): games.planetwars.agents.Action {
        val idle = state.planets.filter { it.owner == me && it.transporter == null && it.nShips > minSourceShips }
        if (idle.isEmpty()) return games.planetwars.agents.Action.doNothing()
        var bestSource: Planet? = null; var bestTarget: Planet? = null
        var bestShips = 0.0; var bestScore = Double.NEGATIVE_INFINITY
        for (source in idle) {
            val reserve = (enemyIncoming(state, me, source.id) - myIncoming(state, me, source.id)).coerceAtLeast(0.0)
            val available = source.nShips - reserve
            if (available <= 1.0) continue
            for (target in state.planets) {
                if (target.owner == me) continue
                val arr = travelTicks(source, target, params)
                val defenders = defendersAtArrival(target, arr)
                val need = defenders + captureBuffer - myIncoming(state, me, target.id)
                if (need <= 0.0 || need > available) continue
                val deny = if (target.owner == Player.Neutral) 1.0 else enemyDenyWeight
                val score = growthWeight * target.growthRate * deny - need - distanceWeight * arr
                if (score > bestScore) { bestScore = score; bestSource = source; bestTarget = target; bestShips = need }
            }
        }
        val src = bestSource ?: return games.planetwars.agents.Action.doNothing()
        val tgt = bestTarget ?: return games.planetwars.agents.Action.doNothing()
        val ships = minOf(bestShips, src.nShips)
        if (ships < 1.0) return games.planetwars.agents.Action.doNothing()
        return games.planetwars.agents.Action(me, src.id, tgt.id, ships)
    }
}
