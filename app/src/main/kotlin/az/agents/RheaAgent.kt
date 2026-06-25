package az.agents

import az.core.Heuristics
import az.core.TimeBudget
import games.planetwars.agents.Action
import games.planetwars.agents.PlanetWarsPlayer
import games.planetwars.core.ForwardModel
import games.planetwars.core.GameParams
import games.planetwars.core.GameState
import games.planetwars.core.GameStateFactory
import games.planetwars.core.Player
import kotlin.random.Random

/**
 * Rolling-Horizon Evolutionary Agent (RHEA) — Lucas's signature SFP method, the class that
 * tends to beat MCTS under tight real-time budgets on this game family because it AMORTISES
 * planning across ticks via a shift buffer (each tick refines the plan carried from the last).
 *
 * This is a strengthened take on the repo's reference [games.planetwars.agents.evo.SimpleEvoAgent],
 * fixing its three weaknesses:
 *   1. SimpleEvo always sends `nShips/2`. Here a gene is DECODED through the same smart sizing the
 *      heuristic/SFP use — "send exactly enough to capture (+buffer)" for attacks, "send the exact
 *      shortfall" for reinforcing a threatened own planet — with a defensive reserve at the source.
 *   2. SimpleEvo scores raw ship-difference. Here fitness is the economy-aware [Heuristics.evaluate]
 *      (material + growth*horizon + territory), evaluated after a PLANNED prefix + a HEURISTIC
 *      ROLLOUT TAIL so a short, precise plan still sees the long-horizon economic payoff.
 *   3. SimpleEvo assumes a do-nothing opponent. Here the opponent is modelled with [HeuristicAgent].
 *
 * Genome: `2*planTicks` floats in [0,1); gene t = (from_t, to_t) selects source/target by index, with
 * a reserved [doNothingProb] slice for "wait/accumulate". shiftBy=2 (one action per tick).
 *
 * Budget-bounded (1+1 EA, accept-ties) via [TimeBudget]; falls back to the greedy heuristic move if
 * the budget is too tight to complete a single evaluation.
 */
class RheaAgent(
    private val budgetMs: Long = 28,
    private val planTicks: Int = 30,
    private val tailTicks: Int = 120,
    private val doNothingProb: Float = 0.1f,
    private val mutProb: Double = 0.18,
    private val flipAtLeastOne: Boolean = true,
    private val useShiftBuffer: Boolean = true,
    seed: Long = 987654321L,
) : PlanetWarsPlayer() {

    private val fallback = HeuristicAgent()
    private val oppModel = HeuristicAgent()   // opponent model during the planned prefix
    private val rollMe = HeuristicAgent()      // heuristic rollout tail (my seat)
    private val rollOpp = HeuristicAgent()     // heuristic rollout tail (opponent seat)
    private val rng = Random(seed)

    private val genomeLen get() = 2 * planTicks
    private class Scored(val score: Double, val genome: FloatArray)
    private var best: Scored? = null

    override fun getAgentType(): String = "AZ-RHEA"

    override fun prepareToPlayAs(player: Player, params: GameParams, opponent: String?): String {
        super.prepareToPlayAs(player, params, opponent)
        fallback.prepareToPlayAs(player, params, opponent)
        oppModel.prepareToPlayAs(player.opponent(), params, opponent)
        rollMe.prepareToPlayAs(player, params, opponent)
        rollOpp.prepareToPlayAs(player.opponent(), params, opponent)
        warmUp()
        best = null   // discard the shift buffer built during warmup (different boards)
        return getAgentType()
    }

    /** JIT-warm the evolve path on full-size boards so the first timed getAction stays in budget. */
    private fun warmUp() {
        try {
            repeat(3) {
                best = null
                getAction(GameStateFactory(params).createGame())
            }
        } catch (_: Exception) { /* warmup must never fail the game */ }
    }

    override fun getAction(gameState: GameState): Action {
        val budget = TimeBudget(budgetMs)
        // If we have no playable move at all this tick, don't waste budget evolving.
        if (gameState.planets.none { it.owner == player && it.transporter == null && it.nShips > 2.0 }) {
            best = null
            return Action.doNothing()
        }

        // Seed this tick's search: shift the carried plan forward one tick, else start random.
        val seed = best
        var current: Scored = if (seed == null || !useShiftBuffer) {
            val g = randomGenome()
            Scored(evalGenome(gameState, g), g)
        } else {
            val g = shiftLeftAppendRandom(seed.genome)
            Scored(evalGenome(gameState, g), g)
        }

        var evals = 1
        // 1+1 EA with accept-ties (allows neutral drift across the fitness plateau).
        while (!budget.expired(3.0)) {
            val mut = mutate(current.genome)
            val score = evalGenome(gameState, mut)
            evals++
            if (score >= current.score) current = Scored(score, mut)
        }

        best = current
        // Decode the first planned gene against the REAL state -> this tick's action.
        val action = decode(gameState, current.genome[0], current.genome[1])
        return if (evals <= 1 && action == Action.DO_NOTHING) fallback.getAction(gameState) else action
    }

    /** Roll the planned prefix (my genes vs the opponent model), then a heuristic tail, then score. */
    private fun evalGenome(root: GameState, genome: FloatArray): Double {
        val sim = root.deepCopy()
        val fm = ForwardModel(sim, params)
        val me = player; val opp = me.opponent()
        var i = 0
        while (i < planTicks && !fm.isTerminal()) {
            val myAction = decode(fm.state, genome[2 * i], genome[2 * i + 1])
            val oppAction = oppModel.getAction(fm.state)
            fm.step(mapOf(me to myAction, opp to oppAction))
            i++
        }
        var t = 0
        while (t < tailTicks && !fm.isTerminal()) {
            fm.step(mapOf(me to rollMe.getAction(fm.state), opp to rollOpp.getAction(fm.state)))
            t++
        }
        return Heuristics.evaluate(fm.state, me, params)
    }

    /**
     * Decode one gene into a legal action against [state]. Reserves a defensive garrison at the
     * source, then either reinforces a threatened own planet by its exact shortfall or captures a
     * non-owned target with exactly enough ships (+buffer); a [doNothingProb] slice means "wait".
     */
    private fun decode(state: GameState, gFrom: Float, gTo: Float): Action {
        val me = player
        val mine = state.planets.filter { it.owner == me && it.transporter == null && it.nShips > 2.0 }
        if (mine.isEmpty()) return Action.DO_NOTHING
        if (gFrom < doNothingProb) return Action.DO_NOTHING

        val srcIdx = (((gFrom - doNothingProb) / (1f - doNothingProb)) * mine.size).toInt().coerceIn(0, mine.size - 1)
        val src = mine[srcIdx]

        // Defensive reserve: keep the net enemy ships already inbound to the source.
        val srcThreat = (Heuristics.enemyIncoming(state, me, src.id) - Heuristics.myIncoming(state, me, src.id)).coerceAtLeast(0.0)
        val available = src.nShips - srcThreat
        if (available < 1.0) return Action.DO_NOTHING

        val targets = state.planets.filter { it.id != src.id }
        if (targets.isEmpty()) return Action.DO_NOTHING
        val tgt = targets[(gTo * targets.size).toInt().coerceIn(0, targets.size - 1)]

        val send: Double = if (tgt.owner == me) {
            // Reinforce only if the target is actually under threat; send its exact shortfall.
            val shortfall = Heuristics.enemyIncoming(state, me, tgt.id) - tgt.nShips - Heuristics.myIncoming(state, me, tgt.id)
            if (shortfall <= 0.0) return Action.DO_NOTHING
            (shortfall + 2.0).coerceAtMost(available)
        } else {
            val arr = Heuristics.travelTicks(src, tgt, params)
            val defenders = Heuristics.defendersAtArrival(tgt, arr)
            val need = defenders + 1.0 - Heuristics.myIncoming(state, me, tgt.id)
            when {
                need <= 0.0 -> return Action.DO_NOTHING          // already being captured
                need > available -> available                     // can't take alone -> commit, gang up
                else -> (need + 1.0).coerceAtMost(available)
            }
        }
        if (send < 1.0) return Action.DO_NOTHING
        return Action(me, src.id, tgt.id, send)
    }

    private fun randomGenome(): FloatArray = FloatArray(genomeLen) { rng.nextFloat() }

    private fun shiftLeftAppendRandom(v: FloatArray): FloatArray {
        val p = FloatArray(v.size)
        for (i in 0 until v.size - 2) p[i] = v[i + 2]   // shiftBy = 2 (one action/tick)
        p[v.size - 2] = rng.nextFloat()
        p[v.size - 1] = rng.nextFloat()
        return p
    }

    private fun mutate(v: FloatArray): FloatArray {
        val x = FloatArray(v.size)
        val forced = if (flipAtLeastOne) rng.nextInt(v.size) else -1
        for (i in v.indices) {
            x[i] = if (i == forced || rng.nextDouble() < mutProb) rng.nextFloat() else v[i]
        }
        return x
    }
}
