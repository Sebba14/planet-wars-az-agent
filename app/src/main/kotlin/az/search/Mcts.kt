package az.search

import az.agents.HeuristicAgent
import az.core.ActionGen
import az.core.FeatureSpec
import az.core.Heuristics
import az.core.TimeBudget
import az.net.OnnxNet
import games.planetwars.agents.Action
import games.planetwars.core.ForwardModel
import games.planetwars.core.GameParams
import games.planetwars.core.GameState
import games.planetwars.core.Player
import kotlin.math.exp
import kotlin.math.sqrt

/**
 * UCT/PUCT Monte-Carlo Tree Search over the native forward model, from [me]'s perspective.
 *
 * Simultaneous moves -> deterministic SELF-MODEL opponent (HeuristicAgent), so the tree is a
 * single-agent search. Two leaf-evaluation / prior modes:
 *   - [net] == null  : uniform priors + short heuristic rollout value (pre-AlphaZero baseline).
 *   - [net] != null  : AlphaZero mode — net source/target logits give candidate PRIORS for PUCT,
 *                      and the net value head replaces the rollout at leaves.
 *
 * Returns the root visit distribution (the AlphaZero POLICY TARGET), the most-visited action, value.
 */
class Mcts(
    private val me: Player,
    private val params: GameParams,
    private val cPuct: Double = 1.4,
    private val maxDepth: Int = 6,
    private val rolloutTicks: Int = 80,
    private val maxCandidates: Int = 20,
    private val net: OnnxNet? = null,
) {
    private val opp = me.opponent()
    private val rollMe = HeuristicAgent().also { it.prepareToPlayAs(me, params) }
    private val rollOpp = HeuristicAgent().also { it.prepareToPlayAs(opp, params) }
    private val useNet = net != null

    class Result(val best: Action, val policy: List<Pair<Action, Double>>, val rootValue: Double, val sims: Int)

    private fun isTerminal(s: GameState): Boolean =
        s.gameTick > params.maxTicks ||
            s.planets.none { it.owner == Player.Player1 } ||
            s.planets.none { it.owner == Player.Player2 }

    private inner class Node(val state: GameState) {
        val actions: List<Action> =
            if (isTerminal(state)) emptyList() else ActionGen.candidates(state, me, params, maxCandidates)
        val n = IntArray(actions.size)
        val w = DoubleArray(actions.size)
        val children = arrayOfNulls<Node>(actions.size)
        var totalN = 0
        val priors: DoubleArray
        val netValue: Double   // valid only when useNet

        init {
            if (actions.isEmpty()) {
                priors = DoubleArray(0)
                netValue = if (useNet) Heuristics.normalizedValue(Heuristics.evaluate(state, me, params)) else 0.0
            } else if (useNet) {
                val out = net!!.infer(state, me, params)
                netValue = out.value.toDouble()
                priors = priorsFromNet(out, actions)
            } else {
                priors = DoubleArray(actions.size) { 1.0 / actions.size }
                netValue = 0.0
            }
        }
    }

    private fun priorsFromNet(out: OnnxNet.Output, actions: List<Action>): DoubleArray {
        val raw = DoubleArray(actions.size)
        for (i in actions.indices) {
            val a = actions[i]
            val sIdx = if (a == Action.DO_NOTHING) FeatureSpec.MAX_PLANETS else a.sourcePlanetId
            val sLog = out.srcLogits.getOrElse(sIdx) { -1e9f }.toDouble()
            val tLog = if (a == Action.DO_NOTHING || a.destinationPlanetId < 0) 0.0
                       else out.tgtLogits.getOrElse(a.destinationPlanetId) { -1e9f }.toDouble()
            raw[i] = sLog + tLog
        }
        val mx = raw.max()
        var sum = 0.0
        for (i in raw.indices) { raw[i] = exp(raw[i] - mx); sum += raw[i] }
        for (i in raw.indices) raw[i] /= sum
        return raw
    }

    private fun rolloutValue(state: GameState): Double {
        if (isTerminal(state)) return Heuristics.normalizedValue(Heuristics.evaluate(state, me, params))
        val sim = state.deepCopy()
        val fm = ForwardModel(sim, params)
        var t = 0
        while (t < rolloutTicks && !fm.isTerminal()) {
            fm.step(mapOf(me to rollMe.getAction(fm.state), opp to rollOpp.getAction(fm.state)))
            t++
        }
        return Heuristics.normalizedValue(Heuristics.evaluate(fm.state, me, params))
    }

    private fun leafValue(node: Node): Double = if (useNet) node.netValue else rolloutValue(node.state)

    private fun simulate(node: Node, depth: Int): Double {
        if (node.actions.isEmpty()) return leafValue(node)
        if (depth >= maxDepth) return leafValue(node)

        var idx = node.actions.indices.firstOrNull { node.n[it] == 0 } ?: -1
        if (idx < 0) {
            var bestU = Double.NEGATIVE_INFINITY
            val sqrtN = sqrt(node.totalN.toDouble() + 1.0)
            for (i in node.actions.indices) {
                val q = node.w[i] / node.n[i]
                val u = q + cPuct * node.priors[i] * sqrtN / (1.0 + node.n[i])
                if (u > bestU) { bestU = u; idx = i }
            }
        }

        val value: Double
        if (node.children[idx] == null) {
            val sim = node.state.deepCopy()
            val fm = ForwardModel(sim, params)
            val oppA = rollOpp.getAction(node.state)
            fm.step(mapOf(me to node.actions[idx], opp to oppA))
            val child = Node(sim)
            node.children[idx] = child
            value = leafValue(child)
        } else {
            value = simulate(node.children[idx]!!, depth + 1)
        }
        node.n[idx]++
        node.w[idx] += value
        node.totalN++
        return value
    }

    fun search(root: GameState, budget: TimeBudget): Result {
        val rootNode = Node(root)
        if (rootNode.actions.isEmpty()) return Result(Action.DO_NOTHING, emptyList(), 0.0, 0)
        if (rootNode.actions.size == 1) return Result(rootNode.actions[0], listOf(rootNode.actions[0] to 1.0), 0.0, 0)

        var sims = 0
        while (!budget.expired(3.0)) { simulate(rootNode, 0); sims++ }

        val bestIdx = rootNode.n.indices.maxByOrNull { rootNode.n[it] } ?: 0
        val denom = rootNode.totalN.coerceAtLeast(1).toDouble()
        val policy = rootNode.actions.indices.map { rootNode.actions[it] to (rootNode.n[it] / denom) }
        val rootValue = if (rootNode.totalN > 0) rootNode.w.sum() / rootNode.totalN else 0.0
        return Result(rootNode.actions[bestIdx], policy, rootValue, sims)
    }
}
