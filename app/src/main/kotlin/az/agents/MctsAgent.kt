package az.agents

import az.core.Heuristics
import az.core.TimeBudget
import az.net.OnnxNet
import az.search.Mcts
import games.planetwars.agents.Action
import games.planetwars.agents.PlanetWarsPlayer
import games.planetwars.core.GameParams
import games.planetwars.core.GameState
import games.planetwars.core.GameStateFactory
import games.planetwars.core.Player

/**
 * Agent wrapper around the UCT/PUCT [Mcts]. With [useNet]=true it loads the bundled ONNX net
 * (/az/model.onnx) for net-guided PUCT (priors + value); if the model is absent it transparently
 * falls back to rollout MCTS. Also the AlphaZero teacher (its search yields the policy targets).
 */
open class MctsAgent(
    private val budgetMs: Long = 28,
    private val cPuct: Double = 1.4,
    private val maxDepth: Int = 6,
    private val rolloutTicks: Int = 80,
    private val maxCandidates: Int = 20,
    private val useNet: Boolean = false,
    private val preloadedNet: OnnxNet? = null,  // gating/self-play loop passes a net loaded from a file
    private val evalPreset: String = "base",    // leaf-eval variant (see Heuristics.EVAL_PRESETS)
    private val fpuReduction: Double = Double.NaN, // NaN = forced unvisited-first sweep; set to enable FPU
    private val ensureGreedy: Boolean = false,    // always keep the greedy/rollout move in the candidate set
    // The FIRST move of each game uses this (smaller) budget to absorb the one-time cold-start spike
    // (JIT + JSON-RMI (de)serialization the in-agent warmup can't reach). -1 = no special-casing.
    private val firstMoveBudgetMs: Long = -1,
) : PlanetWarsPlayer() {

    private var mcts: Mcts? = null
    private var net: OnnxNet? = preloadedNet
    private var movesPlayed = 0
    private val evalWeights: Heuristics.EvalWeights = Heuristics.EVAL_PRESETS[evalPreset] ?: Heuristics.EvalWeights.DEFAULT

    override fun getAgentType(): String = if (net != null) "AZ-NetMCTS" else "AZ-MCTS"

    override fun prepareToPlayAs(player: Player, params: GameParams, opponent: String?): String {
        super.prepareToPlayAs(player, params, opponent)
        if (net == null && useNet) net = OnnxNet.fromResource()
        mcts = Mcts(player, params, cPuct, maxDepth, rolloutTicks, maxCandidates, net, evalWeights = evalWeights, fpuReduction = fpuReduction, ensureGreedy = ensureGreedy)
        movesPlayed = 0
        // Heavier warmup (search path JIT) — prepareToPlayAs is not under the per-move timeout.
        try { repeat(8) { mcts!!.search(GameStateFactory(params).createGame(), TimeBudget(budgetMs)) } } catch (_: Exception) {}
        return getAgentType()
    }

    override fun getAction(gameState: GameState): Action {
        val m = mcts ?: Mcts(player, params, cPuct, maxDepth, rolloutTicks, maxCandidates, net, evalWeights = evalWeights, fpuReduction = fpuReduction, ensureGreedy = ensureGreedy).also { mcts = it }
        // First move of the game runs on a tighter budget to leave headroom for the cold-start spike.
        val b = if (movesPlayed == 0 && firstMoveBudgetMs >= 0) firstMoveBudgetMs else budgetMs
        movesPlayed++
        return m.search(gameState, TimeBudget(b)).best
    }
}
