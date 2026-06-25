package az.agents

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
class MctsAgent(
    private val budgetMs: Long = 28,
    private val cPuct: Double = 1.4,
    private val maxDepth: Int = 6,
    private val rolloutTicks: Int = 80,
    private val maxCandidates: Int = 20,
    private val useNet: Boolean = false,
    private val preloadedNet: OnnxNet? = null,  // gating/self-play loop passes a net loaded from a file
) : PlanetWarsPlayer() {

    private var mcts: Mcts? = null
    private var net: OnnxNet? = preloadedNet

    override fun getAgentType(): String = if (net != null) "AZ-NetMCTS" else "AZ-MCTS"

    override fun prepareToPlayAs(player: Player, params: GameParams, opponent: String?): String {
        super.prepareToPlayAs(player, params, opponent)
        if (net == null && useNet) net = OnnxNet.fromResource()
        mcts = Mcts(player, params, cPuct, maxDepth, rolloutTicks, maxCandidates, net)
        try { repeat(3) { mcts!!.search(GameStateFactory(params).createGame(), TimeBudget(budgetMs)) } } catch (_: Exception) {}
        return getAgentType()
    }

    override fun getAction(gameState: GameState): Action {
        val m = mcts ?: Mcts(player, params, cPuct, maxDepth, rolloutTicks, maxCandidates, net).also { mcts = it }
        return m.search(gameState, TimeBudget(budgetMs)).best
    }
}
