package az.selfplay

import az.core.Featurizer
import az.core.TimeBudget
import az.net.OnnxNet
import az.search.Mcts
import games.planetwars.core.ForwardModel
import games.planetwars.core.GameParamGenerator
import games.planetwars.core.GameStateFactory
import games.planetwars.core.Player

/**
 * AlphaZero self-play game generator: MCTS vs MCTS over randomized competition parameters.
 * Records, for each seat at each real-decision tick, (features, aggregated source/target visit
 * policy) and backfills the game outcome z. Returns the encoded fixed-size records for a shard.
 */
class SelfPlay(
    private val budgetMs: Long = 15,
    private val mctsRolloutTicks: Int = 60,
    private val maxDepth: Int = 6,
    private val maxCandidates: Int = 16,
    private val cPuct: Double = 1.4,
    private val net: OnnxNet? = null,  // net-guided self-play once a net exists (gen > 0)
) {
    private class Pending(val record: FloatArray, val seat: Player)

    fun playGame(seed: Long): List<FloatArray> {
        val params = GameParamGenerator.randomParams(seed)
        val state = GameStateFactory(params).createGame()
        val fm = ForwardModel(state, params)
        val mcts1 = Mcts(Player.Player1, params, cPuct, maxDepth, mctsRolloutTicks, maxCandidates, net)
        val mcts2 = Mcts(Player.Player2, params, cPuct, maxDepth, mctsRolloutTicks, maxCandidates, net)

        val pending = ArrayList<Pending>()
        while (!fm.isTerminal()) {
            val s = fm.state
            val r1 = mcts1.search(s, TimeBudget(budgetMs))
            val r2 = mcts2.search(s, TimeBudget(budgetMs))
            if (r1.policy.size > 1) {
                val rec = ShardFormat.encode(Featurizer.featurize(s, Player.Player1, params), ShardFormat.aggregatePolicy(r1.policy))
                pending.add(Pending(rec, Player.Player1))
            }
            if (r2.policy.size > 1) {
                val rec = ShardFormat.encode(Featurizer.featurize(s, Player.Player2, params), ShardFormat.aggregatePolicy(r2.policy))
                pending.add(Pending(rec, Player.Player2))
            }
            fm.step(mapOf(Player.Player1 to r1.best, Player.Player2 to r2.best))
        }

        val winner = fm.getLeader()
        for (pd in pending) {
            val z = when {
                winner == Player.Neutral -> 0f
                winner == pd.seat -> 1f
                else -> -1f
            }
            ShardFormat.setValue(pd.record, z)
        }
        return pending.map { it.record }
    }
}
