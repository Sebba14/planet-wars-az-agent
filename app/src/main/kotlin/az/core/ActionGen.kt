package az.core

import games.planetwars.agents.Action
import games.planetwars.core.GameParams
import games.planetwars.core.GameState
import games.planetwars.core.Player

/**
 * Candidate-action generator for the SFP search (keeps per-tick branching small but covers
 * the strategically important move types):
 *
 *  - EXPANSION/ATTACK: for each idle owned planet, its most attractive non-owned target,
 *    at several SHIP-SIZE variants (bare capture / capture+hold / full commit) — the lever
 *    the pure greedy lacked (it always sent the minimum or a blind half).
 *  - REINFORCEMENT/DEFENSE: when an owned planet is predicted to be lost (net enemy ships
 *    inbound exceed its garrison + our inbound), send the shortfall from a surplus planet.
 *    The forward-model rollout judges whether it arrives in time.
 *  - DO_NOTHING is always included (accumulate / wait).
 */
object ActionGen {
    private data class Scored(val action: Action, val score: Double)

    fun candidates(state: GameState, me: Player, params: GameParams, maxCandidates: Int = 24): List<Action> {
        val idle = state.planets.filter { it.owner == me && it.transporter == null && it.nShips > 2.0 }
        if (idle.isEmpty()) return listOf(Action.DO_NOTHING)

        val scored = ArrayList<Scored>()

        // --- Expansion / attack candidates ---
        for (source in idle) {
            var bestTargetId = -1
            var bestNeed = 0.0
            var bestScore = Double.NEGATIVE_INFINITY
            for (target in state.planets) {
                if (target.owner == me) continue
                val arr = Heuristics.travelTicks(source, target, params)
                val defenders = Heuristics.defendersAtArrival(target, arr)
                val alreadyComing = Heuristics.myIncoming(state, me, target.id)
                val need = defenders + 1.0 - alreadyComing
                if (need <= 0.0 || need > source.nShips) continue
                val deny = if (target.owner == Player.Neutral) 1.0 else 1.5
                val score = 80.0 * target.growthRate * deny - need - 0.6 * arr
                if (score > bestScore) { bestScore = score; bestTargetId = target.id; bestNeed = need }
            }
            if (bestTargetId >= 0) {
                val sizes = doubleArrayOf(
                    bestNeed + 1.0,
                    (bestNeed * 2.0 + 2.0).coerceAtMost(source.nShips),
                    source.nShips,
                )
                for (s in sizes.toSortedSet()) {
                    if (s in 1.0..source.nShips) scored.add(Scored(Action(me, source.id, bestTargetId, s), bestScore))
                }
            }
        }

        // --- Reinforcement / defense candidates (capped so they don't crowd out attacks) ---
        val threatened = state.planets
            .filter { p ->
                p.owner == me &&
                    (Heuristics.enemyIncoming(state, me, p.id) - p.nShips - Heuristics.myIncoming(state, me, p.id)) > 0.0
            }
            .sortedByDescending { Heuristics.enemyIncoming(state, me, it.id) - it.nShips - Heuristics.myIncoming(state, me, it.id) }
            .take(2)
        for (tp in threatened) {
            val shortfall = Heuristics.enemyIncoming(state, me, tp.id) - tp.nShips - Heuristics.myIncoming(state, me, tp.id)
            if (shortfall <= 0.0) continue
            val source = idle.filter { it.id != tp.id }.minByOrNull { it.position.distance(tp.position) } ?: continue
            val send = (shortfall + 2.0).coerceAtMost(source.nShips)
            if (send >= 1.0) {
                // moderate priority: competes with attacks rather than dominating the cap
                scored.add(Scored(Action(me, source.id, tp.id, send), 40.0 + 80.0 * tp.growthRate))
            }
        }

        val top = scored.sortedByDescending { it.score }
            .map { it.action }
            .distinct()
            .take(maxCandidates)
            .toMutableList()
        top.add(Action.DO_NOTHING)
        return top
    }
}
