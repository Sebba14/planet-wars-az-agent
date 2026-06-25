package az.core

import games.planetwars.core.GameParams
import games.planetwars.core.GameState
import games.planetwars.core.Player
import kotlin.math.ln
import kotlin.math.tanh

/**
 * Net input representation — the SINGLE SOURCE OF TRUTH for state featurization.
 * The Python trainer must mirror this layout BYTE-FOR-BYTE (a golden-vector test guards it);
 * train/infer skew was a recurring, painful bug class in the prior project.
 *
 * Seat-canonical: features are produced from `me`'s perspective, and because maps are
 * point-symmetric (Player2 = reflection of Player1 through the board centre), Player2's
 * positions are mirrored so every seat sees a canonical "home" frame. One shared net serves
 * both seats; swap `me` to get the opponent's view.
 *
 * Fixed, padded shapes for easy .npz serialization and batched ONNX inference.
 */
object FeatureSpec {
    const val MAX_PLANETS = 30
    const val MAX_TRANSPORTERS = 30   // at most one transporter per planet
    const val PLANET_F = 12
    const val TRANSPORTER_F = 8
    const val GLOBAL_F = 8

    val SHIP_LOG_NORM = ln(2001.0)        // per-planet / per-fleet ship counts
    val GLOBAL_LOG_NORM = ln(20001.0)     // aggregate ship totals
    const val GROWTH_MAX = 0.2            // max sampled growth rate
}

class Features(
    val planetFeats: Array<FloatArray>,      // [MAX_PLANETS][PLANET_F]
    val planetMask: FloatArray,              // [MAX_PLANETS] 1=present
    val transporterFeats: Array<FloatArray>, // [MAX_TRANSPORTERS][TRANSPORTER_F]
    val transporterMask: FloatArray,         // [MAX_TRANSPORTERS]
    val globals: FloatArray,                 // [GLOBAL_F]
)

object Featurizer {
    private fun shipLog(x: Double) = (ln(1.0 + x.coerceAtLeast(0.0)) / FeatureSpec.SHIP_LOG_NORM).toFloat()
    private fun globalLog(x: Double) = (ln(1.0 + x.coerceAtLeast(0.0)) / FeatureSpec.GLOBAL_LOG_NORM).toFloat()

    fun featurize(state: GameState, me: Player, params: GameParams): Features {
        val opp = me.opponent()
        val mirror = me == Player.Player2
        val w = params.width.toDouble()
        val h = params.height.toDouble()
        val radiusNorm = (params.maxGrowthRate * params.growthToRadiusFactor).coerceAtLeast(1e-6)

        fun cx(x: Double) = (if (mirror) (w - x) else x) / w
        fun cy(y: Double) = (if (mirror) (h - y) else y) / h

        val pf = Array(FeatureSpec.MAX_PLANETS) { FloatArray(FeatureSpec.PLANET_F) }
        val pmask = FloatArray(FeatureSpec.MAX_PLANETS)
        val tf = Array(FeatureSpec.MAX_TRANSPORTERS) { FloatArray(FeatureSpec.TRANSPORTER_F) }
        val tmask = FloatArray(FeatureSpec.MAX_TRANSPORTERS)

        var myShips = 0.0; var oppShips = 0.0; var neutralShips = 0.0
        var myGrow = 0.0; var oppGrow = 0.0; var nPlanets = 0
        var tIdx = 0

        for (p in state.planets) {
            val i = p.id
            if (i < 0 || i >= FeatureSpec.MAX_PLANETS) continue
            nPlanets++
            pmask[i] = 1f
            val myIn = Heuristics.myIncoming(state, me, p.id)
            val enIn = Heuristics.enemyIncoming(state, me, p.id)
            val f = pf[i]
            f[0] = if (p.owner == me) 1f else 0f
            f[1] = if (p.owner == opp) 1f else 0f
            f[2] = if (p.owner == Player.Neutral) 1f else 0f
            f[3] = shipLog(p.nShips)
            f[4] = (p.growthRate / FeatureSpec.GROWTH_MAX).toFloat()
            f[5] = cx(p.position.x).toFloat()
            f[6] = cy(p.position.y).toFloat()
            f[7] = (p.radius / radiusNorm).toFloat()
            f[8] = if (p.transporter != null) 1f else 0f
            f[9] = shipLog(myIn)
            f[10] = shipLog(enIn)
            f[11] = tanh((p.nShips + myIn - enIn) / 50.0).toFloat()

            when (p.owner) {
                me -> { myShips += p.nShips; myGrow += p.growthRate }
                opp -> { oppShips += p.nShips; oppGrow += p.growthRate }
                else -> { neutralShips += p.nShips }
            }

            val tr = p.transporter
            if (tr != null && tIdx < FeatureSpec.MAX_TRANSPORTERS) {
                val dest = state.planets.getOrNull(tr.destinationIndex)
                val tta = if (dest != null)
                    ((tr.s.distance(dest.position) - dest.radius).coerceAtLeast(0.0) / params.transporterSpeed) else 0.0
                val g = tf[tIdx]
                g[0] = if (tr.owner == me) 1f else 0f
                g[1] = if (tr.owner == opp) 1f else 0f
                g[2] = shipLog(tr.nShips)
                g[3] = cx(tr.s.x).toFloat()
                g[4] = cy(tr.s.y).toFloat()
                g[5] = if (dest != null) cx(dest.position.x).toFloat() else 0f
                g[6] = if (dest != null) cy(dest.position.y).toFloat() else 0f
                g[7] = (tta / 200.0).coerceAtMost(1.0).toFloat()
                if (tr.owner == me) myShips += tr.nShips else if (tr.owner == opp) oppShips += tr.nShips
                tmask[tIdx] = 1f
                tIdx++
            }
        }

        val gMaxGrow = FeatureSpec.MAX_PLANETS * FeatureSpec.GROWTH_MAX
        val globals = FloatArray(FeatureSpec.GLOBAL_F)
        globals[0] = globalLog(myShips)
        globals[1] = globalLog(oppShips)
        globals[2] = globalLog(neutralShips)
        globals[3] = (state.gameTick.toDouble() / params.maxTicks).coerceIn(0.0, 1.0).toFloat()
        globals[4] = (nPlanets.toDouble() / FeatureSpec.MAX_PLANETS).toFloat()
        globals[5] = (myGrow / gMaxGrow).toFloat()
        globals[6] = (oppGrow / gMaxGrow).toFloat()
        globals[7] = ((params.transporterSpeed - 2.0) / 3.0).coerceIn(0.0, 1.0).toFloat()  // sampled 2.0..5.0
        return Features(pf, pmask, tf, tmask, globals)
    }
}
