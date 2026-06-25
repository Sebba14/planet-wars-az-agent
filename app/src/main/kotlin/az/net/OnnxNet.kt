package az.net

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import az.core.FeatureSpec
import az.core.Featurizer
import games.planetwars.core.GameParams
import games.planetwars.core.GameState
import games.planetwars.core.Player
import java.nio.FloatBuffer

/**
 * JVM inference wrapper around the exported AlphaZero ONNX net (onnxruntime-java).
 *
 * Featurizes a GameState with the SAME Kotlin [Featurizer] used during self-play data generation
 * (so there is no train/infer skew), runs the net, and returns source/target prior logits + value.
 * Single-threaded ORT (latency-bound). Bundled native libs make it work on win/linux x64.
 */
class OnnxNet(modelBytes: ByteArray) : AutoCloseable {
    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession

    init {
        val opts = OrtSession.SessionOptions()
        opts.setIntraOpNumThreads(1)
        opts.setInterOpNumThreads(1)
        session = env.createSession(modelBytes, opts)
    }

    class Output(val srcLogits: FloatArray, val tgtLogits: FloatArray, val value: Float)

    fun infer(state: GameState, me: Player, params: GameParams): Output {
        val f = Featurizer.featurize(state, me, params)
        val P = FeatureSpec.MAX_PLANETS
        val T = FeatureSpec.MAX_TRANSPORTERS

        val pf = FloatArray(P * FeatureSpec.PLANET_F)
        var i = 0
        for (p in 0 until P) for (k in 0 until FeatureSpec.PLANET_F) pf[i++] = f.planetFeats[p][k]
        val ff = FloatArray(T * FeatureSpec.TRANSPORTER_F)
        i = 0
        for (t in 0 until T) for (k in 0 until FeatureSpec.TRANSPORTER_F) ff[i++] = f.transporterFeats[t][k]

        val tPf = OnnxTensor.createTensor(env, FloatBuffer.wrap(pf), longArrayOf(1, P.toLong(), FeatureSpec.PLANET_F.toLong()))
        val tPm = OnnxTensor.createTensor(env, FloatBuffer.wrap(f.planetMask), longArrayOf(1, P.toLong()))
        val tFf = OnnxTensor.createTensor(env, FloatBuffer.wrap(ff), longArrayOf(1, T.toLong(), FeatureSpec.TRANSPORTER_F.toLong()))
        val tFm = OnnxTensor.createTensor(env, FloatBuffer.wrap(f.transporterMask), longArrayOf(1, T.toLong()))
        val tG = OnnxTensor.createTensor(env, FloatBuffer.wrap(f.globals), longArrayOf(1, FeatureSpec.GLOBAL_F.toLong()))

        val inputs = mapOf(
            "planet_feats" to tPf, "planet_mask" to tPm,
            "fleet_feats" to tFf, "fleet_mask" to tFm, "globals" to tG,
        )
        try {
            session.run(inputs).use { res ->
                @Suppress("UNCHECKED_CAST")
                val src = (res.get("src_logits").get().value as Array<FloatArray>)[0]
                @Suppress("UNCHECKED_CAST")
                val tgt = (res.get("tgt_logits").get().value as Array<FloatArray>)[0]
                val value = (res.get("value").get().value as FloatArray)[0]
                return Output(src, tgt, value)
            }
        } finally {
            tPf.close(); tPm.close(); tFf.close(); tFm.close(); tG.close()
        }
    }

    override fun close() {
        session.close()
    }

    companion object {
        /** Load the bundled model from the JAR resources, or null if absent (agent falls back to SFP). */
        fun fromResource(path: String = "/az/model.onnx"): OnnxNet? {
            val bytes = OnnxNet::class.java.getResourceAsStream(path)?.readBytes() ?: return null
            return OnnxNet(bytes)
        }

        /** Load a model from a filesystem path (used by the self-play / gating loop), or null if absent. */
        fun fromFile(path: String): OnnxNet? {
            val f = java.io.File(path)
            if (!f.isFile) return null
            return OnnxNet(f.readBytes())
        }
    }
}
