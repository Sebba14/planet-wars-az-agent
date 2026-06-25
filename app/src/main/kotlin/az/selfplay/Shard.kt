package az.selfplay

import az.core.FeatureSpec
import az.core.Features
import games.planetwars.agents.Action
import java.io.BufferedOutputStream
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Compact little-endian binary shard format for AlphaZero self-play samples — the Kotlin↔Python
 * data contract (Python reader: pwaz/data/shard_io.py must match these offsets exactly).
 *
 * Header (4 × int32 LE): MAGIC, VERSION, RECORD_FLOATS, nSamples.
 * Then nSamples fixed-size records of RECORD_FLOATS float32 (LE), laid out as:
 *   planet_feats   [MAX_PLANETS * PLANET_F]
 *   planet_mask    [MAX_PLANETS]
 *   fleet_feats    [MAX_TRANSPORTERS * TRANSPORTER_F]
 *   fleet_mask     [MAX_TRANSPORTERS]
 *   globals        [GLOBAL_F]
 *   cand_src       [MAX_CAND]   (source planet id; STOP/DO_NOTHING = MAX_PLANETS; pad = -2)
 *   cand_tgt       [MAX_CAND]   (target planet id; STOP = -1; pad = -2)
 *   cand_prob      [MAX_CAND]   (MCTS visit probability for that (src,tgt); pad = 0)
 *   value          [1]          (outcome z from this sample's seat: +1 win / -1 loss / 0 draw)
 */
object ShardFormat {
    const val MAGIC = 0x415A5350 // 'AZSP'
    const val VERSION = 1
    const val MAX_CAND = 24
    const val STOP_SRC = FeatureSpec.MAX_PLANETS  // source index meaning DO_NOTHING

    val RECORD_FLOATS =
        FeatureSpec.MAX_PLANETS * FeatureSpec.PLANET_F +
            FeatureSpec.MAX_PLANETS +
            FeatureSpec.MAX_TRANSPORTERS * FeatureSpec.TRANSPORTER_F +
            FeatureSpec.MAX_TRANSPORTERS +
            FeatureSpec.GLOBAL_F +
            3 * MAX_CAND + 1

    /** Build one fixed-size record. [cands] is the aggregated (src,tgt,prob) policy; value set later. */
    fun encode(features: Features, cands: List<Triple<Int, Int, Double>>): FloatArray {
        val rec = FloatArray(RECORD_FLOATS)
        var o = 0
        for (p in 0 until FeatureSpec.MAX_PLANETS) { for (k in 0 until FeatureSpec.PLANET_F) rec[o++] = features.planetFeats[p][k] }
        for (p in 0 until FeatureSpec.MAX_PLANETS) rec[o++] = features.planetMask[p]
        for (t in 0 until FeatureSpec.MAX_TRANSPORTERS) { for (k in 0 until FeatureSpec.TRANSPORTER_F) rec[o++] = features.transporterFeats[t][k] }
        for (t in 0 until FeatureSpec.MAX_TRANSPORTERS) rec[o++] = features.transporterMask[t]
        for (k in 0 until FeatureSpec.GLOBAL_F) rec[o++] = features.globals[k]
        val srcBase = o; val tgtBase = o + MAX_CAND; val probBase = o + 2 * MAX_CAND
        for (i in 0 until MAX_CAND) { rec[srcBase + i] = -2f; rec[tgtBase + i] = -2f; rec[probBase + i] = 0f }
        for (i in cands.indices.take(MAX_CAND)) {
            rec[srcBase + i] = cands[i].first.toFloat()
            rec[tgtBase + i] = cands[i].second.toFloat()
            rec[probBase + i] = cands[i].third.toFloat()
        }
        // value at the very end (index RECORD_FLOATS-1) left 0; set via setValue()
        return rec
    }

    fun setValue(rec: FloatArray, z: Float) { rec[RECORD_FLOATS - 1] = z }

    /** Aggregate an MCTS policy (per concrete Action) into (source,target) visit mass. */
    fun aggregatePolicy(policy: List<Pair<Action, Double>>): List<Triple<Int, Int, Double>> {
        val map = LinkedHashMap<Pair<Int, Int>, Double>()
        for ((a, p) in policy) {
            val key = if (a == Action.DO_NOTHING) Pair(STOP_SRC, -1) else Pair(a.sourcePlanetId, a.destinationPlanetId)
            map[key] = (map[key] ?: 0.0) + p
        }
        return map.entries.sortedByDescending { it.value }.map { Triple(it.key.first, it.key.second, it.value) }
    }
}

/** Writes records to a shard file (header + raw little-endian float32). */
object ShardWriter {
    fun write(path: File, records: List<FloatArray>) {
        path.parentFile?.mkdirs()
        val tmp = File(path.parentFile, path.name + ".tmp")  // atomic: write tmp then rename
        BufferedOutputStream(tmp.outputStream()).use { out ->
            val header = ByteBuffer.allocate(16).order(ByteOrder.LITTLE_ENDIAN)
            header.putInt(ShardFormat.MAGIC).putInt(ShardFormat.VERSION)
                .putInt(ShardFormat.RECORD_FLOATS).putInt(records.size)
            out.write(header.array())
            val buf = ByteBuffer.allocate(ShardFormat.RECORD_FLOATS * 4).order(ByteOrder.LITTLE_ENDIAN)
            for (rec in records) {
                buf.clear()
                for (v in rec) buf.putFloat(v)
                out.write(buf.array())
            }
        }
        if (path.exists()) path.delete()
        if (!tmp.renameTo(path)) throw IllegalStateException("Failed to rename ${tmp.name} -> ${path.name}")
    }
}
