package az

import az.net.OnnxNet
import az.selfplay.SelfPlay
import az.selfplay.ShardFormat
import az.selfplay.ShardWriter
import java.io.File

/**
 * Self-play data generator entry point. Writes one shard per invocation (the distributed
 * orchestration runs many invocations across machines; atomic tmp+rename keeps shards safe).
 *
 * Run: gradlew :app:runSelfPlay -Pargs="<nGames> <outDir> <seedBase> <genTag>"
 * Or via the fat jar on a worker:  java -cp client-server.jar az.RunSelfPlayKt "8 /workspace/az_shards 1 gen0"
 */
fun main(args: Array<String>) {
    val parts = (args.getOrNull(0) ?: "").trim().split(Regex("\\s+")).filter { it.isNotEmpty() }
    val nGames = parts.getOrNull(0)?.toIntOrNull() ?: 4
    val outDir = parts.getOrNull(1) ?: "selfplay_out"
    val seedBase = parts.getOrNull(2)?.toLongOrNull() ?: 1L
    val genTag = parts.getOrNull(3) ?: "gen0"
    val onnxArg = parts.getOrNull(4)                       // path, or "none"/absent for rollout self-play
    val budgetMs = parts.getOrNull(5)?.toLongOrNull() ?: 15L
    val onnxPath = if (onnxArg == null || onnxArg == "none") null else onnxArg

    val net = onnxPath?.let { OnnxNet.fromFile(it) }
    println("[selfplay] nGames=$nGames out=$outDir seedBase=$seedBase gen=$genTag budgetMs=$budgetMs net=${if (net != null) onnxPath else "none(rollout)"} recordFloats=${ShardFormat.RECORD_FLOATS}")
    val sp = SelfPlay(budgetMs = budgetMs, net = net)
    val all = ArrayList<FloatArray>()
    val t0 = System.currentTimeMillis()
    for (g in 0 until nGames) {
        val recs = sp.playGame(seedBase + g)
        all.addAll(recs)
        val dt = (System.currentTimeMillis() - t0) / 1000.0
        println("[selfplay] game ${g + 1}/$nGames seed=${seedBase + g} samples=${recs.size} total=${all.size} elapsed=${"%.1f".format(dt)}s")
    }
    val shard = File(outDir, "$genTag/shard_${seedBase}_${nGames}.bin")
    ShardWriter.write(shard, all)
    val dt = (System.currentTimeMillis() - t0) / 1000.0
    println("[selfplay] DONE wrote ${all.size} samples to ${shard.path} in ${"%.1f".format(dt)}s")
}
