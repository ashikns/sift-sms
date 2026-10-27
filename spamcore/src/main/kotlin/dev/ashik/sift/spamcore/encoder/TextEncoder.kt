package dev.ashik.sift.spamcore.encoder

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import dev.ashik.sift.spamcore.model.ModelBundle
import java.nio.ByteBuffer
import java.nio.LongBuffer

/** Maps token ids (no padding) to the L2-normalized 384-d sentence embedding. Implementations are thread-safe. */
interface TextEncoder : AutoCloseable {
    fun embed(ids: LongArray): FloatArray
}

/**
 * ONNX Runtime encoder on the CPU provider. Compiled against the desktop ORT jar (compileOnly); the Android app ships
 * the AAR, which has the same `ai.onnxruntime` API at the same version.
 */
class OrtTextEncoder private constructor(
    private val env: OrtEnvironment,
    private val session: OrtSession,
    private val inputIds: String,
    private val attentionMask: String,
    private val output: String,
    @Suppress("unused") private val modelBytes: ByteBuffer?, // must outlive the session when ORT uses it in place
) : TextEncoder {

    override fun embed(ids: LongArray): FloatArray {
        val shape = longArrayOf(1, ids.size.toLong())
        OnnxTensor.createTensor(env, LongBuffer.wrap(ids), shape).use { idsT ->
            OnnxTensor.createTensor(env, LongBuffer.wrap(LongArray(ids.size) { 1L }), shape).use { maskT ->
                session.run(mapOf(inputIds to idsT, attentionMask to maskT)).use { result ->
                    val tensor = result.get(output).orElseThrow() as OnnxTensor
                    val fb = tensor.floatBuffer
                    return FloatArray(fb.remaining()).also { fb.get(it) }
                }
            }
        }
    }

    override fun close() = session.close()

    companion object {
        /** Creates the session from the bundle's encoder (memory-mapped when the source allows it) and warms it up. */
        fun create(bundle: ModelBundle, threads: Int = 2): OrtTextEncoder {
            val cfg = bundle.config.encoder
            val env = OrtEnvironment.getEnvironment()
            runCatching { env.setTelemetry(false) }
            val mapped = runCatching { bundle.source.map(cfg.file) }.getOrNull()?.takeIf { it.isDirect }
            val opts = OrtSession.SessionOptions().apply {
                setIntraOpNumThreads(threads)
                setInterOpNumThreads(1)
                setExecutionMode(OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL)
                // The model is already optimized offline (ORT format, fixed optimization style).
                setOptimizationLevel(OrtSession.SessionOptions.OptLevel.NO_OPT)
                addConfigEntry("session.intra_op.allow_spinning", "0")
                if (mapped != null) {
                    addConfigEntry("session.use_ort_model_bytes_directly", "1")
                    addConfigEntry("session.use_ort_model_bytes_for_initializers", "1")
                }
            }
            val session = opts.use {
                if (mapped != null) env.createSession(mapped, it)
                else env.createSession(bundle.source.open(cfg.file).use { s -> s.readBytes() }, it)
            }
            val encoder = OrtTextEncoder(env, session, cfg.inputs[0], cfg.inputs[1], cfg.output, mapped)
            encoder.embed(longArrayOf(101, 102)) // warm-up run
            return encoder
        }
    }
}
