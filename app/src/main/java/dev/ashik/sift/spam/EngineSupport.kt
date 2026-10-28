package dev.ashik.sift.spam

import android.content.Context
import dev.ashik.sift.data.db.AppDatabase
import dev.ashik.sift.data.db.LabelCode
import dev.ashik.sift.data.db.TrainingExampleEntity
import dev.ashik.sift.data.db.UserLabelCode
import dev.ashik.sift.spamcore.ExampleStore
import dev.ashik.sift.spamcore.SenderKind
import dev.ashik.sift.spamcore.SpamEngine
import dev.ashik.sift.spamcore.TrainingExample
import android.content.res.AssetManager
import android.util.Log
import dev.ashik.sift.spamcore.encoder.OrtTextEncoder
import dev.ashik.sift.spamcore.engine.ModelSpamEngine
import dev.ashik.sift.spamcore.model.BundleSource
import kotlinx.coroutines.CoroutineScope
import java.io.File
import java.io.FileInputStream
import java.io.IOException
import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

/**
 * The single place the app creates its spam engine: the on-device model from `assets/spam_model` (see
 * ml/scripts/s11_package.py), its personalized weights in [stateDir], and the Room-backed [store].
 */
object SpamEngineFactory {
    const val BUNDLE_DIR = "spam_model"

    fun create(context: Context, stateDir: File, store: ExampleStore, scope: CoroutineScope): SpamEngine {
        stateDir.mkdirs()
        return ModelSpamEngine(
            source = AssetBundleSource(context.assets, BUNDLE_DIR),
            store = store,
            stateDir = stateDir,
            scope = scope,
            encoderFactory = { OrtTextEncoder.create(it) },
            log = { message, error -> Log.w("SiftEngine", message, error) },
        )
    }
}

/**
 * Model bundle files from the APK's assets. The encoder is stored uncompressed (`noCompress += "ort"`), so it can
 * be memory-mapped straight out of the APK instead of copied onto the heap.
 */
class AssetBundleSource(private val assets: AssetManager, private val dir: String) : BundleSource {
    override fun open(name: String): InputStream = assets.open("$dir/$name")

    override fun map(name: String): ByteBuffer? = try {
        assets.openFd("$dir/$name").use { fd ->
            FileInputStream(fd.fileDescriptor).use { input ->
                input.channel.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
            }
        }
    } catch (_: IOException) {
        null // compressed asset: the engine falls back to reading the bytes
    }
}

/** float32 little-endian ⇄ ByteArray for Room BLOBs. */
object EmbeddingCodec {
    fun encode(v: FloatArray): ByteArray {
        val buf = ByteBuffer.allocate(v.size * 4).order(ByteOrder.LITTLE_ENDIAN)
        buf.asFloatBuffer().put(v)
        return buf.array()
    }

    fun decode(bytes: ByteArray): FloatArray {
        val fb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
        return FloatArray(fb.remaining()).also { fb.get(it) }
    }
}

/** spamcore's [ExampleStore] on top of Room. */
class RoomExampleStore(private val db: AppDatabase) : ExampleStore {
    override suspend fun loadAll(): List<TrainingExample> = db.trainingExamples().all().map { it.toExample() }

    override suspend fun upsert(examples: List<TrainingExample>) {
        if (examples.isEmpty()) return
        db.trainingExamples().upsertAll(examples.map { it.toEntity() })
    }

    override suspend fun delete(messageIds: Collection<Long>) {
        messageIds.toList().chunked(500).forEach { db.trainingExamples().deleteIds(it) }
    }

    private fun TrainingExampleEntity.toExample() = TrainingExample(
        messageId = messageId,
        body = body,
        senderKind = runCatching { SenderKind.valueOf(senderKind) }.getOrDefault(SenderKind.UNKNOWN),
        international = international,
        firstContact = firstContact,
        label = UserLabelCode.toUserLabel(label),
        createdAt = createdAt,
        embedding = embedding?.let(EmbeddingCodec::decode),
        senderKey = senderKey,
    )

    private fun TrainingExample.toEntity() = TrainingExampleEntity(
        messageId = messageId,
        body = body,
        senderKey = senderKey,
        senderKind = senderKind.name,
        international = international,
        firstContact = firstContact,
        label = UserLabelCode.of(label),
        createdAt = createdAt,
        embedding = embedding?.let(EmbeddingCodec::encode),
    )
}

/** Turns a user label and an engine verdict into the label that decides folders. */
object FolderPolicy {
    /** The user's label on the message wins; otherwise the engine's; unclassified messages count as not spam. */
    fun effectiveLabel(userLabel: Int?, engineLabel: Int?): Int = when (userLabel) {
        null -> engineLabel ?: LabelCode.HAM
        else -> LabelCode.of(UserLabelCode.toUserLabel(userLabel))
    }
}

/** Messages a rescan moved between Inbox and Spam (ones that appeared or disappeared meanwhile don't count). */
data class FolderMoves(val toSpam: Int, val toInbox: Int) {
    companion object {
        /** [before] and [after] map message ids to folders. */
        fun between(before: Map<Long, Int>, after: Map<Long, Int>): FolderMoves {
            var toSpam = 0
            var toInbox = 0
            for ((messageId, was) in before) {
                val now = after[messageId] ?: continue
                if (now != was) if (now == dev.ashik.sift.data.db.Folder.SPAM) toSpam++ else toInbox++
            }
            return FolderMoves(toSpam, toInbox)
        }
    }
}

/** Classification latencies (ms) of encoder runs since process start, for the model stats screen. */
class LatencyStats(private val capacity: Int = 512) {
    private val samples = LongArray(capacity)
    private var size = 0
    private var next = 0

    @Synchronized
    fun record(ms: Long) {
        samples[next] = ms
        next = (next + 1) % capacity
        if (size < capacity) size++
    }

    /** (p50, p95, count) or null when nothing was measured. */
    @Synchronized
    fun summary(): Triple<Long, Long, Int>? {
        if (size == 0) return null
        val sorted = samples.copyOf(size).sorted()
        fun pct(p: Double) = sorted[((sorted.size - 1) * p).toInt()]
        return Triple(pct(0.50), pct(0.95), size)
    }
}
