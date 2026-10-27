package dev.ashik.sift.data.sync

import android.util.Log
import androidx.room3.withWriteTransaction
import dev.ashik.sift.data.contacts.ContactsRepository
import dev.ashik.sift.data.db.AppDatabase
import dev.ashik.sift.data.db.MessageKey
import dev.ashik.sift.data.db.MsgType
import dev.ashik.sift.data.repo.ConversationProjector
import dev.ashik.sift.data.repo.MessageMapping
import dev.ashik.sift.data.repo.MessageRepository
import dev.ashik.sift.spam.work.WorkScheduler
import dev.ashik.sift.telephony.DefaultSmsRole
import dev.ashik.sift.telephony.ProviderSms
import dev.ashik.sift.telephony.SmsProviderGateway
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext

/** What a merge-join of provider keys against Room keys says to do. Pure, so it's unit-tested. */
data class ReconcilePlan(
    val inserts: List<Long> = emptyList(),
    /** The provider reused these ids for different messages: drop local state and re-import. */
    val replaces: List<Long> = emptyList(),
    val updates: List<Pair<ProviderSms, MessageKey>> = emptyList(),
    val deletes: List<Long> = emptyList(),
    val markMissing: List<Long> = emptyList(),
    val clearMissing: List<Long> = emptyList(),
) {
    val isEmpty: Boolean
        get() = inserts.isEmpty() && replaces.isEmpty() && updates.isEmpty() && deletes.isEmpty() &&
            markMissing.isEmpty() && clearMissing.isEmpty()
}

object ReconcilePlanner {
    /** Rows missing from the provider are kept this long (Android 17 can hide fresh OTP messages for 3 h). */
    const val GRACE_MS = 3L * 60 * 60 * 1000

    /** Both inputs must be sorted by ascending id. */
    fun plan(provider: List<ProviderSms>, room: List<MessageKey>, now: Long): ReconcilePlan {
        val inserts = ArrayList<Long>()
        val replaces = ArrayList<Long>()
        val updates = ArrayList<Pair<ProviderSms, MessageKey>>()
        val deletes = ArrayList<Long>()
        val markMissing = ArrayList<Long>()
        val clearMissing = ArrayList<Long>()
        var i = 0
        var j = 0
        while (i < provider.size || j < room.size) {
            val p = provider.getOrNull(i)
            val r = room.getOrNull(j)
            when {
                r == null || (p != null && p.id < r.id) -> {
                    inserts += p!!.id
                    i++
                }
                p == null || p.id > r.id -> {
                    if (now - r.date >= GRACE_MS) deletes += r.id else if (r.missingSince == null) markMissing += r.id
                    j++
                }
                else -> {
                    if (MessageMapping.fingerprint(p.address, p.date) != r.fingerprint) {
                        replaces += p.id
                    } else {
                        if (MessageMapping.mutableHash(p) != r.mutableHash) updates += p to r
                        if (r.missingSince != null) clearMissing += r.id
                    }
                    i++
                    j++
                }
            }
        }
        return ReconcilePlan(inserts, replaces, updates, deletes, markMissing, clearMissing)
    }
}

/**
 * Keeps the Room mirror in step with the system SMS provider. Our own writes mirror directly; this catches
 * everything else (messages written by the system or another app, edits made while another app was default).
 */
class SyncEngine(
    private val db: AppDatabase,
    private val gateway: SmsProviderGateway,
    private val repository: MessageRepository,
    private val projector: ConversationProjector,
    private val role: DefaultSmsRole,
    private val scheduler: WorkScheduler,
    private val contacts: ContactsRepository,
) {
    private val mutex = Mutex()

    /** First run: the newest 500 messages right away so the UI has content; the rest in the background. */
    suspend fun initialImport() = mutex.withLock {
        if (!role.isHeldNow()) return@withLock
        val newest = withContext(Dispatchers.IO) { gateway.queryNewest(500) }
        repository.mirror(newest, notify = false)
        contacts.refreshAll()
        scheduler.enqueueSync(markInitialDone = true)
    }

    /** Diffs every provider row against the mirror (new, changed and deleted rows) and classifies what it imported. */
    suspend fun fullReconcile() = mutex.withLock {
        if (!role.isHeldNow()) return@withLock
        if (fullLocked() > 0) scheduler.enqueueBackfill()
    }

    private suspend fun fullLocked(): Int {
        val provider = withContext(Dispatchers.IO) { gateway.allKeys() }
        val room = db.messages().allKeys()
        val plan = ReconcilePlanner.plan(provider, room, System.currentTimeMillis())
        if (!plan.isEmpty) Log.i(TAG, "reconcile: +${plan.inserts.size} ~${plan.updates.size} -${plan.deletes.size} replace=${plan.replaces.size}")
        return apply(plan)
    }

    /** Returns how many incoming rows were imported (they need classifying). */
    private suspend fun apply(plan: ReconcilePlan): Int {
        if (plan.isEmpty) return 0
        val now = System.currentTimeMillis()
        if (plan.replaces.isNotEmpty()) repository.removeLocal(plan.replaces)
        val imported = importRows(plan.inserts + plan.replaces)
        if (plan.updates.isNotEmpty()) {
            db.withWriteTransaction {
                val threads = HashSet<Long>()
                for ((p, r) in plan.updates) {
                    db.messages().updateMutable(p.id, p.type, p.status, p.read, p.threadId, MessageMapping.mutableHash(p))
                    threads += p.threadId
                    threads += r.threadId
                }
                projector.rebuild(threads)
            }
        }
        if (plan.deletes.isNotEmpty()) repository.removeLocal(plan.deletes)
        plan.markMissing.chunked(500).forEach { db.messages().markMissing(it, now) }
        plan.clearMissing.chunked(500).forEach { db.messages().clearMissing(it) }
        return imported
    }

    private suspend fun importRows(ids: List<Long>): Int {
        var incoming = 0
        for (chunk in ids.chunked(500)) {
            val rows = withContext(Dispatchers.IO) { gateway.rows(chunk) }
            repository.mirror(rows, notify = false)
            incoming += rows.count { it.type == MsgType.INBOX }
        }
        return incoming
    }

    private companion object {
        const val TAG = "SiftSync"
    }
}
