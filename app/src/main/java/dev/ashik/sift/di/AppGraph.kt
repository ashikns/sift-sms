package dev.ashik.sift.di

import android.content.Context
import android.util.Log
import dev.ashik.sift.data.contacts.ContactsRepository
import dev.ashik.sift.data.db.AppDatabase
import dev.ashik.sift.data.repo.ConversationProjector
import dev.ashik.sift.data.repo.MessageRepository
import dev.ashik.sift.data.settings.SettingsRepository
import dev.ashik.sift.data.sync.SyncEngine
import dev.ashik.sift.incoming.IncomingSmsHandler
import dev.ashik.sift.notifications.ConversationShortcuts
import dev.ashik.sift.notifications.MessageNotifier
import dev.ashik.sift.notifications.VisibleThreadTracker
import dev.ashik.sift.spam.ClassificationCoordinator
import dev.ashik.sift.spam.FeedbackService
import dev.ashik.sift.spam.LatencyStats
import dev.ashik.sift.spam.RescoreService
import dev.ashik.sift.spam.RoomExampleStore
import dev.ashik.sift.spam.SenderInfoProvider
import dev.ashik.sift.spam.SpamEngineFactory
import dev.ashik.sift.spam.work.WorkScheduler
import dev.ashik.sift.spamcore.SpamEngine
import dev.ashik.sift.telephony.AddressNormalizer
import dev.ashik.sift.telephony.DefaultSmsRole
import dev.ashik.sift.telephony.SmsProviderGateway
import dev.ashik.sift.telephony.SmsProviderObserver
import kotlinx.coroutines.CoroutineExceptionHandler
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import java.io.File

/** Manual dependency graph: process-wide lazy singletons. */
class AppGraph(private val context: Context) {
    val appScope = CoroutineScope(
        SupervisorJob() + Dispatchers.Default + CoroutineExceptionHandler { _, t -> Log.e("Sift", "uncaught in appScope", t) },
    )

    val db: AppDatabase by lazy { AppDatabase.build(context) }
    val settings by lazy { SettingsRepository(context) }
    val role by lazy { DefaultSmsRole(context) }
    val normalizer by lazy { AddressNormalizer.forContext(context) }
    val gateway by lazy { SmsProviderGateway(context) }
    val projector by lazy { ConversationProjector(db) }
    val repository by lazy { MessageRepository(db, gateway, normalizer, projector) }
    val contacts by lazy { ContactsRepository(context, db, normalizer) }
    val exampleStore by lazy { RoomExampleStore(db) }
    val engine: SpamEngine by lazy {
        SpamEngineFactory.create(context, File(context.noBackupFilesDir, "spam_state"), exampleStore, appScope)
    }
    val latency = LatencyStats()
    val senderInfo by lazy { SenderInfoProvider(db, normalizer) }
    val coordinator by lazy { ClassificationCoordinator(db, engine, senderInfo, settings, projector, latency) }
    val rescore by lazy { RescoreService(db, coordinator, projector) }
    val visibleThread = VisibleThreadTracker()
    val shortcuts by lazy { ConversationShortcuts(context) }
    val notifier by lazy { MessageNotifier(context, db, shortcuts) }
    val feedback by lazy { FeedbackService(db, engine, rescore, projector, notifier, normalizer, appScope) }
    val scheduler by lazy { WorkScheduler(context) }
    val sync by lazy { SyncEngine(db, gateway, repository, projector, role, scheduler, contacts) }
    val incoming by lazy {
        IncomingSmsHandler(db, gateway, repository, coordinator, notifier, settings, visibleThread, scheduler, appScope)
    }
    val observer by lazy { SmsProviderObserver(context, appScope) { sync.incremental() } }

    private val startupMutex = Mutex()
    private var startupDone = false

    /**
     * Once per process, when the UI starts with the role held: import history on first run, otherwise a full
     * reconcile, plus recovery of interrupted classifications. Never run from a receiver-started process.
     */
    suspend fun onUiStart() = startupMutex.withLock {
        if (!role.refresh()) return@withLock
        observer.register()
        if (startupDone) {
            sync.incremental()
            return@withLock
        }
        startupDone = true
        if (!settings.current().initialImportDone) {
            sync.initialImport()
        } else {
            sync.fullReconcile()
            contacts.refreshAll()
        }
        if (db.classifications().pendingCount() > 0) scheduler.enqueueClassifyPending()
    }
}
