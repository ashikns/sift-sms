package dev.ashik.sift

import android.app.Application
import dev.ashik.sift.di.AppGraph
import kotlinx.coroutines.launch

class SiftApp : Application() {
    lateinit var graph: AppGraph
        private set

    override fun onCreate() {
        super.onCreate()
        graph = AppGraph(this)
        graph.notifier.ensureChannels()
        // Load the model early: a process started by SMS_DELIVER needs it within the receiver's budget.
        graph.appScope.launch { runCatching { graph.engine.warmUp() } }
        if (graph.role.isHeldNow()) graph.observer.register()
    }
}
