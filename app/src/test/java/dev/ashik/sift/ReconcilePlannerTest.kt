package dev.ashik.sift

import dev.ashik.sift.data.db.MessageKey
import dev.ashik.sift.data.repo.MessageMapping
import dev.ashik.sift.data.sync.ReconcilePlanner
import dev.ashik.sift.telephony.ProviderSms
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test

class ReconcilePlannerTest {
    private val now = 1_800_000_000_000L
    private val hour = 3_600_000L

    private fun p(id: Long, address: String = "+16505550001", date: Long = now - 10 * hour, read: Boolean = true, type: Int = 1) =
        ProviderSms(id, threadId = 7, address = address, body = null, date = date, type = type, read = read, status = -1)

    private fun r(row: ProviderSms, missingSince: Long? = null) =
        MessageKey(row.id, MessageMapping.fingerprint(row.address, row.date), MessageMapping.mutableHash(row), row.date, row.threadId, missingSince)

    @Test
    fun `identical sides produce an empty plan`() {
        val rows = listOf(p(1), p(2), p(5))
        assertTrue(ReconcilePlanner.plan(rows, rows.map { r(it) }, now).isEmpty)
    }

    @Test
    fun `new provider rows are inserted and mutable changes updated`() {
        val plan = ReconcilePlanner.plan(listOf(p(1, read = true), p(2), p(3)), listOf(r(p(1, read = false)), r(p(2))), now)
        assertEquals(listOf(3L), plan.inserts)
        assertEquals(listOf(1L), plan.updates.map { it.first.id })
        assertTrue(plan.deletes.isEmpty())
    }

    @Test
    fun `a reused id with a different message is replaced`() {
        val old = p(4, address = "+16505550009", date = now - 50 * hour)
        val plan = ReconcilePlanner.plan(listOf(p(4)), listOf(r(old)), now)
        assertEquals(listOf(4L), plan.replaces)
        assertTrue(plan.updates.isEmpty())
    }

    @Test
    fun `missing rows are deleted only after the 3 hour grace period`() {
        val fresh = p(8, date = now - hour)
        val stale = p(9, date = now - 4 * hour)
        val plan = ReconcilePlanner.plan(emptyList(), listOf(r(fresh), r(stale)), now)
        assertEquals(listOf(8L), plan.markMissing)
        assertEquals(listOf(9L), plan.deletes)
        val again = ReconcilePlanner.plan(emptyList(), listOf(r(fresh, missingSince = now - 1000)), now)
        assertTrue(again.markMissing.isEmpty(), "already marked rows aren't marked again")
    }

    @Test
    fun `a row that reappears clears its missing mark`() {
        val row = p(8, date = now - hour)
        val plan = ReconcilePlanner.plan(listOf(row), listOf(r(row, missingSince = now - 1000)), now)
        assertEquals(listOf(8L), plan.clearMissing)
    }
}
