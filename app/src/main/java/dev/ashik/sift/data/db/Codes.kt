package dev.ashik.sift.data.db

import dev.ashik.sift.spamcore.SpamLabel
import dev.ashik.sift.spamcore.UserLabel

/** Telephony.Sms.MESSAGE_TYPE_* values as stored by the system SMS provider. */
object MsgType {
    const val INBOX = 1
    const val SENT = 2
    const val DRAFT = 3
    const val OUTBOX = 4
    const val FAILED = 5
    const val QUEUED = 6
}

/** Telephony.Sms.STATUS_* values. */
object MsgStatus {
    const val COMPLETE = 0
    const val FAILED = 64
}

/** Stored label codes: identical to [SpamLabel] ordinals (HAM, SUSPICIOUS, SPAM). */
object LabelCode {
    const val HAM = 0
    const val SUSPICIOUS = 1
    const val SPAM = 2

    fun of(label: SpamLabel) = label.ordinal
    fun of(label: UserLabel) = if (label == UserLabel.SPAM) SPAM else HAM
}

/** Stored user-label codes: identical to [UserLabel] ordinals (HAM, SPAM). */
object UserLabelCode {
    fun of(label: UserLabel) = label.ordinal
    fun toUserLabel(code: Int) = UserLabel.entries.getOrElse(code) { UserLabel.HAM }
}

object Folder {
    const val INBOX = 0
    const val SPAM = 1
}

/** Where a message's effective label came from. */
object Source {
    const val PENDING = "PENDING"
    const val ENGINE = "ENGINE"
    const val USER = "USER"
    const val FAIL_OPEN = "FAIL_OPEN"
}

object ClassState {
    const val PENDING = 0
    const val DONE = 1
}

object FeedbackKind {
    /** The user marked this message. */
    const val EXPLICIT = 0

    /** A similar message that moved because of an explicit mark (undone together). */
    const val AUTO_MOVE = 1
}
