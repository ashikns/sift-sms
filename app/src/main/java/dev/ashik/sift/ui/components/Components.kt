package dev.ashik.sift.ui.components

import android.content.Context
import android.graphics.BitmapFactory
import android.net.Uri
import android.telephony.PhoneNumberUtils
import android.text.format.DateUtils
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Person
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.produceState
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import dev.ashik.sift.SiftApp
import dev.ashik.sift.di.AppGraph
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.util.Calendar
import java.util.Locale

@Composable
fun rememberGraph(): AppGraph = (LocalContext.current.applicationContext as SiftApp).graph

fun displayAddress(address: String): String =
    PhoneNumberUtils.formatNumber(address, Locale.getDefault().country.ifEmpty { "US" }) ?: address

/** "3:41 PM" today, "Tue" this week, "Oct 3" this year, else "10/3/24". */
fun shortTime(context: Context, ts: Long, now: Long = System.currentTimeMillis()): String {
    val flags = when {
        DateUtils.isToday(ts) -> DateUtils.FORMAT_SHOW_TIME
        now - ts < 6 * DateUtils.DAY_IN_MILLIS -> DateUtils.FORMAT_SHOW_WEEKDAY or DateUtils.FORMAT_ABBREV_WEEKDAY
        sameYear(ts, now) -> DateUtils.FORMAT_SHOW_DATE or DateUtils.FORMAT_NO_YEAR or DateUtils.FORMAT_ABBREV_MONTH
        else -> DateUtils.FORMAT_SHOW_DATE or DateUtils.FORMAT_NUMERIC_DATE
    }
    return DateUtils.formatDateTime(context, ts, flags)
}

fun dayHeader(context: Context, ts: Long): String = DateUtils.formatDateTime(
    context, ts,
    DateUtils.FORMAT_SHOW_DATE or DateUtils.FORMAT_SHOW_WEEKDAY or DateUtils.FORMAT_ABBREV_MONTH,
)

fun timeOfDay(context: Context, ts: Long): String = DateUtils.formatDateTime(context, ts, DateUtils.FORMAT_SHOW_TIME)

fun sameDay(a: Long, b: Long): Boolean {
    val ca = Calendar.getInstance().apply { timeInMillis = a }
    val cb = Calendar.getInstance().apply { timeInMillis = b }
    return ca.get(Calendar.YEAR) == cb.get(Calendar.YEAR) && ca.get(Calendar.DAY_OF_YEAR) == cb.get(Calendar.DAY_OF_YEAR)
}

private fun sameYear(a: Long, b: Long): Boolean =
    Calendar.getInstance().apply { timeInMillis = a }.get(Calendar.YEAR) == Calendar.getInstance().apply { timeInMillis = b }.get(Calendar.YEAR)

/** The contact photo, else the first letter on a color derived from the sender, else (numbers) a person glyph. */
@Composable
fun Avatar(name: String, key: String, photoUri: String?, size: Dp = 44.dp) {
    val context = LocalContext.current
    val photo = produceState<ImageBitmap?>(null, photoUri) {
        value = photoUri?.let { uri ->
            withContext(Dispatchers.IO) {
                runCatching {
                    context.contentResolver.openInputStream(Uri.parse(uri))?.use { BitmapFactory.decodeStream(it)?.asImageBitmap() }
                }.getOrNull()
            }
        }
    }.value
    if (photo != null) {
        Image(photo, contentDescription = null, contentScale = ContentScale.Crop, modifier = Modifier.size(size).clip(CircleShape))
        return
    }
    val palette = listOf(
        Color(0xFF1B6B5A), Color(0xFF3F51B5), Color(0xFF8E24AA), Color(0xFFD84315), Color(0xFF00838F),
        Color(0xFF6D4C41), Color(0xFF2E7D32), Color(0xFFAD1457),
    )
    val color = palette[(key.hashCode() and 0x7fffffff) % palette.size]
    // Unknown numbers have no meaningful initial (it would be an area-code digit), so show a person glyph instead.
    val initial = name.firstOrNull { it.isLetter() }?.uppercase()
    Box(Modifier.size(size).clip(CircleShape).background(color), contentAlignment = Alignment.Center) {
        if (initial != null) {
            Text(initial, color = Color.White, style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
        } else {
            Icon(Icons.Filled.Person, contentDescription = null, tint = Color.White, modifier = Modifier.size(size * 0.6f))
        }
    }
}
