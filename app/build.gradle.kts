import com.android.build.api.artifact.SingleArtifact

plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.compose)
    alias(libs.plugins.kotlin.serialization)
    alias(libs.plugins.ksp)
    alias(libs.plugins.room3)
}

android {
    namespace = "dev.ashik.sift"
    compileSdk = 37

    defaultConfig {
        applicationId = "dev.ashik.sift"
        minSdk = 34
        targetSdk = 37
        versionCode = 1
        versionName = "0.1.0"
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        // The Pixel 8a is arm64; debug builds add x86_64 for the emulator.
        ndk { abiFilters += listOf("arm64-v8a") }
    }

    buildTypes {
        debug {
            ndk { abiFilters += listOf("x86_64") }
        }
        release {
            isMinifyEnabled = true
            isShrinkResources = true
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
            // Personal sideload build: signed with the debug key.
            signingConfig = signingConfigs.getByName("debug")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    buildFeatures {
        compose = true
    }

    androidResources {
        // The ONNX Runtime model is memory-mapped straight from the APK, so it must stay uncompressed.
        noCompress += listOf("ort")
    }

    testOptions {
        unitTests.all { it.useJUnitPlatform() }
    }
}

room3 {
    schemaDirectory("$projectDir/schemas")
}

/** Fails the build if the merged manifest requests INTERNET: the app must never be able to reach the network. */
abstract class CheckNoInternetTask : DefaultTask() {
    @get:InputFile
    @get:PathSensitive(PathSensitivity.NONE)
    abstract val mergedManifest: RegularFileProperty

    @get:OutputFile
    abstract val marker: RegularFileProperty

    @TaskAction
    fun run() {
        val text = mergedManifest.get().asFile.readText()
        check("android.permission.INTERNET" !in text) {
            "Merged manifest ${mergedManifest.get().asFile} declares android.permission.INTERNET"
        }
        marker.get().asFile.writeText("ok\n")
    }
}

/**
 * Fails a shrunk build if R8 stripped OrtTextEncoder.modelBytes. ONNX Runtime reads the model's weights straight from
 * that memory mapping; without the reference the mapping is freed on the next GC and inference segfaults.
 */
abstract class CheckEncoderBufferKeptTask : DefaultTask() {
    /** R8's usage.txt, written next to mapping.txt: everything R8 removed. */
    @get:InputFile
    @get:PathSensitive(PathSensitivity.NONE)
    abstract val usage: RegularFileProperty

    @get:OutputFile
    abstract val marker: RegularFileProperty

    @TaskAction
    fun run() {
        val lines = usage.get().asFile.readLines()
        val start = lines.indexOfFirst { it.startsWith("dev.ashik.sift.spamcore.encoder.OrtTextEncoder:") }
        val removed = if (start < 0) emptyList() else lines.drop(start + 1).takeWhile { it.startsWith(" ") }
        check(removed.none { "modelBytes" in it }) {
            "R8 removed OrtTextEncoder.modelBytes: the memory-mapped model would be freed while ONNX Runtime uses it"
        }
        marker.get().asFile.writeText("ok\n")
    }
}

androidComponents {
    onVariants { variant ->
        val name = variant.name.replaceFirstChar(Char::uppercase)
        val check = tasks.register<CheckNoInternetTask>("check${name}NoInternet") {
            mergedManifest.set(variant.artifacts.get(SingleArtifact.MERGED_MANIFEST))
            marker.set(layout.buildDirectory.file("intermediates/no_internet/${variant.name}.ok"))
        }
        tasks.matching { it.name == "package$name" }.configureEach { dependsOn(check) }
        if (variant.isMinifyEnabled) {
            val keep = tasks.register<CheckEncoderBufferKeptTask>("check${name}EncoderBufferKept") {
                val mapping = variant.artifacts.get(SingleArtifact.OBFUSCATION_MAPPING_FILE)
                usage.fileProvider(mapping.map { it.asFile.resolveSibling("usage.txt") })
                marker.set(layout.buildDirectory.file("intermediates/encoder_buffer/${variant.name}.ok"))
            }
            tasks.matching { it.name == "package$name" }.configureEach { dependsOn(keep) }
        }
    }
}

dependencies {
    implementation(project(":spamcore"))

    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.activity.compose)
    implementation(libs.androidx.lifecycle.runtime.compose)
    implementation(libs.androidx.lifecycle.viewmodel.compose)
    implementation(libs.androidx.lifecycle.viewmodel.navigation3)
    implementation(libs.androidx.lifecycle.process)
    implementation(libs.androidx.navigation3.runtime)
    implementation(libs.androidx.navigation3.ui)

    implementation(platform(libs.compose.bom))
    implementation(libs.compose.ui)
    implementation(libs.compose.ui.tooling.preview)
    implementation(libs.compose.material3)
    implementation(libs.compose.material.icons.core)
    debugImplementation(libs.compose.ui.tooling)

    implementation(libs.room3.runtime)
    ksp(libs.room3.compiler)
    implementation(libs.sqlite.bundled)
    implementation(libs.work.runtime)
    implementation(libs.datastore.preferences)
    implementation(libs.kotlinx.coroutines.android)
    implementation(libs.kotlinx.serialization.core)
    implementation(libs.kotlinx.serialization.json)
    implementation(libs.onnxruntime.android)

    testImplementation(platform(libs.junit.bom))
    testImplementation(libs.junit.jupiter)
    testRuntimeOnly(libs.junit.platform.launcher)
    testImplementation(libs.kotlinx.coroutines.test)

    androidTestImplementation(libs.junit4)
    androidTestImplementation(libs.androidx.test.runner)
    androidTestImplementation(libs.androidx.test.core)
    androidTestImplementation(libs.androidx.test.ext.junit)
    androidTestImplementation(libs.androidx.test.uiautomator)
    androidTestImplementation(libs.androidx.test.espresso.core)
    androidTestImplementation(libs.kotlinx.coroutines.test)
    androidTestImplementation(libs.room3.testing)
    androidTestImplementation(platform(libs.compose.bom))
    androidTestImplementation(libs.compose.ui.test.junit4)
    debugImplementation(libs.compose.ui.test.manifest)
}
