import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    alias(libs.plugins.kotlin.jvm)
    alias(libs.plugins.kotlin.serialization)
}

java {
    sourceCompatibility = JavaVersion.VERSION_17
    targetCompatibility = JavaVersion.VERSION_17
}

kotlin {
    compilerOptions {
        jvmTarget.set(JvmTarget.JVM_17)
        // This code also runs on Android 14+, so only allow JDK 17 APIs even when built on a newer JDK.
        freeCompilerArgs.add("-Xjdk-release=17")
    }
}

dependencies {
    implementation(libs.kotlinx.coroutines.core)
    implementation(libs.kotlinx.serialization.json)
    // Same ai.onnxruntime API as the Android AAR at the same version; the app supplies the AAR at runtime.
    compileOnly(libs.onnxruntime.jvm)

    testImplementation(libs.onnxruntime.jvm)
    testImplementation(platform(libs.junit.bom))
    testImplementation(libs.junit.jupiter)
    testRuntimeOnly(libs.junit.platform.launcher)
    testImplementation(libs.kotlinx.coroutines.test)
}

tasks.test {
    useJUnitPlatform()
    maxHeapSize = "3g"
    // GoldenBundleTest runs the packaged model from the app's assets.
    val bundleDir = rootProject.layout.projectDirectory.dir("app/src/main/assets/spam_model").asFile
    systemProperty("sift.bundleDir", bundleDir.path)
    inputs.files(fileTree(bundleDir)).withPropertyName("spamModelBundle").optional()
    val bundleGolden = rootProject.layout.projectDirectory.file("app/src/androidTest/assets/golden/bundle_v1.jsonl.gz").asFile
    systemProperty("sift.bundleGolden", bundleGolden.path)
    inputs.files(bundleGolden).withPropertyName("bundleGolden").optional()
    testLogging {
        events("failed", "skipped")
        exceptionFormat = org.gradle.api.tasks.testing.logging.TestExceptionFormat.FULL
        showStandardStreams = false
    }
}
