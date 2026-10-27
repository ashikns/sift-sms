# Source this before running Gradle or adb from a shell: `source scripts/env.sh`.
# Uses Android Studio's bundled JDK (the system JDK 27 is outside the Kotlin/AGP tested range)
# and puts the SDK's adb ahead of the system android-tools adb so only one adb server version runs.
export JAVA_HOME="${SIFT_JAVA_HOME:-/opt/android-studio/jbr}"
export ANDROID_HOME="${ANDROID_HOME:-$HOME/Android/Sdk}"
export PATH="$ANDROID_HOME/platform-tools:$ANDROID_HOME/emulator:$JAVA_HOME/bin:$PATH"
