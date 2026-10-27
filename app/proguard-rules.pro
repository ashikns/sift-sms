# ONNX Runtime's Java API is called from JNI and ships no consumer rules.
-keep class ai.onnxruntime.** { *; }

# ONNX Runtime reads the encoder's weights straight from the memory-mapped model file, so the encoder's reference to
# that mapping must survive shrinking. Without it the mapping is released on the next GC and ORT reads freed memory.
-keepclassmembers class dev.ashik.sift.spamcore.encoder.OrtTextEncoder {
    java.nio.ByteBuffer modelBytes;
}
