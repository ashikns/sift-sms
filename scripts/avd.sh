#!/usr/bin/env bash
# Creates the `sift_api37` AVD (Pixel 8a-like: 1080x2400, 420 dpi) from the installed API 37 Google Play x86_64
# image by writing its config by hand (Android Studio's command-line tools aren't required).
set -euo pipefail
source "$(dirname "$0")/env.sh"
NAME=${SIFT_AVD:-sift_api37}
IMAGE_REL="system-images/android-37.0/google_apis_playstore/x86_64/"
AVD_HOME=${ANDROID_AVD_HOME:-$HOME/.android/avd}
DIR="$AVD_HOME/$NAME.avd"
if [[ -f "$DIR/config.ini" ]]; then echo "AVD $NAME already exists"; exit 0; fi
[[ -d "$ANDROID_HOME/$IMAGE_REL" ]] || { echo "missing system image $IMAGE_REL" >&2; exit 1; }
mkdir -p "$DIR"
cat > "$AVD_HOME/$NAME.ini" <<INI
avd.ini.encoding=UTF-8
path=$DIR
path.rel=avd/$NAME.avd
target=android-37.0
INI
cat > "$DIR/config.ini" <<INI
AvdId=$NAME
PlayStore.enabled=true
abi.type=x86_64
avd.ini.displayname=Sift API 37 (Pixel 8a-like)
avd.ini.encoding=UTF-8
disk.dataPartition.size=6G
fastboot.forceColdBoot=no
fastboot.forceFastBoot=yes
hw.accelerometer=yes
hw.audioInput=no
hw.battery=yes
hw.camera.back=none
hw.camera.front=none
hw.cpu.arch=x86_64
hw.cpu.ncore=4
hw.dPad=no
hw.device.manufacturer=Google
hw.device.name=pixel_8a
hw.gps=yes
hw.gpu.enabled=yes
hw.gpu.mode=swiftshader_indirect
hw.initialOrientation=portrait
hw.keyboard=yes
hw.lcd.density=420
hw.lcd.height=2400
hw.lcd.width=1080
hw.mainKeys=no
hw.ramSize=4096
hw.sdCard=no
hw.sensors.proximity=yes
hw.trackBall=no
image.sysdir.1=$IMAGE_REL
runtime.network.latency=none
runtime.network.speed=full
showDeviceFrame=no
skin.dynamic=yes
tag.display=Google Play
tag.id=google_apis_playstore
target=android-37.0
vm.heapSize=512
INI
echo "created AVD $NAME at $DIR"
