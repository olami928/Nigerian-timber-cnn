# Android APK build (P-F) — Timber ID

This document covers the Android build environment, debug/release APK generation, signing, installation, and CI artifact builds. **No APK is claimed as generated unless the corresponding build command succeeds.**

## Current workspace status (this environment)

- **Checked:** no `java/javac/keytool`, no Android SDK/`adb`/`sdkmanager`, `JAVA_HOME`/`ANDROID_HOME` unset, no privileged install access (`sudo` unavailable). Docker available but not used for build.
- **Result:** `assembleDebug` **was not run here** and **no APK was generated in this environment** because required build dependencies are missing.
- **Action taken:** a GitHub Actions workflow (`.github/workflows/android-apk.yml`) is provided to produce a downloadable debug APK artifact when run on GitHub-hosted runners.

## Application identity (Capacitor/Android)

| Field | Value | Source |
|---|---|---|
| App name | Timber ID | `app/capacitor.config.json` (`appName`) |
| Application ID / namespace | `com.timberid.nigeriantimberid` | `app/capacitor.config.json`, `app/android/app/build.gradle` |
| Version name | `1.0` | `app/android/app/build.gradle` (`versionName`) |
| Version code | `1` | `app/android/app/build.gradle` (`versionCode`) |
| Web assets dir | `app/public` → packaged to `android/app/src/main/assets/public/` | Capacitor `webDir`, build/sync |

Update `versionCode`/`versionName` (and keep consistent with releases) before distributing new builds.

## Build environment requirements

| Component | Required / used by project | Notes |
|---|---|---|
| JDK | **JDK 17** | Required for AGP 8.2.x / Gradle 8.2.x. Set `JAVA_HOME`. |
| Android SDK | Android SDK with command-line tools | Set `ANDROID_HOME` or `ANDROID_SDK_ROOT`. Create `app/android/local.properties` with `sdk.dir=/path/to/sdk` if needed (do not commit). |
| SDK Platform | **Android API 34** (`platforms;android-34`) | `compileSdkVersion` / `targetSdkVersion` = 34 (`app/android/variables.gradle`). |
| Build Tools | **34.0.0** (recommended) | `sdkmanager "build-tools;34.0.0"` |
| Platform Tools | latest | for `adb` install/testing |
| Gradle | **8.2.1** (wrapper) | `app/android/gradle/wrapper/gradle-wrapper.properties` (`gradle-8.2.1-all.zip`) |
| Android Gradle Plugin | **8.2.1** | `app/android/build.gradle` (`com.android.tools.build:gradle:8.2.1`) |
| Node.js / npm | Node 20 LTS recommended | to build web assets (`app/package-lock.json` present) |
| minSdk | **22** | `app/android/variables.gradle` |

Reference files: `app/android/build.gradle`, `app/android/variables.gradle`, `app/android/app/build.gradle`, `app/android/gradle/wrapper/gradle-wrapper.properties`.

## Prepare web assets and Android package (always before APK build)

From repository root:

```sh
cd app
npm ci
npm run build
rm -rf android/app/src/main/assets/public
cp -R public android/app/src/main/assets/public
```

This regenerates the offline `species.json` from authoritative sources, copies `model.onnx`/configs/local ONNX Runtime WASM, rebuilds the bundle, and synchronizes packaged Android assets. (On a fully configured Capacitor/SDK machine you may alternatively use `npx cap sync android`.)

**Do not modify** model weights, `class_names.json` ordering, `preprocess_config.json` preprocessing values, or inference/aggregation logic.

## Debug APK (installable)

```sh
cd app/android
chmod +x gradlew
./gradlew --no-daemon assembleDebug
```

- **Expected output (on success):** `app/android/app/build/outputs/apk/debug/app-debug.apk`
- **Verify:** `ls -lh app/android/app/build/outputs/apk/debug/app-debug.apk`
- Upload/share that file as the downloadable debug artifact for phone installation. **Only report “built/generated” if this file exists and the Gradle task succeeded.**

## Signed release APK (`assembleRelease`)

Release signing is configured in `app/android/app/build.gradle` to read `app/android/keystore.properties` **only if present** (file is gitignored). Unsigned/CI debug builds are unaffected.

### 1. Create a keystore (secure, local only)

```sh
keytool -genkeypair -v -keystore timberid-release.keystore -alias timberid \
  -keyalg RSA -keysize 2048 -validity 10000
```

Store the keystore outside the repo or in a secure location; **never commit** `.keystore`/`.jks` files or passwords.

### 2. Configure `keystore.properties` (local)

```sh
cp app/android/keystore.properties.example app/android/keystore.properties
# edit values; keep file uncommitted
```

`app/android/keystore.properties.example` documents required keys: `storeFile`, `storePassword`, `keyAlias`, `keyPassword`.

### 3. Build release

```sh
cd app/android
./gradlew --no-daemon assembleRelease
```

- **Expected output (on success, when signed):** `app/android/app/build/outputs/apk/release/app-release.apk`
- If `keystore.properties` is absent, the release build may produce an **unsigned** APK depending on Gradle behavior; for distribution use a properly signed release artifact.

### Secure CI signing (GitHub Actions) — instructions

Do **not** commit credentials. To produce a signed release in CI:
1. Base64-encode the keystore and store as a repository secret (e.g. `ANDROID_KEYSTORE_BASE64`).
2. Store `ANDROID_KEYSTORE_PASSWORD`, `ANDROID_KEY_ALIAS`, `ANDROID_KEY_PASSWORD` as secrets.
3. In a workflow step, decode the keystore to a runner temp path and generate `app/android/keystore.properties` pointing to it (file remains only in runner workspace).
4. Run `./gradlew assembleRelease` and upload `app-release.apk` as an artifact.

No signing secrets are configured in this repository.

## GitHub Actions workflow (downloadable debug APK artifact)

- Workflow: `.github/workflows/android-apk.yml`
- Triggers: `workflow_dispatch` (manual) and pushes to `main` affecting app/config/build inputs.
- Runner: `ubuntu-latest` with Temurin JDK 17, Android SDK (API 34, build-tools 34.0.0), Node 20.
- Steps: build web assets → sync Android assets → `assembleDebug` → upload artifact **`timber-id-debug-apk`** containing `app-debug.apk`.
- **Download:** GitHub repo → Actions → select successful run → Artifacts → `timber-id-debug-apk`. The artifact is only available if the workflow run succeeds.

## Installing the debug APK on an Android phone

1. Download `app-debug.apk` (from a successful local build or the GitHub Actions artifact; unzip the artifact if downloaded as a zip).
2. Transfer to the phone (USB, Drive, email, etc.).
3. On the phone, enable installation from the source you use (e.g. allow your browser/file manager to install unknown apps) — Android will prompt.
4. Open the APK and tap **Install**. If a previous debug signature conflicts, uninstall the old Timber ID build first.
5. Launch **Timber ID**. For P-F deliverable this confirms installability; **offline/camera/inference/device validation is P-G** (do not claim tested until performed).

## Offline/model integrity notes

- All inference assets (ONNX model, `class_names.json`, `preprocess_config.json`, local ORT WASM, bundled JS/CSS, `species.json`) are packaged under `android/app/src/main/assets/public/`.
- Android manifest declares no `INTERNET` permission (offline-oriented). Camera/media permissions are declared for capture/gallery flows (to be validated in P-G).
- **Preserved:** model weights, preprocessing, class ordering, and aggregation logic unchanged by P-F.

## Known limitations (build)

- Local APK generation in this workspace is blocked by missing JDK/Android SDK; CI workflow provided instead.
- Release APK artifact requires signing credentials/secrets not present here; secure config/instructions provided.
- No device/emulator testing performed in P-F. P-G (parity, camera/gallery, airplane-mode on real devices) not started.
