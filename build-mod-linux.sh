#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

echo "========================================"
echo "  Voxy one-click mod build - Linux"
echo "========================================"
echo

if ! command -v java >/dev/null 2>&1; then
    echo "[ERROR] Java was not found in PATH." >&2
    echo "Install JDK 25 or newer and set JAVA_HOME/PATH, then run this script again." >&2
    exit 1
fi

JAVA_SPEC="$(java -XshowSettings:properties -version 2>&1 | awk -F'= ' '/java.specification.version =/ {print $2; exit}')"
JAVA_MAJOR="${JAVA_SPEC%%.*}"
if [[ -z "$JAVA_MAJOR" || ! "$JAVA_MAJOR" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] Could not determine the Java version." >&2
    exit 1
fi
if (( JAVA_MAJOR < 25 )); then
    echo "[ERROR] JDK 25 or newer is required. Current Java specification version: $JAVA_SPEC" >&2
    exit 1
fi

if [[ ! -f ./gradlew ]]; then
    echo "[ERROR] ./gradlew was not found. Run this script from the Voxy repository root." >&2
    exit 1
fi
chmod +x ./gradlew 2>/dev/null || true

echo "[1/3] Building Voxy with Gradle..."
# Build for the current platform. The optional includeOtherArchs flag is reserved
# for dedicated universal/release packaging and should not be forced here.
./gradlew clean build --stacktrace "$@"

echo "[2/3] Locating the remapped mod JAR..."
MOD_JAR=""
while IFS= read -r jar; do
    case "$jar" in
        *-sources.jar|*-dev.jar|*-javadoc.jar) continue ;;
    esac
    MOD_JAR="$jar"
    break
done < <(ls -1t build/libs/*.jar 2>/dev/null || true)

if [[ -z "$MOD_JAR" || ! -f "$MOD_JAR" ]]; then
    echo "[ERROR] Build completed but no installable JAR was found under build/libs." >&2
    exit 1
fi

echo "[3/3] Copying mod JAR to dist/..."
mkdir -p dist
rm -f dist/*.jar
OUT="dist/$(basename "$MOD_JAR")"
cp -f "$MOD_JAR" "$OUT"

echo
echo "[SUCCESS] Mod JAR created:"
printf '  %s\n' "$ROOT/$OUT"
echo
echo "This is the Fabric mod JAR. Put it in your Minecraft mods folder together with its required dependencies."
