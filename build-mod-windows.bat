@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"
title Voxy Mod Builder

echo ========================================
echo   Voxy one-click mod build - Windows
echo ========================================
echo.

where java >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Java was not found in PATH.
    echo Install JDK 25 or newer and set JAVA_HOME/PATH, then run this file again.
    goto :failed
)

set "JAVA_SPEC="
for /f "tokens=2 delims==" %%V in ('java -XshowSettings:properties -version 2^>^&1 ^| findstr /C:"java.specification.version ="') do set "JAVA_SPEC=%%V"
for /f "tokens=* delims= " %%V in ("!JAVA_SPEC!") do set "JAVA_SPEC=%%V"
for /f "tokens=1 delims=." %%V in ("!JAVA_SPEC!") do set "JAVA_MAJOR=%%V"
if not defined JAVA_MAJOR (
    echo [ERROR] Could not determine the Java version.
    goto :failed
)
if !JAVA_MAJOR! LSS 25 (
    echo [ERROR] JDK 25 or newer is required. Current Java specification version: !JAVA_SPEC!
    goto :failed
)

if not exist "gradlew.bat" (
    echo [ERROR] gradlew.bat was not found. Run this script from the Voxy repository root.
    goto :failed
)

echo [1/3] Building Voxy with Gradle...
rem Build the installable mod for the current platform. Do not force
rem includeOtherArchs here: that option is for dedicated universal/release
rem packaging and can make Loom resolve foreign Minecraft runtime natives.
call gradlew.bat clean build --stacktrace
if errorlevel 1 goto :gradle_failed

echo [2/3] Locating the remapped mod JAR...
set "MOD_JAR="
for /f "delims=" %%F in ('dir /b /a-d /o-d "build\libs\*.jar" 2^>nul') do (
    set "NAME=%%F"
    echo(!NAME!| findstr /I /E /C:"-sources.jar" /C:"-dev.jar" /C:"-javadoc.jar" >nul
    if errorlevel 1 if not defined MOD_JAR set "MOD_JAR=build\libs\%%F"
)

if not defined MOD_JAR (
    echo [ERROR] Build completed but no installable JAR was found under build\libs.
    goto :failed
)

echo [3/3] Copying mod JAR to dist\...
if not exist "dist" mkdir "dist"
del /q "dist\*.jar" >nul 2>&1
for %%F in ("!MOD_JAR!") do set "OUT_NAME=%%~nxF"
copy /y "!MOD_JAR!" "dist\!OUT_NAME!" >nul
if errorlevel 1 goto :failed

echo.
echo [SUCCESS] Mod JAR created:
for %%F in ("dist\!OUT_NAME!") do echo   %%~fF
echo.
echo This is the Fabric mod JAR. Put it in your Minecraft mods folder together with its required dependencies.
echo.
pause
exit /b 0

:gradle_failed
echo.
echo [ERROR] Gradle build failed. Java 25+ was already verified; check the dependency/build error above.
:failed
echo.
pause
exit /b 1
