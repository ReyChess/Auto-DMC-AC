@echo off
setlocal EnableExtensions

set "CYGWIN_BASH="
if exist "C:\cygwin64\bin\bash.exe" set "CYGWIN_BASH=C:\cygwin64\bin\bash.exe"
if not defined CYGWIN_BASH if exist "C:\cygwin\bin\bash.exe" set "CYGWIN_BASH=C:\cygwin\bin\bash.exe"

if not defined CYGWIN_BASH (
    echo ERROR: Cygwin bash.exe was not found.
    echo Expected C:\cygwin64\bin\bash.exe or C:\cygwin\bin\bash.exe.
    exit /b 1
)

set "PROJECT_DIR=%~dp0"
echo Using Cygwin: %CYGWIN_BASH%
echo Project: %PROJECT_DIR%
echo.

pushd "%PROJECT_DIR%" || (
    echo ERROR: Could not enter the project directory.
    exit /b 1
)

rem Create the destination with Windows before Cygwin starts.
if not exist "%PROJECT_DIR%build" mkdir "%PROJECT_DIR%build"
if not exist "%PROJECT_DIR%build" (
    echo ERROR: Could not create the build directory.
    popd
    exit /b 1
)

rem Do not use sed -i on NTFS. Bash reads the LF-formatted script directly.
rem Clear inherited compiler-path variables before invoking the build.
"%CYGWIN_BASH%" --noprofile --norc -c "unset GCC_EXEC_PREFIX COMPILER_PATH CPATH CPLUS_INCLUDE_PATH C_INCLUDE_PATH LIBRARY_PATH; /usr/bin/bash ./build_cygwin.sh"
set "RC=%ERRORLEVEL%"

popd

if not "%RC%"=="0" (
    echo.
    echo ERROR: Cygwin compilation failed with exit code %RC%.
    exit /b %RC%
)

echo.
echo Cygwin compilation completed successfully.
echo Executables are in: %PROJECT_DIR%build
exit /b 0
