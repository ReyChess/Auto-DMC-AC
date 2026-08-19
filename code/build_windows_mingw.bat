@echo off
if not exist build mkdir build
cd build
cmake -G "MinGW Makefiles" ..
if errorlevel 1 exit /b 1
cmake --build . -j 4
