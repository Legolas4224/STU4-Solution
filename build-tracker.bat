@echo off

cd /D "%~dp0"

if not exist "cpp-tracker" (
    git clone --recursive --branch barebones https://github.com/joosm4n/filtration-station.git cpp-tracker
    if errorlevel 1 exit /b 1
)

cd cpp-tracker
cmake -S . -B build -A x64 -DPYBIND11_PYTHON_VERSION=3.12
if errorlevel 1 exit /b 1

cmake --build build -j4 --config Release 
if errorlevel 1 exit /b 1

copy "build\Release\rock_tracker*.pyd" "..\"
