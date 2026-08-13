@echo off
echo Building TestOperators...
cmake --build build --target TestOperators
if %ERRORLEVEL% NEQ 0 (
    echo Build failed!
    exit /b %ERRORLEVEL%
)

echo.
echo Running Tests...
.\build\Debug\TestOperators.exe
if %ERRORLEVEL% NEQ 0 (
    echo Tests FAILED!
    exit /b %ERRORLEVEL%
)

echo.
echo SUCCESS! All tests passed.
