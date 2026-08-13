@echo off
rem -----------------------------------------------------------------------------
rem Script de Compilación y Ejecución para el Proyecto de Regresión Simbólica
rem Autor: Juan Manuel Peña Usuga (con ayuda de Gemini)
rem Descripción:
rem Este script automatiza todo el proceso de compilación:
rem 1. Configura el entorno de Visual Studio 2022.
rem 2. Crea un directorio de compilación limpio.
rem 3. Ejecuta CMake para generar los archivos de proyecto de Visual Studio.
rem 4. Compila el proyecto para crear el ejecutable.
rem 5. Ejecuta el programa resultante.
rem -----------------------------------------------------------------------------

echo.
echo =======================================================
echo      INICIANDO SCRIPT DE COMPILACION
echo =======================================================
echo.

rem --- PASO 1: Configurar el Entorno de Visual Studio 2022 ---
rem Busca la ruta de vcvarsall.bat. Esta es una ruta común.
rem Si tu instalación es diferente, ajusta esta línea.
set "VS_ENV_SCRIPT=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat"

if not exist "%VS_ENV_SCRIPT%" (
    echo ERROR: No se pudo encontrar vcvarsall.bat en la ruta esperada.
    echo Por favor, ajusta la variable VS_ENV_SCRIPT en este script.
    rem pause
    exit /b 1
)

rem Llama al script para configurar el entorno para x64 (64 bits)
call "%VS_ENV_SCRIPT%" x64
echo Entorno de Visual Studio 2022 (x64) configurado.
echo.

rem --- PASO 2: Preparar el Directorio de Compilación ---
set "BUILD_DIR=build"
echo Preparando el directorio de compilacion: %BUILD_DIR%...
if exist "%BUILD_DIR%" (
    echo Directorio existente. Limpiando...
    rmdir /S /Q "%BUILD_DIR%"
)
mkdir "%BUILD_DIR%"
cd "%BUILD_DIR%"
echo Directorio de compilacion listo.
echo.

rem --- PASO 3: Ejecutar CMake para Configurar el Proyecto ---
echo Configurando el proyecto con CMake...
rem Usamos la ruta completa a CMake de VS para máxima seguridad.
"C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" -G "Visual Studio 17 2022" ..

rem Comprobar si CMake falló
if %errorlevel% neq 0 (
    echo.
    echo ***********************************************
    echo ** ERROR: La configuracion con CMake fallo. **
    echo ***********************************************
    rem pause
    exit /b 1
)
echo Configuracion completada con exito.
echo.

rem --- PASO 4: Compilar el Proyecto ---
echo Compilando el proyecto...
rem Elige la configuración: "Debug" o "Release"
set "CONFIG=Debug"
"C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" --build . --config %CONFIG%

rem Comprobar si la compilación falló
if %errorlevel% neq 0 (
    echo.
    echo **************************************
    echo ** ERROR: La compilacion fallo. **
    echo **************************************
    rem pause
    exit /b 1
)
echo Compilacion completada con exito.
echo.

rem --- PASO 5: Ejecutar el Programa ---
set "EXECUTABLE_PATH=.\%CONFIG%\SymbolicRegressionGP.exe"
echo Ejecutando el programa: %EXECUTABLE_PATH%
echo -------------------------------------------------------
echo.

if exist "%EXECUTABLE_PATH%" (
    call "%EXECUTABLE_PATH%"
) else (
    echo ERROR: No se encontro el archivo ejecutable.
)

echo.
echo -------------------------------------------------------
echo La ejecucion ha finalizado.
echo.

rem --- Final ---
rem exit /b 0
