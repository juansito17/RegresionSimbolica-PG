
@echo off
rem -- Use the path confirmed by the user references --
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

set DISTUTILS_USE_SDK=1
set MSSdk=1

rem -- Consistent build dir --
set BUILD_DIR=%USERPROFILE%\cuda_build_native_v1
if exist "%BUILD_DIR%" (
   rmdir /S /Q "%BUILD_DIR%"
)
mkdir "%BUILD_DIR%" 2>nul

echo Copying sources...
copy setup.py "%BUILD_DIR%"
copy *.cu "%BUILD_DIR%"
copy *.cpp "%BUILD_DIR%"

pushd "%BUILD_DIR%"
echo Building in %CD% ...
python setup.py build_ext --inplace
popd

echo Copying artifact back...
copy "%BUILD_DIR%\*.pyd" .

echo Done.
