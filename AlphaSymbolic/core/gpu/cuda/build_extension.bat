call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
set DISTUTILS_USE_SDK=1

set BUILD_DIR=%USERPROFILE%\cuda_build_tmp
mkdir "%BUILD_DIR%" 2>nul
copy setup.py "%BUILD_DIR%"
copy rpn_kernels.cu "%BUILD_DIR%"
copy bindings.cpp "%BUILD_DIR%"

pushd "%BUILD_DIR%"
python setup.py build_ext --inplace
popd

copy "%BUILD_DIR%\*.pyd" .
copy "%BUILD_DIR%\*.pyd" ..
