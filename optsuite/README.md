# OptSuite
---

## 安装

框架使用 CMake 构建，目前支持 Linux，msys2/MinGW。
Windows 用户也可选择使用 WSL 构建。

必要依赖：
- gcc/g++/gfortran
- Eigen3
- BLAS/LAPACK 实现（或 Intel MKL，当使用 MKL 时，Eigen 会将后端替换为 MKL 中的函数）

可选依赖：
- fftw
- PROPACK（用于计算大规模奇异值分解，核范数相关算子）
- ARPACK（用于计算大规模特征值分解）
- SuiteSparse (用于稀疏矩阵分解)
- Matlab（用于读取 `.mat` 文件）

### Linux 下的安装

#### 安装系统依赖

gcc/g++/gfortran, Eigen3, fftw, blas/lapack, SuiteSparse 和 MKL 均可从源里拉取或从官网上安装（MKL）。
源中的 ARPACK 编译使用的是默认的 blas/lapack，使用的时候可能会出问题。
在这里建议使用 OptSuite 相对应的 blas/lapack 实现重新编译一遍 ARPACK，这样达到的效果最好。
详见编译 ARPACK 这一小节。

对于依赖 PROPACK，我自己维护了一个版本，并将其设置为了 submodule。
编译 OptSuite 的时候也就自带 PROPACK 了，不需要手动安装了。

以下是在 Ubuntu 20.04 上安装依赖的例子：
```
sudo apt install build-essential cmake gfortran git libeigen3-dev \
    libopenblas-dev liblapacke-dev \
    libfftw3-dev libsuitesparse-dev libmetis-dev \
    libbz2-dev zlib1g-dev
```
其中一些可选依赖如果不需要可以不安装，如果打算使用 MKL 则 openblas 和 lapacke
也无需安装。

#### 手动安装 ARPACK
ARPACK 是可选依赖，如果不需要启用可以跳过该步骤。

首先下载源码
```
git clone https://github.com/opencollab/arpack-ng.git
```

然后使用 `cmake` 构建。注意，ARPACK 也依赖 blas/lapack，请记住编译 ARPACK
时你所使用的 blas/lapack 库。
```
cd arpack-ng
cmake -S . -B build -DICB=ON -DCMAKE_INSTALL_PREFIX=/opt/ARPACK
cmake --build build -j4
cmake --install build
```
这里 `-DICB=ON` 是必要的，否则无法安装 ARPACK 的 C/C++ 接口。示例中的 ARPACK
的安装目录是 `/opt/ARPACK`。

#### 获取 OptSuite 代码
首先下载源码
```
git clone --recursive https://github.com/Strong-Wuhj/OptSuite.git
```
optsuite在OptSuite内

#### Configure

使用 `cmake` 命令构建，我们默认将会在 `optsuite` 根目录下新建一个名为 `build` 的目录进行构建。首先进入 optsuite 根目录。

Configure 的命令为
```
cmake -S . -B build [OPTIONS]
```
比较重要的参数有：
- `-B` 构建目录。我们默认使用 `build` 目录。
- `-DCMAKE_BUILD_TYPE=Debug|Release` 是否启用优化，默认是 Debug（不启用）
- `-DCMAKE_PREFIX_PATH` cmake 默认的额外搜索路径。如果将 ARPACK 装在了非系统默认目录，
  一般需要设置这个变量。例如 `-DCMAKE_PREFIX_PATH=/opt/ARPACK`。多个目录请用分号 `;`
  连接。其它库（Eigen，fftw 同理）。
- `-DENABLE_IF_MATLAB` 是否构建 MATLAB 相关接口，默认不构建（OFF）
- `-DMatlab_ROOT_DIR` 在构建 MATLAB 相关接口条件下，指定 MATLAB 安装根目录

和可选依赖相关的参数有：
- `-DUSE_FFTW` 是否启用 fftw 支持
- `-DUSE_ARPACK` 是否启用 ARPACK 支持
- `-DUSE_PROPACK` 是否启用内置 PROPACK
- `-DUSE_SUITESPARSE` 是否启用 SuiteSparse 支持

以上可选参数的默认值均为 `ON`，当 CMake 成功找到对应的库时就会在 OptSuite
中打开相关的支持（对于 PROPACK 则会直接使用 submodule 中的 PROPACK 代码）。
当不需要使用时，可以手动将这些开关设置为 `OFF`，表示不启用这些功能。

下面是一些例子：
```
# 默认安装（不启用优化，不使用 matlab）
cmake -S . -B build

# 指定额外搜索的位置，启用优化，不使用 matlab
cmake -S . -B build -DCMAKE_PREFIX_PATH="/opt/ARPACK;/opt/fftw" -DCMAKE_BUILD_TYPE=Release

# 指定额外搜索位置，启用优化，使用 matlab（目前为止的全部功能）
cmake -S . -B build -DCMAKE_PREFIX_PATH="/opt/ARPACK;/opt/fftw" -DCMAKE_BUILD_TYPE=Release \
      -DENABLE_IF_MATLAB=ON -DMatlab_ROOT_DIR=/opt/MATLAB

# 使用默认安装，但关闭 PROPACK 支持
cmake -S . -B build -DUSE_PROPACK=OFF

```

目前 cmake 会自动查找环境中的 BLAS/LAPACK 或者是 MKL，使用 MKL 前请设置好相关
环境变量，如 `MKLROOT`。BLAS/LAPACK 库请选择 32 位整数的 interface（LP64），这是 Eigen
本身的要求。

执行 Configure 成功后，系统会额外打印各种信息，例如实际找到的 Eigen，ARPACK
等。编译前可检查是否符合预期。以下是一个例子：
```
-- Configuration summary for OptSuite:
   -- PREFIX: /usr/local
   -- BUILD: Release
   -- SHARED_LIBS: ON
   -- PATH: /opt/ARPACK
   -- ENABLE_SINGLE: OFF
   -- MKL: TRUE
      -- compile: /opt/intel/compilers_and_libraries_2020.4.304/linux/mkl/include
   -- CXX:      /usr/bin/c++
   -- CXXFLAGS: -O3 -DNDEBUG
   -- MATLAB: ON
      -- DIR: /opt/MATLAB/R2020b
   -- Eigen:
      -- compile: /usr/include/eigen3
   -- BLAS:
      -- link:    /opt/intel/compilers_and_libraries_2020.4.304/linux/mkl/lib/intel64_lin/libmkl_intel_lp64.so
      -- link:    /opt/intel/compilers_and_libraries_2020.4.304/linux/mkl/lib/intel64_lin/libmkl_intel_thread.so
      -- link:    /opt/intel/compilers_and_libraries_2020.4.304/linux/mkl/lib/intel64_lin/libmkl_core.so
      -- link:    /opt/intel/compilers_and_libraries_2020.4.304/linux/compiler/lib/intel64_lin/libiomp5.so
      -- link:    -lpthread
      -- link:    -lm
      -- link:    -ldl
   -- LAPACKE:
      -- link:    /opt/intel/compilers_and_libraries_2020.4.304/linux/mkl/lib/intel64_lin/libmkl_intel_lp64.so
      -- link:    /opt/intel/compilers_and_libraries_2020.4.304/linux/mkl/lib/intel64_lin/libmkl_intel_thread.so
      -- link:    /opt/intel/compilers_and_libraries_2020.4.304/linux/mkl/lib/intel64_lin/libmkl_core.so
      -- link:    /opt/intel/compilers_and_libraries_2020.4.304/linux/compiler/lib/intel64_lin/libiomp5.so
      -- link:    -lpthread
      -- link:    -lm
      -- link:    -ldl
   -- FFTW:
      -- compile: /usr/include
      -- link:    /usr/lib/x86_64-linux-gnu/libfftw3.so
   -- PROPACK: (bundled)
   -- ARPACK:
      -- compile: /opt/ARPACK/include/arpack
      -- link:    /opt/ARPACK/lib/libarpack.so
   -- SuiteSparse: OFF
```

#### 使用 intel 编译器
编译 MATLAB 支持时，由于 MATLAB 仅支持 gcc 编译器，一些库会依赖 `libstdc++.so`
的内容。非 gcc 编译器因为不会自动链接这个库所以会报错。通常情况下，MATLAB
会自带一个 `libstdc++.so`，一般位置在 `${Matlab_ROOT}/sys/os/glnxa64/libstdc++.so.6`。
需要手动将这个库设置进去。

#### 编译
Configure 成功后，使用 `cmake --build build -j N` 编译，其中 `N` 是编译的线程数。

编译结束后，会在 `build` 目录看到如下文件：
- `libOptSuite.so` 主要代码框架库。
- `libOptSuite_matlab.so` 若选择构建 matlab，则会出现该库。
- `test/xxx` 功能型测试程序。
- `test/<prereq>/xxx` 可选依赖的功能型测试程序。如 `test/arpack/eigs`。
- `test/matlab/xxx` 功能型测试程序（matlab）。
- `example/xxx` 优化实例测试程序。
- `example/matlab/xxx` 优化实例测试程序（matlab）。

也可以到 `example`，`example/matlab`，`test`，`test/matlab` 目录下查看、修改这些代码，并重新编译。

#### 安装
(TODO) CMake 中目前没有写 Install 相关的目标。

### msys2/MinGW 下的安装
OptSuite 也可以在 msys2/MinGW 下编译。为此需要先下载 [msys2](https://www.msys2.org/)。

按照提示安装。安装完毕后点击 `msys64/mingw64.exe` 启动 msys2 中的 MinGW。

![mingw64](_img/mingw64.png)

（上图程序图标较多，注意区分）

初次启动会花费较多时间安装，准备完毕后，首先安装依赖：
```
pacman -S mingw-w64-x86_64-toolchain mingw-w64-x86_64-cmake git \
    mingw-w64-x86_64-openblas mingw-w64-x86_64-eigen3 mingw-w64-x86_64-fftw mingw-w64-x86_64-suitesparse
```
以上命令中的可选依赖根据个人情况进行开启或关闭。

#### 安装 ARPACK
和 Linux 基本相同。获取源码后，按照如下步骤安装（注意命令的变化）：
```
cd arpack-ng
cmake -S . -B build -G "MinGW Makefiles" -DICB=ON -DCMAKE_INSTALL_PREFIX=/opt/ARPACK
cmake --build build -j4
cmake --install build
```
这里 `-DICB=ON` 是必要的，否则无法安装 ARPACK 的 C/C++ 接口。示例中的 ARPACK
的安装目录是 `/opt/ARPACK`。

#### Configure
和 Linux 基本相同。获取源码并进行子模块初始化后，可以调用 `cmake` 进行配置。
下面是一些例子：
```
# 默认安装（不启用优化，不使用 matlab）
cmake -S . -B build -G "MinGW Makefiles"

# 指定额外搜索的位置，启用优化，不使用 matlab
cmake -S . -B build -G "MinGW Makefiles" -DCMAKE_PREFIX_PATH=/opt/ARPACK -DCMAKE_BUILD_TYPE=Release

# 指定额外搜索位置，启用优化，使用 matlab（目前为止的全部功能）
cmake -S . -B build -G "MinGW Makefiles" -DCMAKE_PREFIX_PATH="/opt/ARPACK;/opt/fftw" \
      -DCMAKE_BUILD_TYPE=Release \
      -DENABLE_IF_MATLAB=ON -DMatlab_ROOT_DIR=/e/MATLAB

# 使用默认安装，但关闭 PROPACK 支持
cmake -S . -B build -DUSE_PROPACK=OFF
```

#### 编译
Configure 成功后，使用 `cmake --build build -j N` 编译，其中 `build` 是 Configure 设置的构建目录，`N` 是编译的线程数。

编译结束后，会在 `build` 目录看到如下文件：
- `libOptSuite.dll` 主要代码框架动态库。
- `libOptSuite.dll.a` 主要代码框架静态库。
- `test/xxx` 功能型测试程序。
- `test/<prereq>/xxx` 可选依赖的功能型测试程序。如 `test/arpack/eigs`。
- `test/matlab/xxx` 功能型测试程序（matlab）。
- `example/xxx` 优化实例测试程序。
- `example/matlab/xxx` 优化实例测试程序（matlab）。

由于 Windows 共享库的机制，运行程序时，所有依赖的动态库都必须在 `PATH` 上。
因此需要修改环境变量：
```
# 没开启 MATLAB 的情况
export PATH=/opt/ARPACK/bin:$(pwd)/build/deps/propack/build/double:$(pwd)/build:$PATH

# 启用 MATLAB 支持，同时还需要加上 MATLAB 相关的路径
export PATH=/e/MATLAB/bin/win64:$PATH
```
之后即可运行程序，例如
```
build/example/glasso.exe
```
