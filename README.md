# Fluid Corpus Manipulation Library

This library comprises a suite of C++ algorithms used by the Fluid Corpus Manipulation (FluCoMa) project. Alongside these algorithms, are encapsulations into *clients* that, using a bit of glue code, allow us to deploy these clients across a range of creative coding host environments (Max, Pure Data, SuperCollider).

# Dependencies


* [CMake](http://cmake.org) (>=3.18)
* [Eigen](https://gitlab.com/libeigen/eigen) (3.4)
* [HISSTools Library](https://github.com/AlexHarker/HISSTools_Library)
* [Spectra](https://github.com/yixuan/spectra)
* [Memory](https://github.com/foonathan/memory.git)
* [fmt](https://github.com/fmtlib/fmt)
* [nlohmann JSON](https://github.com/nlohmann/json)



# Building
It is unlikely that you will need or want to build this repository directly. Rather, the pattern is to build from one of the repositories for a specific creative coding host, which uses this library as a dependency:

* [Max](https://github.com/flucoma/flucoma-max)
* [PureData](https://github.com/flucoma/flucoma-pd)
* [SuperCollider](https://github.com/flucoma/flucoma-sc)
* [Command Line](https://github.com/flucoma/flucoma-cli)

We also host pre-built binary releases for each supported platform in these repositories (check https://www.flucoma.org/download/ for the latest binary releases).

You may however be interested in using this library in your own C++ program or creative environment. Example C++ programs can be found in the `examples` folder.

To build these, clone the repo, change to its directory, and run CMake:
```
mkdir -p build && cd build
cmake ..
```
By default, CMake will download the dependencies above directly at this point. This is the simplest option, and guarantees that the versions used match the versions we build and test against.

However, to use versions already on your file system, you can set CMake cache variables `EIGEN_PATH` and `HISS_PATH`:

```
cmake -DEIGEN_PATH=~/code/eigen -DHISS_PATH=~/code/hisstools_library ..
```
Because CMake is a system for generating build scripts, rather than a build system in and of itself, the other choice you have is about what sort of build system it produces output for. The default on macOS and Linux is to use `make`, and on Windows, to use the most recent version of Visual Studio installed.

* On macOS, you can instead use Xcode by passing `-GXcode` with the `cmake` command.
* On Windows, Visual Studio can consume CMake projects directly. When used this way, the cache variables are set in a `JSON` file that MSVC uses to configure CMake.

# Portability
The code base uses standard-compliant C++17 and, as such, should be portable to a range of platforms. So far, it has been successfully deployed to macOS (>= Mac OS X 10.7, using clang); Windows (10 and up, using MSVC); and Linux (Ubuntu 16.04 and up, using GCC), for 32-bit and 64-bit intel architectures. Please check that your compiler version supports the full C++14 feature set.

In principle, it should be possible to build for other architectures, but this has not yet been properly explored and certain parts of the build process may make assumptions about Intel-ness.

If you find objects causing 'illegal instruction' segmentation faults, it is likely that our default vector instruction flag isn't supported by your CPU.

### I'm in a hurry...

And *only building for your own machine*? You want to enable the maximal set of CPU features for your machine without worrying? Using Clang / GCC?

Pass `-DFLUID_ARCH=-mnative` when you run CMake. This tells the compiler to figure out what instruction sets your personal CPU supports, and enable them. This implies a possible performance gain in return for a portability loss.

### More Detail
By default, on Intel / AMD, we enable AVX instructions (`-mavx` on clang/GCC; `/ARCH:AVX` on MSVC). These should work on all relatively recent CPUs (from, say, 2012 onwards). On ARM (with the Bela platform in mind), we use the following architecture flags:

```
-march=armv7-a -mtune=cortex-a8 -mfloat-abi=hard -mfpu=neon
```

If your needs are different, then these defaults can be overridden by passing the desired compiler flags to CMake via the `FLUID_ARCH` cache variable.

If your Intel / AMD chip is too old to support AVX, it probably still supports SSE. On macOS and Linux, `sysctl -a | grep cpu.feat` can be used to produce a list of the various features your CPU supports.

## Credits 
#### FluCoMa core development team (in alphabetical order)
Owen Green, Gerard Roma, Pierre Alexandre Tremblay

#### Other contributors (in alphabetical order):
James Bradbury, Francesco Cameli, Alex Harker, Ted Moore

> This project has received funding from the European Research Council (ERC) under the European Union’s Horizon 2020 research and innovation programme (grant agreement No 725899).
