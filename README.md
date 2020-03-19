# Fluid Corpus Manipulation Library 

This library comprises a suite of C++ algorithms used by the Fluid Corpus Manipulation (FluCoMa) project. Alongside these algorithms, are encapsulations into *clients* that, using a bit of glue code, allow us to deploy these clients across a range of creative coding host environments (Max, Pure Data, SuperCollider). 

# Dependencies 

* [CMake](http://cmake.org) (>=3.11)
* [Eigen](https://gitlab.com/libeigen/eigen) (3.3.5)
* [HISSTools Library](https://github.com/AlexHarker/HISSTools_Library)

# Building 
It is unlikely that you will need or want to build this repository directly. Rather, the pattern is to build from one of the repositories for a specific creative coding host, which uses this library as a dependency (managed via CMake's `FetchContent` mechanism) 
* [Max](Max)
* [PureData]()
* [SuperCollider]()
* [Command Line]()

All the algorithms, clients and glue code are header-only so, in that sense, there is little to build. However, some of this library's dependencies do require compilation. We use CMake to manage this: the details about how to build these dependencies live in the `CMakeLists` for this library, and are passed on to the other projects that depend on this one. In other words, this process should be transparent. 

## Building Examples
The other exception to this is the `examples` folder, which contains command-line programs that interface directly with the algorithms layer of this library; these are mostly used for prototyping and will blossom, in due course, into a set of actual tests, and a separate set of actual examples, for people who may wish to use this library directly as a C++ API.   

To build these, clone the repo, change to its directory, run CMake: 
```
mkdir -p build && cd build 
cmake .. 
```
By default, CMake will download Eigen and the HISSTools library directly at this point. This is the simplest option, and guarantees that the versions used match the versions we build and test against. 

However, to use versions already on your file system, you can set CMake cache variables `EIGEN_PATH` and `HISS_PATH`. You may wish to do this if you are working directly on one of these dependencies, and wish to experiment with pushable changes to these alongside this library. 

Cache variables are set either in the CMake GUI, or using `ccmake`, or by passing on the command line `cmake -D<CACHE VARIABLE>=<VALUE>`. For instance: 
```
cmake -DEIGEN_PATH=~/code/eigen -DHISS_PATH=~/code/hisstools_library ..
```
Because CMake is a system for generating build scripts, rather than a build system in and of itself, the other choice you have is about what sort of build system it produces output for. The default on Mac OS and Linux is to use `make`, and on Windows, to use the most recent version of Visual Studio installed. This project requires "Visual Studio 17" >= 15.9

* On Mac, you can instead use Xcode by passing `-GXcode` with the `cmake` command. 
* On Windows, Visual Studio can consume CMake projects directly. When used this way, the cache variables are set in a `JSON` file that MSVC uses to configure CMake.

# Portability 
The code base uses standard-compliant C++14 and, as such, should be portable to a range of platforms. So far, it has been successfully deployed to Max OS (>= 10.7, using clang); Windows (10 and up, using MSVC); and Linux (Ubuntu 16.04 and up, using GCC), for intel architectures at 32- and 64-bit. Please check that your compiler version supports the full C++14 feature set.

In principle, it should be possible to build for other architectures, but this has not yet been properly explored and certain parts of the build process may make assumptions about Intel-ness. 

## Vector instructions and CPU architecture. 
Most types of CPU in common use support a range of specialised instructions for processing vectors of numbers at once, which can offer substantial speed gains. For instance, on Intel / AMD chips there have been a succession of such instruction sets (MMX, SSE, AVX and so on).

If you find objects causing 'illegal instruction' segmentation faults, it is likely that our default vector instruction flag isn't supported by your CPU. 

### I'm in a hurry...

And *only building for your own machine*? You want to enable the maximal set of CPU features for your machine without worrying? Using Clang / GCC? 

Pass `-DFLUID_ARCH=-mnative` when you run CMake. This tells the compiler to figure out what instruction sets your personal CPU supports, and enable them. This implies a possible performance gain in return for a portability loss. 

### More Detail 
By default, on Intel / AMD, we enable AVX instructions (`-mavx` on clang/GCC; `/ARCH:AVX` on MSVC). These should work on all relatively recent CPUs (from, say, 2012 onwards). On Arm (with the Bela in mind), we use the following architecture flags: 

```
-march=armv7-a -mtune=cortex-a8 -mfloat-abi=hard -mfpu=neon
```

If your needs are different, then these defaults can be overridden by passing the desired compiler flags to CMake via the `FLUID_ARCH` cache variable. Note that this variable expects you to pass arguments in a form suitable for your compiler, and currently performs no checking or validation. 

* Clang and GCC options are generally the same, see https://clang.llvm.org/docs/ClangCommandLineReference.html
* See MSVC options here https://docs.microsoft.com/en-us/cpp/build/reference/arch-minimum-cpu-architecture?view=vs-2019

If your Intel / AMD chip is too old to support AVX, it probably still supports SSE. On Mac OS and Linux, `sysctl -a | grep cpu.feat` can be used to produce a list of the various features your CPU supports. 

> This project has received funding from the European Research Council (ERC) under the European Union’s Horizon 2020 research and innovation programme (grant agreement No 725899).
