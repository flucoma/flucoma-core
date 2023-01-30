Build System Overview
=====================

FluCoMa uses {term}`CMake` to handle the build process. If you just want to build the existing FluCoMa objects, then you don't really need to understand the gory details: build instructions are available in the `README.md` for each {term}`Host`: 
* [Max](https://github.com/flucoma/flucoma-max)
* [Pure Data](https://github.com/flucoma/flucoma-pd)
* [SuperCollider](https://github.com/flucoma/flucoma-sc)
* [Command Line Interface](https://github.com/flucoma/flucoma-cli) 

Likewise, if you just want to use {term}`Algorithm` implementations on their own, and don't need `FluCoMa` to produce Max, SC, PD plugins (or a CLI executable), then all you need to do is use the header files directly, providing you have a way of getting the needed dependencies and are equipped with a C++ toolchain (see [below](#environment-overview)). 

If you want to develop a new {term}`Client` then you'll need to know a bit more. Here we'll cover the steps involved in adding a new `Client`, and shed some light on how our repos hang together, how dependenices are managed, and so forth. 

# Environment Overview

FluCoMa is developed using C++17, and objects in the official distribution need to run on Mac OS, Windows and desktop Linux-flavours (we typically test on Ubuntu). We don't use compiler-specific extenstions, in the hope that the code can be as portable as possible. 

To use / build FluCoMa code, then you will need 

* a C++17 toolchain for your operating system
* CMake (>= 3.18)
* git 
* if want to build doccumentation, then you will need a python environment >= 3.8 as well ([requirements are here](https://github.com/flucoma/flucoma-docs/blob/main/requirements.txt))

```{warning}
Even though we use C++17 as a language dialect, in order to maintain backwards support for older operating systems, we only use a subset of the standard library. Most notably, we do not use `std::filesystem`. 

Currently, we aim to be able to support as far back as MacOS 10.9 (because we want to be able to support Max  7 for as long as possible). Obviously this committment can not last forever: each change in archiecture or OS APIs makes this harder. 
```

## Repositories 

Each {term}`Host` that FluCoMa supports has a git repository, linked to in the list at the top of this pages. These repos contain the wrapper code that generates viable plugins for that platform, as well as the specific CMake code for that host. Each of these repos depends on [`flucoma-core`](https://github.com/flucoma/flucoma-core) and, if you want to build the documentation as well, on [`flucoma-docs`](https://github.com/flucoma/flucoma-docs). 

To build for a given host, you minimally need to `git clone` the repo for that host. By default it will automatically pull in `flucoma-core` and optionaly `flucoma-docs` (if the CMake cache variable `DOCS=ON`) to `<your build folder>/_deps`, as well as all the other dependencies.  

However, if you're developing, then you probably want to have your own clones of `core` and `docs` to work on. To to this, `git clone` these repos separately, and then tell CMake for the host in question where to find these repos. 

```{note}
The current way we manage our branches is to use `main` as a stable branch with the current release, and to do all new development on a branch called `dev`. So you will want to checkout `dev` for each flucoma repo you are working on. 
```

Step by step: 

1. Clone `flucoma-core`
    ```console
    git clone https://github.com/flucoma/flucoma-core.git 
    ```
2. Clone `flucoma-docs` 
    ```console
    git clone https://github.com/flucoma/flucoma-docs.git 
    ```
3. Clone your host repo 
    ```console
    git clone https://github.com/flucoma/flucoma-max.git 
    ```
    and / or 
    ```console
    git clone https://github.com/flucoma/flucoma-pd.git 
    ```
    and / or 
    ```
    git clone https://github.com/flucoma/flucoma-sc.git 
    ```
   and / or 
   ```console
    git clone https://github.com/flucoma/flucoma-cli.git 
    ```
4. For each repo you've cloned, checkout the `dev` branch:
   ```
   cd <repo directory [core | docs | host]>
   git checkout dev
   ```
5. Go to your host repo, and configure with CMake 
   ```
   cd <host repo directory> 
   mkdir build && cd build
   ```
   When we configure, we will let `CMake`'s `FetchContent` know that we already have the `core` and `docs` repos. 
   ```console
   cmake -DFETCHCONTENT_SOURCE_DIR_FLUCOMA-CORE=<path to where you cloned flucoma-core> -DFETCHCONTENT_SOURCE_DIR_FLUCOMA-DOCS=<<path to where you cloned flucoma-docs> -DDOC=ON ..
   ```
   That's a bit of a mouthful. You may wish to consider either using [CMake presets](https://cmake.org/cmake/help/latest/manual/cmake-presets.7.html) if you're using CMake >= 3.19, or making a .cmake file with these variables in, that can be passed as a set of inital cache values. e.g 
   ```cmake 
    # a file called flucoma-vars.cmake, or whatever you want
    set(FETCHCONTENT_SOURCE_DIR_FLUCOMA-CORE, "<path to where you cloned flucoma-core>") 
    set(FETCHCONTENT_SOURCE_DIR_FLUCOMA-DOCS, "<path to where you cloned flucoma-docs>") 
    set(DOCS, ON)
   ```
   Then the `cmake` incovation simplifies to 
   ```console
   cmake -C<path your .cmake file> ..
   ``` 
6. Try building an object to check that stuff is working. From your `build` folder. For Max / PD: 
   ```console
   cmake --build . --target fluid.gain_tilde
   ```
   For SC
   ```console
   cmake --build . --target FluidGain
   ```

```{note}
The structure of having host repos depend on `core` and `docs` makes life easier for users who just want to build for their preffered host, but it's a bit onerous when developing. 

One could automate a good deal of this with a further `CMake` project that pulls everything in, sets branches etc.  
```

## Dependencies 
If you're using CMake to build, then the dependencies should be pulled in automatically (using CMake's `FetchContent` feature). If you're wanting to use {term}`Algorithm`s stand-alone, then you will need to satisfy these dependencies by hand. You can find out what they are by looking in the [`CMakeLists.txt` file at the top level of the flucoma-core repo](https://github.com/flucoma/flucoma-core/blob/main/CMakeLists.txt), and looking for `FetchContent_Declare` blocks. For example, here is the block for the `Eigen` C++ library that a lot of the algorithms depend on: 
```cmake
FetchContent_Declare(
  Eigen
  GIT_SHALLOW TRUE
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_PROGRESS TRUE
  GIT_BRANCH "3.4"
  GIT_TAG "3.4.0"
)
```

# Adding a New `Client`: QuickStart 

Assuming you've got yourself [set up ok](#environment-overview). Adding a new `Client` starts with making a new `.hpp` C++ header file in a subdirectory of `flucoma-core/inlude/clients`. If your `Client` deals with streaming data (i.e. audio or streams of lists / control signals) then, by convention, it goes in `flucoma-core/inlude/clients/rt`. Otherwise, for offline processors, it goes in `flucoma-core/inlude/clients/nrt`. 

For the purposes of demonstration, let's copy the real-time `GainClient.hpp` and add a new FluCoMa object called `Gain2`. In Max / PD, this will produce an object called `fluid.gain2~`. In SuperCollider, it will make a plugin called `FluidGain2`. 

1. Copy `flucoma-core/inlude/clients/rt/GainClient.hpp`
    ```
    cd include/clients/rt  
    cp GainClient.hpp GainClient2.hpp 
    ```
2. Then, to add the new object to the build system, edit `flucoma-core/FlucomaClients.cmake`, by adding the following line
    ```cmake
    add_client(Gain2 clients/rt/GainClient2.hpp CLASS RTGainClient)
    ```
    This allows the `CMake` to take care of some otherwise tedious work of generating an appropriate `.cpp` file and compilation target for your host 
3. Then go back to the `build` directory for your host, and see if it builds
    ```console
    cmake --build . --target Gain2
    ```

All being well, the build will succeed and your object will appear somewhere, varying depending on the host
* Max: `flucoma-max/externals`
* PD: `flucoma-pd/pd_objects` 
* SC: `flucoma-sc/release-packaging/Plugins` 
* CLI: `flucoma-cli/bin`

```{note}
This diversity of output locations is a mess, we know. Even worse, it's writing into the source tree, which is generally bad practice. 

The tricky thing has been wanting to arranging matters so that newly compiled plugins appear in a structure that is viable for the host in question, i.e. 
* for Max, in a package-like folder 
* for PD, a flat directory that contains both externals and help files
* for SC, a folder hierarchy that follows convention 

Running the `cmake` `install` target makes all this happen anyway, but then decouples one's freshly minted plugin from it's generated location, which then makes debugging harder. 

Anyway, it needs sorting out. For now, be aware that 
* For Max, `flucoma-max` can (for now) be placed directly in packages (or sym-linked) and will work 
* For SC, the `flucoma-sc/release-packaging` can be sym-linked into your extensions folder, and will work 
* For PD, the need to have externals and help files together makes everything awful
```

So, nearly done. The final step is to see if your new object works. In Max / PD, it should be immediately available, providing the environment can see it (see the above note). We should be able to test it by opening the existing `fluid.gain~` / `FluidGain` help file and changing the object name.

## SuperCollider Language Class 
In SC, you will want an `sclang` class file for the object. Let's copy and amend `flucoma-sc/release-packaging/Classes/FluidGain.sc`: 

1. Copy
   ```
   cd flucoma-sc/release-packaging/Classes/
   cp FluidGain.sc FluidGain2.sc
   ```
2. Edit. Open FluidGain2.sc in your favourite editor (which might be the SC IDE for these purposes), and just change the class name
   ```js
   FluidGain2 : UGen {
	   *ar { arg in = 0, gain=1.0;
		   ^this.multiNew('audio', in.asAudioRateInput(this), gain)
	  }
   }
   ```
3. Recompile the class library, amend the `FluidGain` helpfile, start the server and see if it works!

## Next Steps: Your Own Object 

Obviously this object isn't very exciting. For more detail on the manouveres involved in making your own `Client`, you can look at the walk-throughs of some existing `Client`s

* [`PitchClient`](../developing clients/pitch.md) shows a client that takes audio in, does an STFT and produces control-rate descriptors
* [`KDTree`](../developing clients/kdtree.md) shows a 'data' client that uses `FluidDataSet` as a source for running an algorithm. 

Of course, as part of making your own {term}`Client`, presumably you have an {term}`Algorithm` that it runs. You have much more latitude over the form of this than with `Client`s. By convention, it will be in a header file that lives in `flucoma-core/include/algorithms/public` (if it is intended to be part of the distribution)

## Debugging 

Because {term}`client`s need to be run in the context of a host environment (until we make a mock environment, somehow), debugging involves attaching a debugger to the host environment itself. The exact mechanics of this vary by the host and operating system, but 
* for SuperCollider, it's easiest launch the server from the SC IDE and then attach your debugger to the running instance of `scsynth` 
* likewise, for PD. On Mac OS, note that you need to attach to the `pd` executable that lives inside the `.app` package (because the gui and pd istelf are separate processes)
* for Max, attaching to Max.app should work (or Max.exe on Windows). Note that, on Mac, if you use the XCode generator from cmake, we make the  resulting project set up to use `/Applications/Max.app` as the execution / debug target for all the objects, so that just pressing the 'play' button a client's target should build it and launch Max with the debugger attached. Which is nice. 

We (well, weefuzzy) would heartily reccomend working almost exclusively with Debug builds whilst developing, so that `assert` conditions can be tested and caught. If your object really needs some optimisation to be able to work at all, then we have a cmake build configuration called `Test` that will keep `assertions` on but apply some optimiation. However, checking Release builds for surprises is very important too. 

### Sanitizers

We would also wholeheartedly endorse debugging with santizers like `asan` (address sanitizer) and `ubsan` (undefined behaviour sanitizer) whereever possible, because they can make catching and diagnosing bugs much, much quicker. Unfortunately, using these in the context of a host environment isn't straightforward, because the libraries need to be *injected* into the host's running environment to work (otherwise it just crashes when it tries to load your object). 

The basic formula for injecting, say, `asan` on Mac OS is to execute
```console
"export DYLD_INSERT_LIBRARIES=<location of your asan lib>;" <progam>
```

The location of the asan library depends on your toolchain and thus XCode version on Mac OS. It's probably something like 
```
/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/lib/clang/<VERSION>/lib/darwin/libclang_rt.asan_osx_dynamic.dylib
```
where `<VERSION>` is some version number for the `clang` in your toolchain. This might work to save burrowing through folders
```console
find  /Applications/Xcode.app/Contents/Developer/Toolchains/ -name \*asan_osx\*.dylib
```

In SuperCollider, you can enable this in the IDE like this
```
~realserverprogam = Server.program

Server.program = "export DYLD_INSERT_LIBRARIES=<terrifying string from above>;" + Server.program
```
This sets the `Server.program` to run with the asan library injected. We set the variable `~realserverprogram` so you can switch back if needed.

For Max, it doesn't appear possible to inject asan into Max 8 because of something to do with the bundled libchromium. However, it does work for Max 7, which you will need to invoke from the terminal. 

PD can, likewise, be invoked from the terminal. 