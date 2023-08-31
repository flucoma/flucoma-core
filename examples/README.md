# Compiling examples

## Bulletproof configuration

```sh
# in flucoma-core root directory, create a build directory and cd into it
mkdir build && cd build

# CMake to configure targets
cmake .. -DBUILD_EXAXMPLES=ON

# build example target, currently only `describe`
cmake --build . --target describe
#    OR
make describe
```

## Linking local repos

```sh
# in flucoma-core root directory, create a build directory and cd into it
mkdir build && cd build

# CMake to configure targets
cmake .. -DBUILD_EXAXMPLES=ON -C/path/to/cache/file.cmake

# build example target, currently only `describe`
cmake --build . --target describe
#    OR
make describe
```
