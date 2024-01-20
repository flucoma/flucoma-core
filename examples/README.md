# Compiling examples

## Bulletproof configuration

```sh
# in flucoma-core root directory, create a build directory and CD into it
mkdir build && cd build

# CMake to configure targets
cmake .. -DBUILD_EXAMPLES=ON

# build an example target, currently two options: 'describe' and 'dataset'
cmake --build . --target describe
#    OR
make describe
```
