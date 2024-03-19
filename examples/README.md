# Compiling examples

## Bulletproof configuration

```sh
# in flucoma-core root directory, create a build directory and CD into it
mkdir build && cd build

# CMake to configure targets
cmake .. -DBUILD_EXAMPLES=ON

# build an example target, currently three options: 
# 'describe' takes a single file in and prints stats on various descriptors
# 'dataset' takes a list of soundfiles and make an entry per file in a dataset saved as json
# 'umap' takes a multidimensional dataset input as json, and will use UMAP to reduce the dimension count to 2

cmake --build . --target describe
#    OR
make describe
```
