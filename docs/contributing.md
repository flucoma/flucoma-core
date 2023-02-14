Contributing
============

We welcome any contributions from the FluCoMa community. Indeed, we need them if the project is to be maintained!

```{note}
Contributions need not be in C++ code, if that's not your jam, although that's what this page is mostly about. Also valuable are 

* filing issues when you find bugs
* improving help files and online learning materials 
* participating on [discourse](https://discourse.flucoma.org/)
```

Meanwhile, if you're still reading then presumably you want to contribute some C++ code, or perhaps just make a pull request for a help file improvement. 

The steps involved in getting yourself set up are covered in [here](build-system/overview.md)

# Making a Pull Request 

You've made some improvement, either by improving a help file, fixing a bug or perhaps even making a new object. Now you'd like us to integrate it. 

```{warning}
Make sure your changes are on a branch from `dev`. 

Our current way of working is that `main` is a stable branch with the latest release, and all new work takes place on `dev`. 
```

To submit a pull request, you'll need a Github fork of the repo you're submitting a change to, and to tell your local `git` to add this as a remote. 

1. Go to the Github page of the repo you're making the PR for, and press the `fork` button on the top right. This will add a copy of the repo to your own GitHub account. Make sure you're on the page for your new fork, and copy the URL. 
2. Adding the fork as a remote to your local copy:
   ```console
   cd <whatever repo>
   git remote add myfork <url for your fork>
   ```
   Note that this is adding the remote with the name `myfork`. Use whatever you prefer. 
3. Push your branch to your fork. Assuming you're on your new branch: 
   ```console
   git push -u myfork  
   ```
4. If this goes ok, part of the response from GitHub's server will conveniently contain a URL for making a pull request from this branch. Launch that URL, and make sure that the PR is targetting the `dev` branch of the repo concerned. 

   Alternatively, you can make a new PR from the Github web interface (or the `gh` command line utility, if that's how you roll)

```{note}
Branch name conventions, which we don't enforce with much vigour are: 

* For fixes: `fix/<something descriptive>` or `fix/issue-<issue number>`
* For new features: `enhance/<something descriptive>`
```

```{note}
Strive to make PRs as minimal as possible, making the fewest changes to the fewest files needed to address the issue at hand. If you have multiple fixes, please make multiple PRs (from multiple branches). This makes reviewing and verifying changes much more tractable. 

PRs that we can't make sense of are likely to be bounced back! 
```

The project maintainers should be automatically notified of your PR, and will try and review it as soon as they can.

# Code Formatting 

If you can, then please run `clang-format` on your code. Each repo has a `.clang-format` file in its root folder, and a properly configured `clang-format` should find and use that. 

In an *ideal* world, there wouldn't be a separate formatting commit (in the interests of a tidy `git` history). If you are able to format before comtting or (more realistically) squash your formatting commit before pushing, so much the better. But we won't (currently) get stroppy if that's all too much. 

# Tests 

We are **woefully** under-covered by tests. Any help in that direction (by adding test for algorithms not yet covered) is massively appreciated. 

Accompanying PRs for new algorithms with a test is very strongly preferred. We'd suggest writing the test first, in fact. 

We can currently test algorithms in `flucoma-core` using `ctest` and [`catch2`](https://github.com/catchorg/Catch2), [`ApprovalTestsCpp`](https://approvaltestscpp.readthedocs.io/en/latest/index.html). Hopefully all the `ctest` details are handled already, but please refer to the `catch2` and `ApprovalTests` documentation as needed. 

```{note}
We haven't yet migrated to Catch2 v3 (the current version), we were / are waiting for the approval tests framework to become compatible. This means that there will be some disjunctures between the current `catch2` docs and our tests. In other words, use our existing tests as a source of documentation too!
```

## Adding a test: 

* Tests live in `flucoma-core/tests`. The structure of this directory mirrors `flucoma-core/include`). 
* Find the appropriate place for your test (i.e. mirroring the location of the main file in `include`) and make a source file `Test<Name of Main File>.cpp` 
* Make `catch2` `TEST_CASE`s as needed. If you're going for it and writing the tests first, this should all fail to start with. That's fine. 
* Adding the test to the build system is lamentably a multistep procedure. 
  
    1. Open `flucoma-core/tests/CMakeLists.txt`
    2. There will be a block of calls to `add_test_executable`, scroll to the end. Add one for your test 
       ```cmake
       add_test_executable(<YourTestName> <path relative to flucoma-core/tests/<YourTestName>.cpp)
       ```
       e.g, if you have a public algorithm `include/algorithms/public/EndTheWorld.hpp`, then the test file could be `flucoma-core/tests/algorithms/public/TestEndTheWorld.cpp` and the cmake entry would be: 
       ```cmake
       add_test_executable(TestEndTheWorld <path relative to flucoma-core/tests/TestEndTheWorld.cpp)
       ```
    3. Optionally, if your test needs to use any of the test signals that some of the other tests use, then link your test to the TestSignals library
       ```cmake
       target_link_libraries(TestEndTheWorld PRIVATE TestSignals)
       ```
    4. Scroll to the end of the block of calls to `catch_discover_tests` and add an entry for your test 
       ```cmake 
       catch_discover_tests(TestEndTheWorld WORKING_DIRECTORY "${CMAKE_BINARY_DIR}")
       ``` 
```{note}
Yes, this sucks. It seemed that this was the only way to get the tests to properly register with `ctest`. Any better expertise appreciated!
```

## Running a Test 

To build and run the tests, cmake *for `flucoma-core`* should be configured with `FLUCOMA_TESTS=ON`:
```console
cd flucoma-core 
mkdir build && cd build
cmake -DFLUCOMA-TESTS=ON ..
```

First build
```console
cmake --build . 
```
Then you can either run all the tests with `ctest`
```
ctest 
```
or just your new test directly by executing `tests/<execuable name>` from your build folder, e.g.
```console
./tests/TestEndTheWorld
```

```{note}
In `vscode`, with the C++ and CMake extensions installed and set up, this can all be done from within the editor by selecting your test as the build and run targets in the configuration bar at the bottom. 

Likewise in full IDEs like XCode or Visual Studio, you can run, execute and debug the targets for tests invidually. 
```

```{note}
Whenever a PR is made to `flucoma-core`, the test suite is built and run against the PR branch for Mac OS, Windows and Ubuntu. Hopefully your code is very standards compliant and Just Works across platforms, but this is very useful to verify...
```

# Dreaming of a Better World 

Improving our test coverage is partly just a matter of getting on writing more tests, like we should have done to begin with. However, not all of this code is immediately obvious in *how to* test it, and for the `Client`s, is just not (yet) possible to test these in isolation from a host environment. 

### Species of Test

Using [Approval Tests](https://approvaltestscpp.readthedocs.io/en/latest/index.html) is our current solution for things that are tricky to test (because it's hard to say *a priori* what a right answer should be). What Approval Tests allow us to do is test for *consistency* by making sure that the output of functions doesn't change since they were last 'approved'.  Getting set up with new Approval Tests isn't too hard (you need some sort of a merge program set up, I use vscode), and if you're struggling to come up with a test for something, we heartily recommend doing an Approval Test as definitely-much-better-than-nothing. 

There are two other categories of test that we've added, and for which detailed documentation will have to wait: death tests and compile tests. These exist to ensure that either runtime or compile-time assertions are behaving as expected. The schemes for doing both are currently held together with duct tape, unfortunately (but work!). The world is hoping that `catch` will add death tests one day (like `GTest` has): these tests work by verifying that a programme aborts when an `assert` call fails. 

Our own compile time tests do something similar by verifying that certain structures that ought not to compile actually don't compile! This is an artefact of doing quite a lot of compile-time stuff in the framework. By and large, you won't find yourself needing to reach for these at all often. 

### Tests and `Client`s 

There isn't currently a pain-free way to test a {term}`Client` in isolation from a specific host. 
There is a SuperCollider specific test harness [FlucomaTestSuite](https://github.com/flucoma/FlucomaTestSuite) that does Approval-style testing for clients, and is generally kept up to date. However, this isn't ideal for developing and debugging. 

So, a great improvement to world would be some way of mocking a host environment so that clients can be tested natively in C++. The main obstacle to this is that currently that the framework code is gnarly and under- / not documented. 
