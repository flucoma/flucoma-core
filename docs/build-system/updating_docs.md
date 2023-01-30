Updating and Building this Documentation
========================================

This set of documentation lives in the [`flucoma-core`](https://github.com/flucoma/flucoma-core) repository (although I'm not about to rule out host-specific stuff ending up in host-specific repos in the future). It uses the [sphinx](https://www.sphinx-doc.org/en/master/) documentation system, so to build it you will need 

* cmake
* python 3  
* sphinx 
* [doxygen](https://www.doxygen.nl/) (to get C++ code docs)
* [breathe](https://breathe.readthedocs.io/en/latest/index.html) (to integrate doxygen and sphinx)

Doxygen can be intalled via homebrew on macOS. Sphinx can also be installed via homebrew, or via pip. 

We use other some add-ons for Sphinx: 

* [myst-parser](https://myst-parser.readthedocs.io/en/latest/): provides an extended markdown syntax that aims to provide the power of reStructuredTxt (Sphinx's default markup language) with the simplicity of markdown 
* [sphinx-book-theme](https://sphinx-book-theme.readthedocs.io/en/stable/index.html): for the theming 

The repo has a `requirements.txt` listing all these python dependencies, so that you can get going with 
```
python -m pip install -r requirements.txt
```
As ever, it is a very good idea to set up a virtual environment with either `venv` or `conda` beforehand, to sandbox these dependencies and avoid surprising interactions with whatever else you may be doing with python. 

The workflow for running doxygen and configuring sphinx follows the general scheme [described by Sy Brand](https://devblogs.microsoft.com/cppblog/clear-functional-c-documentation-with-sphinx-breathe-doxygen-cmake/). To avoid writing stuff into the source tree, we use cmake to process the `Doxygen.in` file at configure time, and set things up to output to our build directory, whatever that might be. Likewise, cmake dynamically generates the command to invoke sphinx, given configure-time knowledge of where to find the doxygen output. 

This command is available in a camke target `docs`, so that running (from your build folder)

```
cmake --build . --target docs
```

will first run doxygen and then sphinx. The output will appear in `<your build folder>/docs/sphinx`. 