# Glossary

```{glossary}
Host
  A third-party creative-coding environment that hosts FluCoMa objects, such as Max, Pure Data or SuperCollider

Host wrapper
  The outermost layer of the framework. Given a specification for an object in a {term}`client`, the host wrapper translates this specification into a plugin for a given {term}`host` 

Client
  A client represents an interface for a composition of {term}`algorithm`s. Unlike algorithms, clients need to expose certain functions to enable the framework to construct concrete objects for a given {term}`host`

Algorithm
  An algorithm is the innermost component of FluCoMa's architecture, and simply denotes some class (or even function) that processes some data. There are very few restrictions on how an algorithm is represented, though there are some conventions. 

[CMake](https://cmake.org/)
  A build system generator. CMake is its own little language for describing how to build a piece (or pieces) of software, that produces a build script in a choice of formats (e.g Makefile, ninja, XCode project, MSVC project)

```