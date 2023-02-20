% FluCoMa documentation master file, created by
%   sphinx-quickstart on Tue Jan 10 12:02:32 2023.
%   You can adapt this file completely to your liking, but it should at least
%   contain the root `toctree` directive.

Navigating FluCoMa's Code
==========================

[Fluid Corpus Manipulation (or FluCoMa)](https://flucoma.org) is a toolkit and set of resources for experimenting with data-driven music making in Max, Pure Data and SuperCollider. This site is to provide documentation on its source code, to help people contribute (or just see how things work under the hood). If you're looking for help with using toolkit itself, then you may find what you need in our [learning resources](https://learn.flucoma.org) or at our [forum](https://discourse.flucoma.org/). 

If you intend to write some code, then you probably want to get yourself set up and understand [the build system](./build-system/overview.md). If you're ready to go, then here a more general notes about [contributing](./contributing.md). 


---

```{toctree}
---
hidden: true 
maxdepth: 2
caption: Getting Started 
---
contributing.md
build-system/overview.md
build-system/updating_docs.md
```
```{toctree}
---
hidden: true
maxdepth: 2
caption: Developing Clients
---
clients/pitch.md
clients/kdtree.md
```

```{toctree}
---
hidden: true
maxdepth: 2
caption: Reference
---
glossary.md
```

<!-- * {ref}`genindex`
* {ref}`modindex`
* {ref}`search` -->
