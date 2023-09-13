Creating New Objects
=========================

Below is a list of all necessary changes to create a full package for a new object.

## Object Code
In [Core](https://github.com/flucoma/flucoma-core):
* Create a client file at [include/clients/rt/{ObjectName}Client.hpp](https://github.com/flucoma/flucoma-core/tree/main/include/clients/rt).
* Add a new line in [FlucomaClients.cmake](https://github.com/flucoma/flucoma-core/blob/main/FlucomaClients.cmake) for `{ObjectName}`, referencing the client file from the previous step.
* If the client file becomes too bloated, separate the algorithm out into a new file at [include/algorithms/public/{ObjectName}.hpp](https://github.com/flucoma/flucoma-core/tree/main/include/algorithms/public).

## Documentation
In [Docs](https://github.com/flucoma/flucoma-docs):
* Create a reference file at [doc/{ObjectName}.rst](https://github.com/flucoma/flucoma-docs/tree/main/doc/).

In [Learn](https://github.com/flucoma/learn-website):
* Create a reference page. First, create a new folder at [src/routes/(content)/reference/{ObjectName}/](https://github.com/flucoma/learn-website/tree/main/src/routes/(content)/reference).
* Then, create `+page.svx` and `{ObjectName}.svelte` files in this folder. `+page.svx` will have the basic page content, and more specialised custom elements can be coded in the `.svelte` file.

In [Max](https://github.com/flucoma/flucoma-max):
* Create a help patch for the object at [help/{MaxObjectName}.maxhelp](https://github.com/flucoma/flucoma-max/tree/main/help).
* If you need separate subpatchers for objects such as the `poly~`, place htem in the [patchers/](https://github.com/flucoma/flucoma-max/tree/main/patchers) folder.
* Add the object to the object list in the [Fluid Corpus Manipulation Toolkit](https://github.com/flucoma/flucoma-max/tree/main/extras/Fluid Corpus Manipulation Toolkit.maxpat) patch.

In [SC](https://github.com/flucoma/flucoma-sc):

In [PD](https://github.com/flucoma/flucoma-pd):

In [CLI](https://github.com/flucoma/flucoma-cli):
