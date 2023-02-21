A Detailed Walk Through KDTreeClient
====================================

[`PitchClient`]:  pitch.md "PitchClient walk-through"

`KDTreeClient` is an example of a `Client` that handles data from a `FluidDataSet` and that responds to messages (like `fit` and `transform`) rather than processing streams of data. This means that it has some important differences from stream-processor [`PitchClient`][] but also quite a few similarities. 

# KDTreeClient.hpp Overview

The basic mechanics of declaring parameters is the same, but now we add a new entity into the mix: messages. Furthermore, the header files for many of this type of `Client` contain *two* client classes, one of which is there to support real-time querying in Supercollider. 

We end up with the following blocks 

* A namespace within `fluid::client`, in this case `fluid::client::kdtree` 
* An `enum` for indexing the `Client`'s parameters
* A `constexpr` variable that describes the parameters
* The `Client`'s class, in this case `KDTree`, that inherits from `FluidBaseClient`; tag classes that describe its input and output types; and a class template, `DataClient`. 
* A type alias that wraps the `Client` in the `SharedClientRef` template 

So far, this the same shape that we see in [`PitchClient`][]. However, then we see 

* Another `constexpr` variable describing some different parameters in `KDTreeQueryParams`
* A class for the query client, `KDTreeQuery`
* Aliases that register `KDTreeClient` as an offline object, and `KDTreeQuery` as a real-time processor. 

Let's walk through...

# KDTreeClient 

## The Namespace

```c++
namespace fluid {
namespace client {
namespace kdtree {
```

If you've read the [`PitchClient`][] guide, then there is nothing new here. We do our work in a sub-namespace of `fluid::client` named for the `Client` we're defining, `kdtree`. 

## Parameters Declaration

```c++
constexpr auto KDTreeParams = defineParameters(
    StringParam<Fixed<true>>("name", "Name"),
    LongParam("numNeighbours", "Number of Nearest Neighbours", 1),
    FloatParam("radius", "Maximum distance", 0, Min(0)));
```

The parameters are defined into a `constexpr` variable, `KDTreeParams`. Here we define three parameters – a string, an integer and a float. Unlike [`PitchClient`][], none of these parameters refer to each other. We do meet two things though: `StringParam` and this construct `Fixed<true>`. 

`StringParam()` is hopefully obvious from its context: we're declaring a parameter that's a string. The arguments are the parameter's name and it's description (for documentation). The form of the other parameter declarations should be familiar from [`PitchClient`][]. 

The template parameter for `StringParam<Fixed<true>>` says that this parameter has the special property that it is set once at object instantiation and cannot be changed thereafter. In Max and PD, this means that it is available as an object argument, so one can patch `[fluid.kdtree~ foo]` to get an object named 'foo'. 

```{note}
For `Client`s like KDTree, that make objects which are shareable by their name (so, in Max / PD, multiple boxes can address the same underlying object), this `name` parameter is obligatory and *must come first*. 
```

```{admonition} Code Smell
This stipulation is an obvious smell: 

* if the name is a compulsory feature of 'shared' `Client` types, then it is a property of the client and shouldn't need to be declared as a parameter: the framework should add it
* being a string isn't great, because it implies dynamic memory allocation, and doesn't make sense for SuperCollider, where the name is essentially of no interest or use. Deciding on the type of the id for a shared `Client`, and how this manifests in the interface should be delegated to the host wrapper.  
* `<fixed<true>>` is needlessly verbose, and ugly to boot
```

## The Class 

The layout of the `KDTreeClient` class has some features in common with [`PitchClient`][]'s but also some key differences. There is a member function called `process()`, although it has a different signature, and the [boilerplate](./pitch.md#boilerplate) follows the same form. 

The class declaration looks like this 
```c++
class KDTreeClient : public FluidBaseClient,
                     OfflineIn,
                     OfflineOut,
                     ModelObject,
                     public DataClient<algorithm::KDTree>
```

Like [`PitchClient`][], `KDTreeClient` inherits from `FluidBaseClient`, and from a set of tag types: `OfflineIn, OfflineOut, ModelObject`. These are used to tell the host about the sort of I/O the `Client` exhibits (`OfflineIn/Out`), and that `KDTreeClient` is one of these special objects that's referable to by name. 

Meanwhile, `DataClient<algorithm::KDTree>` is a utility class that exposes a bunch of member functions common to these 'model' objects. It warrants a brief aside...

### The `DataClient<T>` Class Template

```{eval-rst}
.. doxygenclass:: fluid::client::DataClient
   :members:
   :undoc-members:
```

`DataClient<T>` defines a common interface for `ModelObjects`, equipping them with the following facilities: 

*  Querying the size and dimensionality of the underlying `Algorithm` with `size()` and `dims()`
* Resetting the object with `clear()` 
* (de-)Serializing to JSON files with `read()` and `write()` or in-memory JSON with `dump()` and `load()`
* Checking that the `Algorithm` is initialized with `initialized()` 
* Getting a reference to the underlying `Algorithm` via `algorithm()`

The template parameter `T` is expected to be the underlying `Algorithm` that a `ModelObject` is representing. So, in the case of `KDTreeClient`, this is `fluid::algorithm::KDTree`. You'll notice that none the methods in `DataClient<T>` are `virtual`, so declaring your own in the client will just hide those in `DataClient` (leaving you free to do your own if really needed).  

```{admonition} Code Smell 
* Having to inherit from two different classes to get the expected functionality isn't great. We should probably be able to (a) have `DataClient` inherit from `FluidBaseClient` and (b) find a way to get rid of `ModelObject`
* `read/write` and `dump/load` are non-DRY: there should be a single pair of functions that take IO streams, and the framework can manage the file / memory distinction elsewhere. (The whole JSON thing needs some TLC)
* why is there a *public* alias to `std::string`? 
```

### Back to `KDTreeClient`: The `enum`

```c++
enum { kName, kNumNeighbors, kRadius };
```
Unlike [`PitchClient`][], the `enum` of parameter indices for `KDTreeClient` is inside the class. We can do that here because none of the parameter declarations need to refer to each other, and it avoids name collisions between `KDTreeClient`s parameters and those of `KDTreeQuery` below. Otherwise, the form and function are the same. 

### The Boilerplate 
After some internal type aliases (which oughtn't be `public`) we have The Boilerplate that infects all `Client`s
```c++
// alias to the type of the parameter descriptors
using ParamDescType = decltype(KDTreeParams);
// alias to the type of the parameter values
using ParamSetViewType = ParameterSetView<ParamDescType>;
// instance of parameter values reference  
std::reference_wrapper<ParamSetViewType> mParams;
// setter, used by the framework
void setParams(ParamSetViewType& p) { mParams = p; }
// get individual values, used locally 
template <size_t N>
auto& get() const
{
  return mParams.get().template get<N>();
}
// return instance to descriptors, used by framework
static constexpr auto& getParameterDescriptors() { return KDTreeParams; }

```

See [the boilerplate section for PitchClient](./pitch.md#boilerplate) for a full discussion of and apology for this monstrosity.


### Constructor 
```c++
KDTreeClient(ParamSetViewType& p, FluidContext&) : mParams(p) 
{
  audioChannelsIn(1);
  controlChannelsOut({1, 1});
}
```

Compare with [`PitchClient`'s constructor](./pitch.md#constructor): things are very similar. The arguments are a set of parameter values, and a `FluidContext` objects. We then initialize the local parameter values, and in the body initialize th channel counts. 

```{admonition} Code Smell 
* Pretty sure that `audioChannelsIn(1)` is an unneeded vestige of some previous design. Makes no sense here 
* I can sort of see *why* `controlChannelsOut({1, 1})`, but only sort of. Seems like we should be able to dispense with it.  
```

### `process()`, Where Nothing Happens

The next thing we see is 
```c++
template <typename T>
Result process(FluidContext&)
{
  return {};
}
```

This is unfortunately necessary, and a legacy of duct-taping the design for these 'model objects' onto the existing data processing model of the earlier `Client`s. It does nothing, except return an instance of this `Result` object, which essentially says 'ok!' and moves on. More boilerplate, basically. 

```{admonition} Code Smell 
Getting rid of this should be easy, although it's worth considering whether we want / need to be able to have `Client`s that have a default processing method like this (that actually does something), along with arbitrary messages. 
```

### Some Member Functions, Where Stuff Happens

Finally, we're at the meat of `KDTreeClient`, represented through these member functions: 
```c++
MessageResult<void> fit(InputDataSetClientRef); 
MessageResult<StringVector> kNearest(InputBufferPtr, Optional<index>) const
MessageResult<RealVector> kNearestDist(InputBufferPtr, Optional<index>) const
```

These implement the interface for `KDTreeClient`, embodying messages `fit`, `kNearest` and `kNearestDist` that can be triggered in the host by users. Respectively, these build a new k-d tree from a `FluidDataSet`, and perform KNN queries using the tree. 

```{note} 
As with `DataClient`, you'll notice that these member functions all return specializations of a class template `MessageResult<T>`. This type functions analogously to the `std::expected<T,E>` class template that comes in C++23: the `MessageResult<T>` holds either a value of type `T`, or an indication of an error. 
```

```{admonition} Code Smell
* Again, having duct taped these facilities on, there is redundancy between the type `Result` and `MessageResult<T>`. Strictly, the former could be replaced everywhere with `MessageResult<void>`
* `MessageResult` is verbose: we should maybe go with something nearer to [`std::expected`](https://en.cppreference.com/w/cpp/utility/expected) now that this is the term of art in C++ circles
```

In this case, the return types of the member functions are `void` for fit (either it worked, or it didn't) and a list of string for `KNearest` or a list of floats for `kNearestDist`. 

Let's go through the functions in detail. 

`fit()` takes as an argument read-only handle to a `FluidDataSet`. After making sure that the handle is valid and useable, it then simply calls through to its `Algorithm`'s constructor (which makes a new `KDTree`). 
```c++
MessageResult<void> fit(InputDataSetClientRef datasetClient)
{
  // Keep a reference to our source dataset
  mDataSetClient = datasetClient;
  // Obtain a pointer (a std::weak_ptr) to the actual data
  auto datasetClientPtr = mDataSetClient.get().lock();
  // If we didn't get one, the dataset no longer exists
  if (!datasetClientPtr) return Error(NoDataSet);
  //Otherwise, finally retrieve the object
  auto dataset = datasetClientPtr->getDataSet();
  // and make sure it's actually got some data in it 
  if (dataset.size() == 0) return Error(EmptyDataSet);
  // All just checking until now. 
  // Actually do some work:
  mAlgorithm = algorithm::KDTree(dataset);
  return OK();
}
```

What's all this `get().lock()` business? That is us first obtaining a `std::weak_ptr` to our `DataSet` with `get()`, and then promoting that to a `std::shared_ptr` with [`lock()`](https://en.cppreference.com/w/cpp/memory/weak_ptr/lock). This gives us a safe way to abort if the DataSet has been deleted since the call was made, and then allows to temporarily prolong the lifetime of the DataSet if it is deleted elsewhere whilst we're working. Obviously such disasters could only occur in a multi-threaded context, but Max is such a thing...

```{admonition} Code Smell 
* As we see here, and in the functions below, there's a cumbersome amount of checking boilerplate involved in all these functions, and the framework should be attempting to take as much of this on as possible. 
* Meanwhile, the representation of references to `DataSets` (and other model object types) leaks far too many implementation details (needing `get()` and `lock()` and all the rest). 
* The implementation of sharing itself is a bit dicey, relying as it does on `shared_ptr`, which has some unfortunate implications as thread safety goes. (it appears safer than it is: the fact of having multiple mutable references, possibly in different threads is badness)  
```

Now, let's look at `kNearest`. This receives a read-only handle to a buffer of data from the host, and an `Optional<index>` for a number of neighbours specified by the user a query-time. Using the `Optional` template (see `std::optional`) is how we express, well, that some message arguments aren't required. `Optional` arguments *have to* come after mandatory ones though.  

```c++
MessageResult<StringVector> kNearest(InputBufferPtr data, Optional<index> nNeighbours) const
{
  // If we have an nNeighbours argument, use it 
  // Otherwise fall back to the object's parameter
  index k = nNeighbours ? nNeighbours.value() : get<kNumNeighbors>();
  // If we're requesting more neighbours than there are points in the tree, return an error
  if (k > mAlgorithm.size()) return Error<StringVector>(SmallDataSet);
  // if there is no tree fitted, return an error:
  if (!mAlgorithm.initialized()) return Error<StringVector>(NoDataFitted);

  // if the input data buffer with the query point is invalid
  // return an error
  InBufferCheck bufCheck(mAlgorithm.dims());
  if (!bufCheck.checkInputs(data.get()))
    return Error<StringVector>(bufCheck.error());
  //finally, do some work: read the query data
  RealVector point(mAlgorithm.dims());
  point <<=
      BufferAdaptor::ReadAccess(data.get()).samps(0, mAlgorithm.dims(), 0);
  //query the tree with that point    
  auto [dists, ids] =  mAlgorithm.kNearest(point, k, get<kRadius>());
  // copy the resulting list of IDs, see below for why we do it like this 
  StringVector result(asSigned(ids.size()));
  std::transform(ids.cbegin(), ids.cend(), result.begin(),
                  [](const std::string* x) {
                    return rt::string{*x, FluidDefaultAllocator()};
                  });
  return result;
}
```

Besides all the checking, there are two things to draw attention to here. First is the use of `FluidTensorView`'s `<<=` operator. What we do here is make a `FluidTensor<double,1>` using the shorthand `RealVector`. We do this because `KDTree::kNearest` expects a vector of `double` and in Max, PD, and Supercollider, the buffer objects contain `float`.

Then, we need to retrieve the float data from the host's buffer object. This happens with
```c++
BufferAdaptor::ReadAccess(data.get()).samps(0, mAlgorithm.dims(), 0);
```
which could be broken down as 
```c++
//obtain a read-only buffer reference from the handle `data` 
auto buffer = BufferAdaptor::ReadAccess(data.get()); 
// read some float data from the buffer reference
FluidTensorView<float, 1> query_data = buffer.samps(0, mAlgorithm.dims(), 0);
```

This two-step of converting the handle to a reference before we can attempt to read (or write) to the buffer is needed to deal with buffer locking in those hosts that use it: it's a familiar C++ [RAII](https://en.cppreference.com/w/cpp/language/raii) technique that guarantees that the lock will only be held for the lifetime of the reference (the `buffer` object above). We then call the `samps()` member function on `buffer`, which returns a `FluidTensorView` pointing to some floats.  

```{admonition} Code Smell 
* If `algorithm::KDTree::kNearest` were a function template, then it could take care of this casting operation internally, rather than delegating to users 
* Why `operator<<=`? Well, before we were using `operator=` but this is counter-intuitive for pointer-like types, which generally don't do a deep-copy with `operator=`. Moreover it meant we couldn't cheaply assign `FluidTensorViews` in a pointer-like way. 
   
   Maybe using the indirection operator `*` would be more idiomatic, as in `*viewA = *viewB` 
```

Finally, for `kNearest` is this mysterious conversion of the returned ids. From `algorithm::KDTree` we get 
```c++
auto [dists, ids] =  mAlgorithm.kNearest(point, k, get<kRadius>());
```
where we're using C++17 structured bindings to unpack the `std::tuple` of distances and ids that `kNearest` coughs up. Then we do this mysterious mess with the `id`s: 
```c++
StringVector result(asSigned(ids.size()));
std::transform(ids.cbegin(), ids.cend(), result.begin(),
                [](const std::string* x) {
                  return rt::string{*x, FluidDefaultAllocator()};
                });
```

Why, for love of all that is unholy, do we need to do this? Two reasons: 
1. `algorithm::KDTree` is returning the ids as `std::string*` (i.e. pointers to `std::string`) rather than doing a new heap-allocation for every id returned. 
2. The string type used downstream in the framework *isn't `std::string`* because we have to use a custom allocator (for Supercollider); note  `return rt::string{...`. Unfortunately, `std::string` and `fluid::rt::string` can't be directly assigned, so we have to use `std::transform` to run over the list of pointers, which we de-reference into new instances of `rt::string`. 

```{note}
The type alias `fluid::rt::string` is a specialisation of `std::basic_string` that uses a non-default allocator, which in turn means that we can use SuperCollider's real-time allocator where we need to (although having strings anywhere near the SC audio thread is smelly). 
```

```{admonition} Code Smell 
Despite both `std::string` and `fluid::rt::string` specialising `std::basic_string` to use `char` as the underlying type, there is no way to make one implicitly convertible to the other. 

Basically, we're currently left with a mismatch between those parts of the `Algorithm`s that are based on `std::string` and the rest of the framework that (has to) use the `rt::string` so that we have control of allocation policy. We stopped short of converting the `Algorithm`s to use `rt::string` instead because 

* all the JSON code will also need updating, and it's not immediately obvious that we can get our JSON framework to play nicely with the allocation stuff (although it can take things that aren't `std::string`)
* `rt::string` might be a pain-point for people using the `Algorithm`s in a stand-alone fashion. So, probably the underlying string type needs to be a customisation point, *and* any assumptions of stringiness in the `Algorithm`s should be minimized, so that non-string `id` types are a possibility. 

```

The `kNearestDist()` function is eerily similar: 

```c++
MessageResult<RealVector> kNearestDist(InputBufferPtr data, Optional<index> nNeighbours) const
{
  index k = nNeighbours ? nNeighbours.value() : get<kNumNeighbors>();
  if (k > mAlgorithm.size()) return Error<RealVector>(SmallDataSet);
  if (!mAlgorithm.initialized()) return Error<RealVector>(NoDataFitted);
  InBufferCheck bufCheck(mAlgorithm.dims());
  if (!bufCheck.checkInputs(data.get()))
    return Error<RealVector>(bufCheck.error());
  RealVector point(mAlgorithm.dims());
  point <<=
      BufferAdaptor::ReadAccess(data.get()).samps(0, mAlgorithm.dims(), 0);
  auto [dist, ids] = mAlgorithm.kNearest(point, k, get<kRadius>());
  return {dist};
}
```

```{admonition} Code Smell
In fact the *only* difference is that we don't use the `id`s, and just return the distance vector instead. This is so non-[DRY](http://wiki.c2.com/?DontRepeatYourself), given all the boilerplate, as to be positively sodden. 
```

### The Message Descriptors 

After `fit`, `kNearest` and `kNearestDist`, we see this static member function:
```c++
static auto getMessageDescriptors()
{
  return defineMessages(
      makeMessage("fit", &KDTreeClient::fit),
      makeMessage("kNearest", &KDTreeClient::kNearest),
      makeMessage("kNearestDist", &KDTreeClient::kNearestDist),
      makeMessage("cols", &KDTreeClient::dims),
      makeMessage("clear", &KDTreeClient::clear),
      makeMessage("size", &KDTreeClient::size),
      makeMessage("load", &KDTreeClient::load),
      makeMessage("dump", &KDTreeClient::dump),
      makeMessage("write", &KDTreeClient::write),
      makeMessage("read", &KDTreeClient::read));
}
```

This is analogous to the parameter descriptors ubiquitous to all `Client`s, however, here we're able to just use a static member function, and there is less information to squeeze into the `makeMessage` function: it's a name (which will function as a selector), and a pointer-to-member-function describing the mapping between the name and function to invoke. Everything else (argument count and types, return type) can be deduced from the member function itself. 

```{admonition} Code Smell 
1. The message definitions happen separately to the parameter definitions, although they're both describing the interface of a client
2. They also use completely separate mechanisms, which is confusing
3. I have a horrible feeling that because we need to reference the member functions of our enclosing class, `getMessageDescriptors` has to come *after* the *declarations* of those member functions to make the compiler happy. This is brittle, hard to signpost and will lead to unclear compiler errors
4. The syntax for pointers to member functions is clunky, but that's the language's fault. Without using a macro, that's probably as good as we can do. 
5. Having to repeat all the messages from `DataClient<T>` isn't ideal: there should at least be a way to simply compose two sets of message descriptors. Think what happens if we want to change the interface of `DataClient` at the moment. 
```

### Public Utility Member Functions 
The final things in the `public` part of `KDTreeClient` are: 
```c++
InputDataSetClientRef getDataSet() const { return mDataSetClient; }

const algorithm::KDTree& algorithm() const { return mAlgorithm; }
```

These are used by other `Client` objects that might, in turn, be used by code that is talking to this `KDTree`. 

```{admonition} Code Smell 
`algorithm()` shadows but replicates exactly `DataClient<T>::algorithm()` *and furthermore* relies on a `protected` member variable (which I don't think are good things). This function should go
```

## Making `KDTreeClient` Shareable 

Following the `KDTreeClient` class is this easily-missed line
```c++
using KDTreeRef = SharedClientRef<const KDTreeClient>;
```

This is important because it sets up the necessary stuff for other `Client`s to reference shared instances of `KDTreeClient`, which we need to immediately following this, in the real-time `KDTreeQuery` class...

## Enabling Real-time Inference 

> ...in Supercollider at least

These message-based model objects are inherently offline beasts: they don't work on streams of data, rather they lurk about, holding on to state, which can be accessed or mutated through a range of different messages, as we've seen. That's all very well, but it's pretty likely that we will want to perform inference / querying on these objects with things that *are* streams of data. 

Because, in SuperCollider, real-time streaming objects (i.e. `UGens`) inhabit a different universe to what we've constructed to deal with these stateful model objects who do all their processing on `scsynth`'s nrt command thread, we need to define a different type of real-time streaming `Client` that will communicate with a reference to a stateful model object and expose the inference functions for use in synths. 

````{admonition} Code Smell 
```{rubric} pooooo-ey! Why do we need to make a *whole new client for this*? 
```
Yes, it sucks. It's quite possible that we could try and refactor things to obviate all that follows. After all, we're not using the `process()` method for anything in `KDTreeClient`. Basically, we just want the framework to be clever enough to generate a second SC plugin, sensibly named, that is a `UGen` with a `.kr` method which should be pretty predictable. 
````

The setup for this second `Client` repeats the same basic form as we've already seen, so we'll cover it more quickly, and then look more closely at what happens in `process()`.

### The Setup, Which Should Look Familiar

The same basic steps are observed as with `KDTreeClient`:

* A `constexpr` variable describing the parameters is created
* The class has an `enum` that is used for indexing the parameters
* There's some regrettable boilerplate 
* The constructor initializes the object
* As well as a `process()` member function, there is a `latency()` member function used by the framework (see [`PitchClient`][] for more on this)

The parameters are very similar to `KDTreeClient`'s 
```c++
constexpr auto KDTreeQueryParams = defineParameters(
    KDTreeRef::makeParam("tree", "KDTree"),
    LongParam("numNeighbours", "Number of Nearest Neighbours", 1),
    FloatParam("radius", "Maximum distance", 0, Min(0)),
    InputDataSetClientRef::makeParam("dataSet", "DataSet Name"),
    InputBufferParam("inputPointBuffer", "Input Point Buffer"),
    BufferParam("predictionBuffer", "Prediction Buffer"));
```

Except that: 
1. The `name` string parameter has been replaced by a `KDTreeRef`, which is – you've guessed – a reference to a `KDTreeClient` (hence needing to setup the sharing alias before declaring the query client)
2. Three new parameters are added. The input and output buffers are used to contain the input query point and the algorithm's response, respectively. The `dataSet` parameter is a reference to a `dataset` instance, whose purpose will made clear presently. 

The `enum` and parameter-related boilerplate in the class hopefully hold no surprises at this point. Meanwhile, the constructor should also follow a pattern that's familiar:
```c++
KDTreeQuery(ParamSetViewType& p, FluidContext& c) 
    : mParams(p), mRTBuffer(c.allocator()) 
{
  controlChannelsIn(1);
  controlChannelsOut({1, 1});
}
```
It initializes member variables, and sets up the channel counts for this instance (which are always the same: we take a `bufnum` in, and return a `bufnum`). 

### The `process()` Member Function 

This will also look very familiar, at least in its broad strokes
```c++
template <typename T>
void process(std::vector<FluidTensorView<T, 1>>& input,
              std::vector<FluidTensorView<T, 1>>& output, FluidContext& c)
{
  // pass through actual input signals. This is weird and mysterious and I don't like it 
  output[0] <<= input[0];

  if (input[0](0) > 0)
  {
    // Get and check reference to KDTreeClient instance
    // Does it exist? 
    auto kdtreeptr = get<kTree>().get().lock();
    if (!kdtreeptr)
      return;

    // Has it been fitted to anything? 
    if (!kdtreeptr->initialized())
      return;

    // Are we asking for a valid number of neighbours? 
    index k = get<kNumNeighbors>();
    if (k > kdtreeptr->size() || k <= 0) return;
    
    // Are the input and output buffers extant and valid? 
    index             dims = kdtreeptr->dims();
    InOutBuffersCheck bufCheck(dims);
    if (!bufCheck.checkInputs(get<kInputBuffer>().get(),
                              get<kOutputBuffer>().get()))
      return;

    // Get and check reference to `DataSetClient` instance
    auto datasetClientPtr = get<kDataSet>().get().lock();    
    // if one wasn't passed, use the `KDTree` client's 
    if (!datasetClientPtr)
      datasetClientPtr = kdtreeptr->getDataSet().get().lock();    
    // is the reference valid? 
    if (!datasetClientPtr)
      return;
    
    // fetch actual data containers from handles
    auto  dataset = datasetClientPtr->getDataSet();
    index pointSize = dataset.pointSize();
    auto  outBuf = BufferAdaptor::Access(get<kOutputBuffer>().get());    
    index outputSize = k * pointSize;
    if (outBuf.samps(0).size() < outputSize) return;

    // convert input data to double for algorithm 
    RealVector point(dims, c.allocator());
    point <<= BufferAdaptor::ReadAccess(get<kInputBuffer>().get())
                  .samps(0, dims, 0);
    
    // ensure output buffer is big enough 
    if (mRTBuffer.size() != outputSize)
    {
      mRTBuffer = RealVector(outputSize, c.allocator());
      mRTBuffer.fill(0);
    }

    //finally, query the tree 
    auto [dists, ids] =
        kdtreeptr->algorithm().kNearest(point, k, 0, c.allocator());

    // we can't output string ids, rather output 
    // the **data points corresponding to those ids from our DataSet** 
    for (index i = 0; i < k; i++)
    {
      dataset.get(*ids[asUnsigned(i)],
                  mRTBuffer(Slice(i * pointSize, pointSize)));
    }
    outBuf.samps(0, outputSize, 0) <<= mRTBuffer;
  }
```

So, the actual action is identical to `KDTreeClient::kNearest[Dist]`: 
```c++
auto [dists, ids] =
    kdtreeptr->algorithm().kNearest(point, k, 0, c.allocator());
```
except that we're talking through a pointer now (and appear to be ignoring the `radius`???). 

The other steps are analogous and / or we've seen similar: 
* Lots of boilerplate for checking the validity of input handles
* Converting those handles to concrete references with predictable lifetimes
* Copying data with `FluidTensorView<T,N>::operator <<=`

Now, this business with the `DataSetClient`: because `id`s are strings (booooo), and Supercollider `UGen`s don't do strings, we need to output something else. Our solution is to output the actual data vectors that the IDs refer to instead. By default, these come from the `DataSetClient` associated with the `KDTreeClient` when we `fit`ted. However, if you want *different* data that happens to be mapped to the *same* `id`s, you can pass in an instance of `DataSetClient` that embodies that mapping, and this will take precedence over `KDTreeClient`s


```{admonition} Code Smell
* That's even more boilerplate which, again, feels like it could be automated away 
* We again have to do this float->double copy, which is irritating
* Thread-safety is basically not a thing here. If the `KDTreeClient` happens to disappear during inference, we maybe ok because we're using `std::weak_ptr`, which should extend the lifetime of its parent in such a circumstance. *However*, if the tree mutates during a call to `process()`, it wil probably crash. 
* Using `std::shared_ptr` and `std::weak_ptr` on an audio thread is dodgy anyway 
```

## Registering the Clients

The final step in this odyssey is to step up into the namespace `fluid::client` and  register the both `KDTreeClient` and `KDTreeQuery`: 
```c++
// Register `KDTreeClient` as a non-real-time Client `NRTThreadedKDTreeClient` (ewww)
using NRTThreadedKDTreeClient =
    NRTThreadingAdaptor<typename kdtree::KDTreeRef::SharedType>;
// Register KDTreeQuery as a real-time Client `RTKDTreeQueryClient` (ewwwww)
using RTKDTreeQueryClient = ClientWrapper<kdtree::KDTreeQuery>;
```
```{note}
Note that when we register `KDTreeClient` we are not registering the class directly, but the shared reference type [we derived above](#making-kdtreeclient-shareable)
```

```{admonition} Code Smell
* That's an easily-missed thing 
* Using the threading adaptor is confusing because these objects don't (yet) support delegation to a worker thread 
* The names of these aliases are *painful* 
```