/*
Part of the Fluid Corpus Manipulation Project (http://www.flucoma.org/)
Copyright University of Huddersfield.
Licensed under the BSD-3 License.
See license.md file in the project root for full license information.
This project has received funding from the European Research Council (ERC)
under the European Union’s Horizon 2020 research and innovation programme
(grant agreement No 725899).
*/

/// Support Code for FluidTensor. This is based on Stroustrop / Andrew
/// Sullivan's implementations from C++PL4 and the origin lib

#pragma once

#include "FluidIndex.hpp"
#include "FluidMeta.hpp"
#include <algorithm>  //copy,copy_n
#include <array>      //std::array
#include <cassert>    //assert()
#include <functional> // less, multiplies
#include <numeric>    //accujmuate, innerprodct

#include <iostream>
namespace fluid {

///*****************************************************************************
/// slice
/// Used for requesting slices from client code using operater() on FluidTensor
/// and FluidTensorView.
///*****************************************************************************
struct Slice
{
  Slice() : start(-1), length(-1), stride(1) {}
  explicit Slice(index s) : start(s), length(-1), stride(1) {}
  Slice(index s, index l, index n = 1) : start(s), length(l), stride(n) {}

  index start;
  index length;
  index stride;
};

///*****************************************************************************
/// FluidTensorSlice describes mappings of indices to points in our storage
///*****************************************************************************
template <size_t N>
struct FluidTensorSlice;

template <typename... Args>
constexpr bool isSliceSequence()
{
  return all((std::is_convertible<Args, index>() ||
              std::is_same<Args, Slice>())...) &&
         some(std::is_same<Args, Slice>()...);
}

template <typename... Args>
constexpr bool isIndexSequence()
{
  return all(std::is_convertible<Args, index>()...);
}


// Alias integral_constant<size_t,N>
template <std::size_t N>
using SizeConstant = std::integral_constant<index, N>;

namespace impl {
///*****************************************************************************
/// Helper templates for the container.

/// Ensure that a set of dimension extents will fit within the FluidTensorSlice
template <size_t N, typename... Dims>
bool checkBounds(const fluid::FluidTensorSlice<N>& slice, Dims... dims)
{
  index indexes[N]{index(dims)...};
  return std::equal(indexes, indexes + N, slice.extents.begin(),
                    std::less<index>{});
}

/// FluidTensorInit is used for instantiating a tensor with values from a braced
/// list

/// Recursion
template <typename T, size_t N>
struct FluidTensorInit
{
  using type = std::initializer_list<typename FluidTensorInit<T, N - 1>::type>;
};

// Terminating case :
template <typename T>
struct FluidTensorInit<T, 1>
{
  using type = std::initializer_list<T>;
};

template <typename T>
struct FluidTensorInit<T, 0>; /// things should barf if this happens

/// addExtents & deriveExtents drill down through the nested init lists to work
/// out the dimensions of the resulting structure.
template <std::size_t N, typename I, typename List> /// terminate
std::enable_if_t<(N == 1)> addExtents(I& first, const List& list)
{
  *first++ = asSigned(list.size()); // deepest nesting
}

template <std::size_t N, typename I, typename List> /// recursion
std::enable_if_t<(N > 1)> addExtents(I& first, const List& list)
{
  *first = asSigned(list.size());
  addExtents<N - 1>(++first, *list.begin());
}


/// This is the function we call from main code.
template <size_t N, typename List>
std::array<index, N> deriveExtents(const List& list)
{
  std::array<index, N> a;
  auto                 f = a.begin();
  addExtents<N>(f, list);
  return a;
}


/// addList and insertFlat populate our container with the contents of
/// a nested initializer list structure.

template <class T, class Vec> // terminate
void addList(const T* first, const T* last, Vec& vec)
{
  vec.insert(vec.end(), first, last);
}

template <class T, class Vec> // recurse
void addList(const std::initializer_list<T>* first,
             const std::initializer_list<T>* last, Vec& vec)
{
  for (; first != last; ++first) addList(first->begin(), first->end(), vec);
}


// Recurse
template <class T, class Vec>
void insertFlat(const std::initializer_list<T> list, Vec& vec)
{
  addList(list.begin(), list.end(), vec);
}

/// Forward iterator for iterating over a FluidTensorSlice with strides
template <typename T, size_t N>
struct SliceIterator
{
  using value_type = typename std::remove_const<T>::type;
  using reference = T&;
  using pointer = T*;
  using difference_type = std::ptrdiff_t;
  using iterator_category = std::forward_iterator_tag;

  SliceIterator(const FluidTensorSlice<N>& s, pointer base, bool end = false)
      : mDesc(s), mBase(base)
  {
    std::fill_n(mIndexes.begin(), N, 0);
    
    mIndexes[0] = s.extents[0];
    
    if (!s.transposed) 
      endoffset = s.strides[0] * s.size ; 
    else 
      endoffset = s.strides[N - 1] * s.size; 
    mPtr = end ? base + s.start + endoffset : base + s.start;
  }
  
  const FluidTensorSlice<N>& descriptor() { return mDesc; }

  T& operator*() const { return *mPtr; }
  T* operator->() const { return mPtr; }

  // Forward iterator pre- and post-increment
  SliceIterator& operator++()
  {
    increment();
    return *this;
  }

  SliceIterator operator++(int)
  {
    SliceIterator tmp = *this;
    increment();
    return tmp;
  }

  bool operator==(const SliceIterator& rhs)
  {
    assert(mDesc == rhs.mDesc);
    return (mPtr == rhs.mPtr);
  }

  bool operator!=(const SliceIterator& rhs) { return !(*this == rhs); }

private:

  index counter{0};
 // index size;
  index endoffset;

  /// TODO I would like this to be more beautiful (this is from Origin impl)
  void increment()
  {
    if(++counter == mDesc.size)
    {
      mPtr = mBase + mDesc.start + endoffset;
      return;
    }
        
    index d = N - 1;
    while (true)
    {
      mPtr += mDesc.strides[asUnsigned(d)];
      ++mIndexes[asUnsigned(d)];
      // If have not yet counted to the extent of the current dimension, then
      // we will continue to do so in the next iteration.
      if (mIndexes[asUnsigned(d)] != mDesc.extents[asUnsigned(d)]) break;

      // Otherwise, if we have not counted to the extent in the outermost
      // dimension, move to the next dimension and try again. If d is 0, then
      // we have counted through the entire slice.
      if (d != 0)
      {
        mPtr -= mDesc.strides[asUnsigned(d)] * mDesc.extents[asUnsigned(d)];
        mIndexes[asUnsigned(d)] = 0;
        --d;
      }
      else
      {
        break;
      }
    }
  }

  FluidTensorSlice<N>  mDesc;
  std::array<index, N> mIndexes;
  pointer              mPtr;
  pointer              mBase;
};
} // namespace impl
///*****************************************************************************
template <typename T, size_t N>
using FluidTensorInitializer = typename impl::FluidTensorInit<T, N>::type;
///*****************************************************************************
/// FluidTensorSlice describes the shape of a Tensor or TensorView,
/// and provides a mapping between points in the flat storage and indicies
/// expressed in N dimensions. It comprises extents (the size of each
/// dimension), strides (the distance to travel in each dimension) and a start
/// point.

template <size_t N>
struct FluidTensorSlice
{
  static constexpr std::size_t order = N;
  // Standard constructors
  FluidTensorSlice()
  {
    std::fill(extents.begin(), extents.end(), 0);
    std::fill(strides.begin(), strides.end(), 0);
    size = 0;
    start = 0;
  };
  // Copy
  FluidTensorSlice(FluidTensorSlice const& other) = default;
  FluidTensorSlice& operator=(const FluidTensorSlice&) = default;

  // Move
  FluidTensorSlice(FluidTensorSlice&& x) noexcept { *this = std::move(x); }
  FluidTensorSlice& operator=(FluidTensorSlice&& other) noexcept = default; 
//  {
//    if (this != &other) swap(*this, other);
//    return *this;
//  }

  /// Construct slice from forward iterator
  template <typename R,
            typename I = typename std::remove_reference<R>::type::iterator,
            typename = std::enable_if_t<
                IsIteratorType<I, std::forward_iterator_tag>::value>>
  FluidTensorSlice(index s, R&& range) : start(s)
  {
    assert(range.size() == N && "Input list size doesn't match dimensions");
    std::copy(range.begin(), range.end(), extents.begin());
    init();
  }


  template <std::size_t M, index D>
  FluidTensorSlice(const FluidTensorSlice<M>& s, SizeConstant<D>, index n)
      : size(s.size / s.extents[D]), start(s.start + n * s.strides[D]),transposed{s.transposed}
  {
    static_assert(D <= N, "");
    static_assert(N < M, "");
    // Copy the extetns and strides, excluding the Dth dimension.
    std::copy_n(s.extents.begin() + D + 1, N - D,
                std::copy_n(s.extents.begin(), D, extents.begin()));
    std::copy_n(s.strides.begin() + D + 1, N - D,
                std::copy_n(s.strides.begin(), D, strides.begin()));
  }

  // Construct from a start point and an initializer_list of dimensions
  // e.g  FluidTensorSlice<2> my_slice<2>(s, {3,4}).
  // The number of dimenions given needs
  // to match the N that the slice is templated on.
  FluidTensorSlice(index s, std::initializer_list<index> exts) : start(s)
  {
    // TODO: we're on  14 now, so this can be enforced statically
    assert(exts.size() == N && "Wrong number of dimensions in extents");
    std::copy(exts.begin(), exts.end(), extents.begin());
    init();
  }

  /// As above, but also taking a list of strides
  FluidTensorSlice(index s, std::initializer_list<index> exts,
                   std::initializer_list<index> str)
      : start(s)
  {
    assert(exts.size() == N && "Wrong number of dimensions in extents");
    assert(str.size() == N && "Wrong number of dimensions in strides");
    std::copy(exts.begin(), exts.end(), extents.begin());
    std::copy(str.begin(), str.end(), strides.begin());
    size = extents[0] * strides[0];
  }

  /// Construct from a variable number of extents.
  template <typename... Dims,
            typename = std::enable_if_t<isIndexSequence<Dims...>()>>
  FluidTensorSlice(Dims... dims)
  {
    static_assert(sizeof...(Dims) == N,
                  "Number of arguments must match matrix dimensions");
    extents = {{index(dims)...}};
    init();
  }

  /// sub-slice
  template <size_t M, typename... Args>
  FluidTensorSlice(FluidTensorSlice<M> s, const Args... args)
  {
    start = s.start + doSlice(s, args...);
    size = std::accumulate(extents.begin(), extents.end(), index(1),
                           std::multiplies<index>());
    transposed = s.transposed;
  }

  /// Operator () is used for mapping indices back onto a flat data structure
  template <typename... Dims> // dims > 2
  std::enable_if_t<(N > 2) && isIndexSequence<Dims...>(), index>
  operator()(Dims... dims) const
  {
    static_assert(sizeof...(Dims) == N, "");
    size_t args[N]{size_t(dims)...};
    return std::inner_product(args, args + N, strides.begin(), index(0));
  }

  template <size_t DIM = N> // 1D
  std::enable_if_t<DIM == 1, index> operator()(index i) const
  {
    return i * strides[0];
  }

  template <size_t DIM = N> // 2D
  std::enable_if_t<DIM == 2, index> operator()(index i, index j) const
  {
    return i * strides[0] + j * strides[1];
  }

  void grow(index dim, index amount)
  {
    size_t uDim = asUnsigned(dim);
    
    assert(uDim < N);
    assert(extents[uDim] + amount >= 0);
    extents[uDim] += amount;
    init();
  }

  FluidTensorSlice<N> transpose()
  {
    FluidTensorSlice<N> res(*this);
    std::reverse(res.extents.begin(), res.extents.end());
    std::reverse(res.strides.begin(), res.strides.end());
    res.transposed = !transposed;
    return res;
  }

  friend void swap(FluidTensorSlice& first, FluidTensorSlice& second)
  {
    using std::swap;

    swap(first.extents, second.extents);
    swap(first.strides, second.strides);
    swap(first.size, second.size);
    swap(first.start, second.start);
    swap(first.transposed,second.transposed); 
  }

  // Slices are equal if they describe the same shape
  bool operator==(const FluidTensorSlice& rhs) const
  {
    return start == rhs.start && extents == rhs.extents &&
           strides == rhs.strides;
  }
  bool operator!=(const FluidTensorSlice& rhs) const { return !(*this == rhs); }

  index                size;      // num of elements
  index                start = 0; // offset
  bool                 transposed = false;
  std::array<index, N> extents;   // number of elements in each dimension
  std::array<index, N> strides;   // offset between elements in each dim

  void reset(index s, std::initializer_list<index> exts)
  {
    std::copy(exts.begin(), exts.end(), extents.begin());
    start = s;
    init();
  }

private:
  // No point calling this before extents have been filled
  // This will generate strides in *row major* order, given a set of
  // extents
  void init()
  {
    strides[N - 1] = 1;
    for (index i = N - 1; i != 0; --i)
      strides[asUnsigned(i - 1)] =
          strides[asUnsigned(i)] * extents[asUnsigned(i)];
     size = std::accumulate(extents.begin(), extents.end(), static_cast<index>(1),std::multiplies<index>());
  }

  /// doSliceDim does the hard work in making an new slice from an existing one
  template <size_t D, size_t M>
  index doSliceDim(const fluid::FluidTensorSlice<M>& newSlice, index n)
  {
    return doSliceDim<D>(newSlice, fluid::Slice(n, 1, 1));
  }

  template <size_t D, size_t M>
  index doSliceDim(const fluid::FluidTensorSlice<M>& ns, Slice s)
  {

    if (s.start < 0 || s.start >= ns.extents[D]) s.start = 0;

    if (s.length < 0 || s.length > ns.extents[D] ||
        s.start + s.length > ns.extents[D])
      s.length = ns.extents[D] - s.start;

    if (s.start + s.length * s.stride > ns.extents[D])
      s.length = ((ns.extents[D] - s.start) + s.stride - 1) / s.stride;

    extents[D] = s.length;
    strides[D] = ns.strides[D] * s.stride;
    size = extents[0] * strides[0];
    return s.start * ns.strides[D];
  }

  /// Recursively populate a new slice from an old one,
  template <size_t M> // terminate
  index doSlice(const fluid::FluidTensorSlice<M>&)
  {
    return 0;
  }

  template <size_t M, typename T, typename... Args> // recurse
  index doSlice(const fluid::FluidTensorSlice<M>& ns, const T& s,
                const Args&... args)
  {
    constexpr size_t D = N - sizeof...(Args) - 1;
    index            m = doSliceDim<D>(ns, s);
    index            n = doSlice(ns, args...);
    return n + m;
  }
};

template <std::size_t N, size_t M>
inline bool sameExtents(const FluidTensorSlice<N>& a,
                        const FluidTensorSlice<M>& b)
{
  return a.order == b.order &&
         std::equal(a.extents.begin(), a.extents.begin() + N,
                    b.extents.begin());
}

template <typename M1, typename M2>
inline bool sameExtents(const M1& a, const M2& b)
{
  return sameExtents(a.descriptor(), b.descriptor());
}

} // namespace fluid
