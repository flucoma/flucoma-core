/*
Part of the Fluid Corpus Manipulation Project (http://www.flucoma.org/)
Copyright 2017-2019 University of Huddersfield.
Licensed under the BSD-3 License.
See license.md file in the project root for full license information.
This project has received funding from the European Research Council (ERC)
under the European Union’s Horizon 2020 research and innovation programme
(grant agreement No 725899).
*/
/// FluidTensor and FluidTensor View
/// based lovingly on Stroustrop's in C++PL4 and Andrew Sullivan's in Origin

#pragma once

#include "FluidTensor_Support.hpp"
#include "FluidIndex.hpp"
#include <array>
#include <cassert>
#include <initializer_list>
#include <iostream>
#include <numeric>
#include <vector>

namespace fluid {
/// FluidTensor is the main container class.
template <typename T, size_t N>
class FluidTensor;
/// FluidTensorView gives you a view over some part of the container or a
/// pointer
template <typename T, size_t N>
class FluidTensorView;

///*****************************************************************************
/// Printing
namespace impl {
template <typename TensorThing>
std::enable_if_t<(TensorThing::order > 1), std::ostream&>
printTensorThing(std::ostream& o, TensorThing& t)
{
  for (index i = 0; i < t.rows(); ++i) o << t.row(i) << '\n';
  return o;
}

template <typename TensorThing>
std::enable_if_t<(TensorThing::order == 1), std::ostream&>
printTensorThing(std::ostream& o, TensorThing& t)
{
  auto first = t.begin();
  o << *first++;
  for (auto x = first; x != t.end(); ++x) o << ',' << *x;
  return o;
}
} // namespace impl


///*****************************************************************************
/// FluidTensor

template <typename T, size_t N>
class FluidTensor //: public FluidTensorBase<T,N>
{
  // embed this so we can change our mind
  using Container =
      std::vector<std::remove_const_t<std::remove_reference_t<T>>>;

public:
  static constexpr size_t order = N;
  using type = std::remove_reference_t<T>;
  // expose this so we can use as an iterator over elements
  using iterator = typename Container::iterator;
  using const_iterator = typename Container::const_iterator;

  // Default constructor / destructor
  explicit FluidTensor() = default;
  ~FluidTensor() = default;

  // Move
  FluidTensor(FluidTensor&&) noexcept = default;
  FluidTensor& operator=(FluidTensor&&) noexcept = default;

  // Copy
  FluidTensor(const FluidTensor&) = default;
  FluidTensor& operator=(const FluidTensor&) = default;

  /// Conversion constructors
  template <typename U, size_t M>
  explicit FluidTensor(const FluidTensor<U, M>& x)
      : mContainer(x.size()), mDesc(x.descriptor())
  {
    static_assert(std::is_convertible<U, T>(),
                  "Cannot convert between container value types");
    std::copy(x.begin(), x.end(), mContainer.begin());
  }

  template <typename U, size_t M>
  explicit FluidTensor(const FluidTensorView<U, M>& x)
      : mContainer(x.size()), mDesc(0, x.descriptor().extents)
  {
    static_assert(std::is_convertible<U, T>(),
                  "Cannot convert between container value types");

    std::copy(x.begin(), x.end(), mContainer.begin());
  }

  /// Conversion assignment
  template <typename U, template <typename, size_t> class O, size_t M = N>
  std::enable_if_t<std::is_same<FluidTensor<U, N>, O<U, M>>() && (N > 1),
                   FluidTensor&>
  operator=(const O<U, M>& x)
  {
    mDesc = x.descriptor();
    mContainer.assign(x.begin(), x.end());
    return *this;
  }

  /// Construct from list of extents
  template <typename... Dims,
            typename = std::enable_if_t<isIndexSequence<Dims...>()>>
  FluidTensor(Dims... dims) : mDesc(dims...)
  {
    static_assert(sizeof...(dims) == N, "Number of dimensions doesn't match");
    mContainer.resize(asUnsigned(mDesc.size));
  }

  /// Construct/assign from nested initializer_list of elements
  FluidTensor(FluidTensorInitializer<T, N> init)
      : mDesc(0, impl::deriveExtents<N>(init))
  {
    mContainer.reserve(this->mDesc.size);
    impl::insertFlat(init, mContainer);
    assert(mContainer.size() == this->mDesc.size);
  }

  FluidTensor& operator=(FluidTensorInitializer<T, N> init)
  {
    FluidTensor f = FluidTensor(init);
    return f;
  }

  /// Delete the standard initalizer_list constructors
  template <typename U>
  FluidTensor(std::initializer_list<U>) = delete;
  template <typename U>
  FluidTensor& operator=(std::initializer_list<U>) = delete;

  /// Copy from a view
  FluidTensor& operator=(const FluidTensorView<T, N> x)
  {
    mDesc = x.descriptor(); // we get the same size, extent and strides
    mDesc.start = 0;        // but start at 0 now
    mContainer.resize(mDesc.size);
    std::copy(x.begin(), x.end(), mContainer.begin());
    return *this;
  }

  /// Converting copy from view
  template <typename U, size_t M>
  FluidTensor& operator=(const FluidTensorView<U, M> x)
  {
    static_assert(M <= N, "View has too many dimensions");
    static_assert(std::is_convertible<U, T>(), "Cannot convert between types");
    // TODO this will barf if they have different orders:  I don't want that
    assert(sameExtents(mDesc, x.descriptor()));

    std::copy(x.begin(), x.end(), begin());
    return *this;
  }

  /// implicit cast to view
  /// TODO should be const T, but need to see what breaks
  operator const FluidTensorView<const T, N>() const { return {mDesc, data()}; }
  operator FluidTensorView<T, N>() { return {mDesc, data()}; }

  /// TODO If these aren't used, removed
  /// 2D copy from T**
//  template <typename U = T, size_t D = N, typename = std::enable_if_t<D == 2>()>
//  FluidTensor(T** input, size_t dim1, size_t dim2)
//      : mContainer(dim1 * dim2, 0), mDesc(0, {dim1, dim2})
//  {
//    for (int i = 0; i < dim1; ++i)
//      std::copy(input[i], input[i] + dim2, mContainer.data() + (i * dim2));
//  }

  /// 1D copy from T*
  template <typename U = T, size_t D = N, typename = std::enable_if_t<D == 1>()>
  FluidTensor(T* input, index dim, index stride = 1)
      : mContainer(dim), mDesc(0, {dim})
  {
    for (index i = 0, j = 0; i < dim; ++i, j += stride)
      mContainer[asUnsigned(i)] = input[asUnsigned(j)];
  }

  /// 1D copy from std::vector
  template <typename U = T, size_t D = N, typename = std::enable_if_t<D == 1>()>
  FluidTensor(std::vector<T>&& input)
      : mContainer(input), mDesc(0, {asSigned(input.size())})
  {}

  template <typename U = T, size_t D = N, typename = std::enable_if_t<D == 1>()>
  FluidTensor(std::vector<T>& input)
      : mContainer(input), mDesc(0, {asSigned(input.size())})
  {}


  ///***************************************************************************
  /// row / col access: returns Views of dim N-1
  const FluidTensorView<const T, N - 1> row(const index i) const
  {
    assert(i < rows());
    FluidTensorSlice<N - 1> row(mDesc, SizeConstant<0>(), i);
    return {row, mContainer.data()};
  }

  FluidTensorView<T, N - 1> row(const index i)
  {
    assert(i < rows());
    FluidTensorSlice<N - 1> row(mDesc, SizeConstant<0>(), i);
    return {row, mContainer.data()};
  }

  const FluidTensorView<const T, N - 1> col(const index i) const
  {
    assert(i < cols());
    FluidTensorSlice<N - 1> col(mDesc, SizeConstant<1>(), i);
    return {col, data()};
  }

  FluidTensorView<T, N - 1> col(const index i)
  {
    assert(i < cols());
    FluidTensorSlice<N - 1> col(mDesc, SizeConstant<1>(), i);
    return {col, data()};
  }

  ///***************************************************************************
  /// slicing and element access (see Slice class)

  FluidTensorView<T, N - 1> operator[](const index i) { return row(i); }

  const FluidTensorView<const T, N - 1> operator[](const index i) const
  {
    return row(i);
  }

  /// element access
  template <typename... Args>
  std::enable_if_t<isIndexSequence<Args...>(), T&>
  operator()(Args... args)
  {
    assert(impl::checkBounds(mDesc, args...) && "Arguments out of bounds");
    return *(data() + mDesc(index(args)...));
  }

  template <typename... Args>
  std::enable_if_t<isIndexSequence<Args...>(), const T&>
  operator()(Args... args) const
  {
    assert(impl::checkBounds(mDesc, args...) && "Arguments out of bounds");
    return *(data() + mDesc(Index(args)...));
  }

  /// Slicing
  template <typename... Args>
  std::enable_if_t<isSliceSequence<Args...>(),
                   const FluidTensorView<const T, N>>
  operator()(const Args&... args) const
  {
    static_assert(sizeof...(Args) == N,
                  "Number of slices must match number of dimensions. Use "
                  "an integral constant to represent the whole of a "
                  "dimension,e.g. matrix(1,slice(0,10)).");
    FluidTensorSlice<N> d{mDesc, args...};
    return {d, data()};
  }

  template <typename... Args>
  std::enable_if_t<isSliceSequence<Args...>(), FluidTensorView<T, N>>
  operator()(const Args&... args)
  {
    static_assert(sizeof...(Args) == N,
                  "Number of slices must match number of dimensions. Use "
                  "an integral constant to represent the whole of a "
                  "dimension,e.g. matrix(1,slice(0,10)).");
    FluidTensorSlice<N> d{mDesc, args...};
    return {d, data()};
  }

  iterator       begin() { return mContainer.begin(); }
  iterator       end() { return mContainer.end(); }
  const_iterator begin() const { return mContainer.cbegin(); }
  const_iterator end() const { return mContainer.cend(); }

  index extent(index n) const { return mDesc.extents[asUnsigned(n)]; };
  index rows() const { return extent(0); }
  index cols() const { return extent(1); }
  index size() const { return asSigned(mContainer.size()); }
  const FluidTensorSlice<N>& descriptor() const { return mDesc; }
  FluidTensorSlice<N>&       descriptor() { return mDesc; }
  const T*                   data() const { return mContainer.data(); }
  T*                         data() { return mContainer.data(); }

  template <typename... Dims,
            typename = std::enable_if_t<isIndexSequence<Dims...>()>>
  void resize(Dims... dims)
  {
    static_assert(sizeof...(dims) == N, "Number of dimensions doesn't match");
    mDesc = FluidTensorSlice<N>(dims...);
    mContainer.resize(asUnsigned(mDesc.size));
  }

  void resizeDim(index dim, index amount)
  {
    if (amount == 0) return;
    mDesc.grow(dim, amount);
    mContainer.resize(mDesc.size);
  }

  // Specialise for N=1
  template <typename dummy = void>
  std::enable_if_t<N == 1, dummy> deleteRow(index index)
  {
    auto begin = mContainer.begin() + index;
    auto end = begin + 1;
    mContainer.erase(begin, end);
    mDesc.grow(0, -1);
  }

  template <typename dummy = void>
  std::enable_if_t<(N > 1), dummy> deleteRow(index index)
  {
    auto r = row(index);
    auto begin = mContainer.begin() + r.descriptor().start;
    auto end = begin + r.descriptor().size;
    mContainer.erase(begin, end);
    mDesc.grow(0, -1);
  }

  void fill(T v) { std::fill(mContainer.begin(), mContainer.end(), v); }

  FluidTensorView<T, N> transpose() { return {mDesc.transpose(), data()}; }
  const FluidTensorView<const T, N> transpose() const
  {
    return {mDesc.transpose(), data()};
  }

  template <typename F>
  FluidTensor& apply(F f)
  {
    for (auto i = begin(); i != end(); ++i) f(*i);
    return *this;
  }

  // Passing by value here allows to pass r-values
  template <typename M, typename F>
  FluidTensor& apply(M m, F f)
  {
    sameExtents(*this, m);
    auto i = begin();
    auto j = m.begin();
    for (; i != end(); ++i, ++j) f(*i, *j);
    return *this;
  }

  friend std::ostream& operator<<(std::ostream& o, const FluidTensor& t)
  {
    return impl::printTensorThing(o, t);
  }

private:
  Container           mContainer;
  FluidTensorSlice<N> mDesc;
};

/// A 0-dim container is just a scalar
template <typename T>
class FluidTensor<T, 0>
{
public:
  static constexpr size_t order = 0;
  using value_type = T;

  FluidTensor(const T& x) : elem(x) {}
  FluidTensor& operator=(const T& value)
  {
    elem = value;
    return this;
  }
  operator T&() { return elem; }
  operator const T&() const { return elem; }

  T&       operator()() { return elem; }
  const T& operator()() const { return elem; }
  index   size() const { return 1; }

private:
  T elem;
};

///*****************************************************************************
/// FluidTensorView
template <typename T, size_t N>
class FluidTensorView
{
public:
  /*****
   STL style shorthand
   *****/
  using pointer = T*;
  using iterator = impl::SliceIterator<T, N>;
  using const_iterator = impl::SliceIterator<const T, N>;
  using type = std::remove_reference_t<T>;
  static constexpr size_t order = N;

  FluidTensorView() = delete;
  ~FluidTensorView() = default;

  /// Construct from a slice and a pointer.
  /// This gets used by row() and col() of FluidTensor and FluidTensorView
  FluidTensorView(const FluidTensorSlice<N>& s, T* p) : mDesc(s), mRef(p) {}

  /// Construct around an arbitary pointer, with an offset and some dimensions
  template <typename... Dims,
            typename = std::enable_if_t<isIndexSequence<Dims...>()>>
  FluidTensorView(T* p, index start, Dims... dims)
      : mDesc(start, {static_cast<index>(dims)...}), mRef(p)
  {}


  // Convert to a larger dim by adding single size dim, like numpy newaxis
  explicit FluidTensorView(FluidTensorView<T, N - 1> x)
  {
    mDesc.start = x.descriptor().start;
    std::copy_n(x.descriptor().extents.begin(), N - 1,
                mDesc.extents.begin() + 1);
    std::copy_n(x.descriptor().strides.begin(), N - 1, mDesc.strides.begin());
    mDesc.extents[0] = 1;
    mDesc.strides[N - 1] = 1;
    mDesc.size = x.descriptor().size;
    mRef = x.data() - mDesc.start;
  }

  /**********
   Disable assigning a FluidTensorView from an r-value FluidTensor, as that's a
  gurranteed memory leak, i.e. you can't do FluidTensorView<double,1> r =
  FluidTensor(double,2);
  **********/
  FluidTensorView(FluidTensor<T, N>&& r) = delete;


  // Move construction is allowed
  FluidTensorView(FluidTensorView&& other) noexcept { swap(*this, other); }

  /// Move assignment disabled because it doesn't make sense to move from a
  /// possibly arbitary pointer into the middle of what might be a FluidTensor's
  /// vector
  /// ==> assignment is always copy


  // Copy
  FluidTensorView(FluidTensorView const&) = default;

  // Copy assignment from same type
  FluidTensorView& operator=(const FluidTensorView& x)
  {
    assert(sameExtents(mDesc, x.descriptor()));
    std::array<index, N> a;
    // Get the element-wise minimum of our extents and x's
    std::transform(mDesc.extents.begin(), mDesc.extents.end(),
                   x.descriptor().extents.begin(), a.begin(),
                   [](index a, index b) { return std::min(a, b); });

    index count =
        std::accumulate(a.begin(), a.end(), index(1), std::multiplies<index>());

    // Have to do this because haven't implemented += for slice iterator
    // (yet), so can't stop at arbitary offset from begin
    auto it = x.begin();
    auto ot = begin();
    for (index i = 0; i < count; ++i, ++it, ++ot) *ot = *it;
    return *this;
  }

  // Copy from Tensor of same type
  FluidTensorView& operator=(const FluidTensor<T, N>& x)
  {
    assert(sameExtents(mDesc, x.descriptor()));
    std::array<index, N> a;
    // Get the element-wise minimum of our extents and x's
    std::transform(mDesc.extents.begin(), mDesc.extents.end(),
                   x.descriptor().extents.begin(), a.begin(),
                   [](index a, index b) { return std::min(a, b); });
    size_t count =
        std::accumulate(a.begin(), a.end(), index(1), std::multiplies<index>());

    // Have to do this because haven't implemented += for slice iterator
    // (yet), so can't stop at arbitary offset from begin
    auto it = x.begin();
    auto ot = begin();
    for (int i = 0; i < count; ++i, ++it, ++ot) *ot = *it;
    return *this;
  }

  /// Converting copy
  template <typename U>
  FluidTensorView& operator=(const FluidTensorView<U, N> x)
  {
    static_assert(std::is_convertible<T, U>(), "Can't convert between types");
    assert(sameExtents(mDesc, x.descriptor()));
    std::array<index, N> a;
    // Get the element-wise minimum of our extents and x's
    std::transform(mDesc.extents.begin(), mDesc.extents.end(),
                   x.descriptor().extents.begin(), a.begin(),
                   [](index a, index b) { return std::min(a, b); });

    index count =
        std::accumulate(a.begin(), a.end(), index(1), std::multiplies<index>());

    // Have to do this because haven't implemented += for slice iterator (yet),
    // so can't stop at arbitary offset from begin
    auto it = x.begin();
    auto ot = begin();
    for (index i = 0; i < count; ++i, ++it, ++ot) *ot = static_cast<T>(*it);
    return *this;
  }

  // Converting copy from Tensor
  template <typename U>
  FluidTensorView& operator=(FluidTensor<U, N>& x)
  {
    static_assert(std::is_convertible<T, U>(), "Can't convert between types");
    assert(sameExtents(*this, x));
    std::copy(x.begin(), x.end(), begin());
    return *this;
  }

  /// Repoint a view (TODO const version?)
  template <typename... Dims,
            typename = std::enable_if_t<isIndexSequence<Dims...>()>>
  void reset(T* p, index start, Dims... dims)
  {
    mRef = p;
    mDesc.reset(start, {static_cast<index>(dims)...});
  }

  /// Element access
  template <typename... Args>
  std::enable_if_t<isIndexSequence<Args...>(), const T&>
  operator()(Args... args) const
  {
    assert(impl::checkBounds(mDesc, args...) && "Arguments out of bounds");
    return *(data() + mDesc(Index(args)...));
  }

  template <typename... Args>
  std::enable_if_t<isIndexSequence<Args...>(), T&> operator()(Args... args)
  {
    assert(impl::checkBounds(mDesc, args...) && "Arguments out of bounds");
    return *(data() + mDesc(index(args)...));
  }

  /// slicing
  template <typename... Args>
  std::enable_if_t<isSliceSequence<Args...>(), FluidTensorView<T, N>>
  operator()(const Args&... args) const
  {
    static_assert(sizeof...(Args) == N,
                  "Number of slices must match number of dimensions. Use "
                  "an integral constant to represent the whole of a "
                  "dimension,e.g. matrix(1,slice(0,10)).");
    FluidTensorSlice<N> d{mDesc, args...};
    return {d, mRef};
  }

  iterator             begin() { return {mDesc, mRef}; }
  const const_iterator begin() const { return {mDesc, mRef}; }
  iterator             end() { return {mDesc, mRef, true}; }
  const const_iterator end() const { return {mDesc, mRef, true}; }

  /// c style element access
  FluidTensorView<T, N - 1> operator[](const index i) { return row(i); }
  /// TODO sould be const T
  const FluidTensorView<const T, N - 1> operator[](const index i) const
  {
    return row(i);
  }

  const FluidTensorView<const T, N - 1> row(const index i) const
  {
    assert(i < extent(0));
    FluidTensorSlice<N - 1> row(mDesc, SizeConstant<0>(), i);
    return {row, mRef};
  }

  FluidTensorView<T, N - 1> row(const index i)
  {
    assert(i < extent(0));
    FluidTensorSlice<N - 1> row(mDesc, SizeConstant<0>(), i);
    return {row, mRef};
  }

  const FluidTensorView<const T, N - 1> col(const index i) const
  {
    assert(i < extent(1));
    FluidTensorSlice<N - 1> col(mDesc, SizeConstant<1>(), i);
    return {col, mRef};
  }

  FluidTensorView<T, N - 1> col(const index i)
  {
    assert(i < extent(1));
    FluidTensorSlice<N - 1> col(mDesc, SizeConstant<1>(), i);
    return {col, mRef};
  }

  index extent(const index n) const
  {
    assert(n < asSigned(mDesc.extents.size()));
    return mDesc.extents[asUnsigned(n)];
  }

  index rows() const { return mDesc.extents[0]; }
  index cols() const { return order > 1 ? mDesc.extents[1] : 0; }
  index size() const { return mDesc.size; }
  void   fill(const T x) { std::fill(begin(), end(), x); }

  FluidTensorView<T, N> transpose() { return {mDesc.transpose(), mRef}; }
  const FluidTensorView<const T, N> transpose() const
  {
    return {mDesc.transpose(), mRef};
  }

  template <typename F>
  FluidTensorView& apply(F f)
  {
    for (auto i = begin(); i != end(); ++i) f(*i);
    return *this;
  }

  // Passing by value here allows to pass r-values
  // this tacilty assumes at the moment that M is
  // a FluidTensor or FluidTensorView. Maybe this should be more explicit
  template <typename M, typename F>
  FluidTensorView& apply(M m, F f)
  {
    // TODO: ensure same size? Ot take min?
    assert(m.descriptor().extents == mDesc.extents);
    assert(!(begin() == end()));
    auto i = begin();
    auto j = m.begin();
    for (; i != end(); ++i, ++j) f(*i, *j);
    return *this;
  }

  const T* data() const { return mRef + mDesc.start; }
  pointer  data() { return mRef + mDesc.start; }

  const FluidTensorSlice<N> descriptor() const { return mDesc; }
  FluidTensorSlice<N>       descriptor() { return mDesc; }

  friend void swap(FluidTensorView& first, FluidTensorView& second)
  {
    using std::swap;
    swap(first.mDesc, second.mDesc);
    swap(first.mRef, second.mRef);
  }

  friend std::ostream& operator<<(std::ostream& o, const FluidTensorView& t)
  {
    return impl::printTensorThing(o, t);
  }

private:
  FluidTensorSlice<N> mDesc;
  pointer             mRef;
};

template <typename T>
class FluidTensorView<T, 0>
{
public:
  using value_type = T;
  using const_value_type = const T;
  using pointer = T*;
  using reference = T&;

  FluidTensorView() = delete;

  FluidTensorView(const FluidTensorSlice<0>& s, pointer x)
      : elem(x + s.start), mStart(s.start)
  {}

  FluidTensorView& operator=(value_type& x)
  {
    *elem = x;
    return *this;
  }

  template <typename U>
  FluidTensorView& operator=(U& x)
  {
    static_assert(std::is_convertible<T, U>(), "Can't convert");
    *elem = static_cast<T>(x);
    return *this;
  }

  value_type       operator()() { return *elem; }
  const_value_type operator()() const { return *elem; }

  operator value_type&() { return *elem; };
  operator const_value_type&() const { return *elem; }

  friend std::ostream& operator<<(std::ostream& o, const FluidTensorView& t)
  {
    o << t();
    return o;
  }

private:
  pointer elem;
  index  mStart;
}; // View<T,0>

} // namespace fluid
