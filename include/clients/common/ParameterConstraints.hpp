#pragma once

#include "ParameterTrackChanges.hpp"
#include "ParameterTypes.hpp"
#include "Result.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
namespace fluid {
namespace client {

namespace impl {

struct Relational {};
      
template <typename T>
struct MinImpl
{
  constexpr MinImpl(const T m)
      : value(m)
  {}
  const T value;
  template <size_t Offset, size_t N, typename U, typename Tuple, typename Descriptor>
  constexpr void clamp(U &x, Tuple &params, Descriptor& d, Result *r) const
  {
    U oldX = x;
    x      = std::max<U>(x, value);
    if (r && oldX != x)
    {
      r->set(Result::Status::kWarning);
      r->addMessage(d.template name<N>()," value, ",oldX, ", below absolute minimum ", x);
    }
  }
};

template <typename T>
struct MaxImpl
{
  constexpr MaxImpl(const T m)
      : value(m)
  {}
  const T value;
  template <size_t Offset, size_t N, typename U, typename Tuple, typename Descriptor>
  constexpr void clamp(U &x, Tuple &params, Descriptor& d, Result *r) const
  {

    U oldX = x;
    x      = std::min<U>(x, value);
    if (r && oldX != x)
    {
      r->set(Result::Status::kWarning);
      r->addMessage(d.template name<N>()," value (",oldX,") above absolute maximum (",x,')');
    }
  }
};

template <int... Is>
struct LowerLimitImpl : public Relational
{
  template <size_t Offset, size_t N, typename T, typename Tuple, typename Descriptor>
  void clamp(T &v, Tuple &params, Descriptor& d, Result *r) const
  {
    T oldV = v;

    v = std::max<T>({v, std::get<Is + Offset>(params)...});

    if (r && oldV != v)
    {
      r->set(Result::Status::kWarning);
      std::array<T, sizeof...(Is)> constraintValues{std::get<Is + Offset>(params)...};
      size_t                       minPos =
          std::distance(constraintValues.begin(), std::min_element(constraintValues.begin(), constraintValues.end()));
      std::array<const char *, sizeof...(Is)> constraintNames{d.template name<Is + Offset>()...};
      r->addMessage(d.template name<N>()," value (", oldV,") below parameter ", constraintNames[minPos], " (",v,')');
    }
  }
};

template <int... Is>
struct UpperLimitImpl : public Relational
{
  template <size_t Offset, size_t N, typename T, typename Tuple, typename Descriptor>
  void clamp(T &v, Tuple &params, Descriptor& d, Result *r) const
  {
    T oldV = v;

    v = std::min<T>({v, std::get<Is + Offset>(params)...});

    if (r && oldV != v)
    {
      r->set(Result::Status::kWarning);
      std::array<T, sizeof...(Is)> constraintValues{std::get<Is + Offset>(params)...};
      size_t                       maxPos =
          std::distance(constraintValues.begin(), std::max_element(constraintValues.begin(), constraintValues.end()));
      std::array<const char *, sizeof...(Is)> constraintNames{d.template name<Is + Offset>()...};
      r->addMessage(d.template name<N>()," value, ",oldV,", above parameter ",constraintNames[maxPos]," (",v,')');
    }
  }
};

template <int FFTIndex>
struct FrameSizeUpperLimitImpl : public Relational
{
  template <size_t Offset, size_t N, typename T, typename Tuple, typename Descriptor>
  void clamp(T &v, Tuple &params, Descriptor& d, Result *r) const
  {
    T      oldV      = v;
    size_t frameSize = std::get<FFTIndex + Offset>(params).frameSize();
    v                = std::min<T>(v, frameSize);

    if (r && oldV != v)
    {
      r->set(Result::Status::kWarning);
      r->addMessage(d.template name<N>(), " value (", oldV, ") above spectral frame size (", v, ')');
    }
  }
};

} // namespace impl

template <typename T>
auto constexpr Min(const T x)
{
  return impl::MinImpl<T>(x);
};

template <typename T>
auto constexpr Max(const T x)
{
  return impl::MaxImpl<T>(x);
}

template <int... Is>
auto constexpr LowerLimit()
{
  return impl::LowerLimitImpl<Is...>{};
}

template <int... Is>
auto constexpr UpperLimit()
{
  return impl::UpperLimitImpl<Is...>{};
}

struct FrequencyAmpPairConstraint
{
  using type = typename FloatPairsArrayT::type;

  constexpr FrequencyAmpPairConstraint() {}

  template <size_t Offset, size_t N, typename Tuple, typename Descriptor>
  constexpr void clamp(type &v, Tuple &allParams, Descriptor &, Result *r) const
  {
    auto& vals = v.value;
    auto& inParams = std::get<N>(allParams);
    // For now I know that array size is 2, just upper and lower vals
    // TODO: make generic for any old monotonic array of freq-amp pairs, should we need it

    // Clip freqs to [0,1]
    vals[0].first = std::max<double>(std::min<double>(vals[0].first, 1), 0);
    vals[1].first = std::max<double>(std::min<double>(vals[1].first, 1), 0);

    inParams.lowerChanged = vals[0].first != inParams.oldLower;
    inParams.upperChanged = vals[1].first != inParams.oldUpper;

    if (v.lowerChanged && !v.upperChanged && vals[0].first > vals[1].first) vals[0].first = vals[1].first;
    if (v.upperChanged && !v.lowerChanged && vals[0].first > vals[1].first) vals[1].first = vals[0].first;
    // If everything changed (i.e. object creation) and in the wrong order, just swap 'em
    if (v.lowerChanged && v.upperChanged && vals[0].first > vals[1].first) std::swap(vals[0], vals[1]);

    inParams.oldLower = vals[0].first;
    inParams.oldUpper = vals[1].first;
  }
};

struct PowerOfTwo
{
  template <size_t Offset, size_t N, typename Tuple, typename Descriptor>
  constexpr void clamp(LongUnderlyingType &x, Tuple &params, Descriptor& d, Result *r) const
  {

    int                exp  = 0;
    double             base = std::frexp(x, &exp);
    LongUnderlyingType res  = base > 0.5 ? (1 << exp) : (1 << (exp - 1));

    if (r && res != x)
    {
      r->set(Result::Status::kWarning);
      r->addMessage(d.template name<N>()," value (",x,") adjusted to power of two (",res,')');
    }
    x = res;
  }
};

struct Odd
{
  template <size_t Offset, size_t N, typename Tuple, typename Descriptor>
  constexpr void clamp(LongUnderlyingType &x, Tuple &params, Descriptor& d, Result *r) const
  {
    x = x % 2 ? x : x + 1;
  }
};

template <int FFTIndex>
auto constexpr FrameSizeUpperLimit()
{
  return impl::FrameSizeUpperLimitImpl<FFTIndex>{};
}

} // namespace client
} // namespace fluid
