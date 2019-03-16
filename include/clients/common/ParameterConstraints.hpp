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
      r->addMessage(std::get<N>(params).name());
      r->addMessage(" value, ");
      r->addMessage(oldX);
      r->addMessage(", below absolute minimum ");
      r->addMessage(x);
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
      r->addMessage(std::get<N>(params).name());
      r->addMessage(" value (");
      r->addMessage(oldX);
      r->addMessage(") above absolute maximum (");
      r->addMessage(x);
      r->addMessage(')');
    }
  }
};

template <int... Is>
struct LowerLimitImpl
{
  template <size_t Offset, size_t N, typename T, typename Tuple, typename Descriptor>
  void clamp(T &v, Tuple &params, Descriptor& d, Result *r) const
  {
    T oldV = v;

    v = std::max<T>({v, std::get<Is + Offset>(params).get()...});

    if (r && oldV != v)
    {
      r->set(Result::Status::kWarning);
      std::array<T, sizeof...(Is)> constraintValues{std::get<Is + Offset>(params).get()...};
      size_t                       minPos =
          std::distance(constraintValues.begin(), std::min_element(constraintValues.begin(), constraintValues.end()));
      std::array<const char *, sizeof...(Is)> constraintNames{std::get<Is + Offset>(params).name()...};
      r->addMessage(std::get<N>(params).name());
      r->addMessage(" value (");
      r->addMessage(oldV);
      r->addMessage(") below parameter ");
      r->addMessage(constraintNames[minPos]);
      r->addMessage(" (");
      r->addMessage(v);
      r->addMessage(')');
    }
  }
};

template <int... Is>
struct UpperLimitImpl
{
  template <size_t Offset, size_t N, typename T, typename Tuple, typename Descriptor>
  void clamp(T &v, Tuple &params, Descriptor& d, Result *r) const
  {
    T oldV = v;

    v = std::min<T>({v, std::get<Is + Offset>(params).get()...});

    if (r && oldV != v)
    {
      r->set(Result::Status::kWarning);
      std::array<T, sizeof...(Is)> constraintValues{std::get<Is + Offset>(params).get()...};
      size_t                       maxPos =
          std::distance(constraintValues.begin(), std::max_element(constraintValues.begin(), constraintValues.end()));
      std::array<const char *, sizeof...(Is)> constraintNames{std::get<Is + Offset>(params).name()...};
      r->addMessage(std::get<N>(params).name());
      r->addMessage(" value, ");
      r->addMessage(oldV);
      r->addMessage(", above parameter ");
      r->addMessage(constraintNames[maxPos]);
      r->addMessage(" (");
      r->addMessage(v);
      r->addMessage(')');
    }
  }
};

template <int FFTIndex>
struct FrameSizeUpperLimitImpl
{
  template <size_t Offset, size_t N, typename T, typename Tuple, typename Descriptor>
  void clamp(T &v, Tuple &params, Descriptor& d, Result *r) const
  {
    T      oldV      = v;
    size_t frameSize = std::get<FFTIndex + Offset>(params).get().frameSize();
    v                = std::min<T>(v, frameSize);

    if (r && oldV != v)
    {
      r->set(Result::Status::kWarning);
      r->addMessage(std::get<N>(params).name(), " value (", oldV, ") above spectral frame size (", v, ')');
    }
  }
};

template <int WinSizeIndex>
struct WinLowerLimitImpl
{
  template <size_t Offset, size_t N, typename T, typename Tuple, typename Descriptor>
  void clamp(T &FFTSize, Tuple &params, Descriptor& d, Result *r) const
  {
    size_t oldFFTSize = FFTSize;
    size_t winSize    = std::get<WinSizeIndex + Offset>(params).get();
    FFTSize           = FFTSize == -1 ? FFTSize : std::max<size_t>(winSize, FFTSize);
    if (r && oldFFTSize != FFTSize)
    {
      r->set(Result::Status::kWarning);
      r->addMessage(std::get<N>(params).name());
      r->addMessage(" value (");
      r->addMessage(oldFFTSize);
      r->addMessage(") below window size (");
      r->addMessage(winSize);
      r->addMessage(')');
    }
  }
};

template <int FFTIndex>
struct FFTUpperLimitImpl
{
  template <size_t Offset, size_t N, typename T, typename Tuple, typename Descriptor>
  void clamp(T &winSize, Tuple &params, Descriptor& d, Result *r) const
  {
    size_t oldWinSize = winSize;
    size_t fftSize    = std::get<FFTIndex + Offset>(params).get();
    winSize           = fftSize == -1 ? winSize : std::min<size_t>(winSize, fftSize);
    if (r && oldWinSize != winSize)
    {
      r->set(Result::Status::kWarning);
      r->addMessage(std::get<N>(params).name());
      r->addMessage(" value (");
      r->addMessage(oldWinSize);
      r->addMessage(") above fft size size (");
      r->addMessage(winSize);
      r->addMessage(')');
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
  constexpr void clamp(type &v, Tuple &, Descriptor &, Result *r) const
  {
    auto& vals = v.value;
    // For now I know that array size is 2, just upper and lower vals
    // TODO: make generic for any old monotonic array of freq-amp pairs, should we need it

    // Clip freqs to [0,1]
    vals[0].first = std::max<double>(std::min<double>(vals[0].first, 1), 0);
    vals[1].first = std::max<double>(std::min<double>(vals[1].first, 1), 0);

    v.lowerChanged = vals[0].first != v.oldLower;
    v.upperChanged = vals[1].first != v.oldUpper;

    if (v.lowerChanged && !v.upperChanged && vals[0].first > vals[1].first) vals[0].first = vals[1].first;
    if (v.upperChanged && !v.lowerChanged && vals[0].first > vals[1].first) vals[1].first = vals[0].first;
    // If everything changed (i.e. object creation) and in the wrong order, just swap 'em
    if (v.lowerChanged && v.upperChanged && vals[0].first > vals[1].first) std::swap(vals[0], vals[1]);

    v.oldLower = vals[0].first;
    v.oldUpper = vals[1].first;
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
      r->addMessage(std::get<N>(params).name());
      r->addMessage(" value (");
      r->addMessage(x);
      r->addMessage(") adjusted to power of two (");
      r->addMessage(res);
      r->addMessage(')');
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

template <int FFTIndex>
auto constexpr FFTUpperLimit()
{
  return impl::FFTUpperLimitImpl<FFTIndex>{};
}

template <int WinSizeIndex>
auto constexpr WinLowerLimit()
{
  return impl::WinLowerLimitImpl<WinSizeIndex>{};
}

} // namespace client
} // namespace fluid
