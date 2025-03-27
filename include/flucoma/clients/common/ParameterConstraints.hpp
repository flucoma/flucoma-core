/*
Part of the Fluid Corpus Manipulation Project (http://www.flucoma.org/)
Copyright University of Huddersfield.
Licensed under the BSD-3 License.
See license.md file in the project root for full license information.
This project has received funding from the European Research Council (ERC)
under the European Union’s Horizon 2020 research and innovation programme
(grant agreement No 725899).
*/

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

struct Relational
{};

template <typename T>
struct MinImpl
{
  constexpr MinImpl(const T m) : value(m) {}
  const T value;
  template <size_t Offset, size_t N, typename U, typename Tuple,
            typename Descriptor>
  constexpr void clamp(U& x, Tuple& /*params*/, Descriptor& d, Result* r) const
  {
    U oldX = x;
    x = std::max<U>(x, static_cast<U>(value));
    if (r && oldX != x)
    {
      r->set(Result::Status::kWarning);
      r->addMessage(d.template get<N>().name, " value, ", oldX,
                    ", below absolute minimum ", x);
    }
  }
  
  template <size_t Offset, size_t N, typename Tuple,
            typename Descriptor>
  constexpr void clamp(LongRuntimeMaxParam& x, Tuple& /*params*/, Descriptor& d, Result* r) const
  {
    index oldX = x();
    index newx = std::max<index>(x(), value);
    x.set(newx); 
    if (r && oldX != x)
    {
      r->set(Result::Status::kWarning);
      r->addMessage(d.template get<N>().name, " value, ", oldX,
                    ", below absolute minimum ", x);
    }
  }
};

template <typename T>
struct MaxImpl
{
  constexpr MaxImpl(const T m) : value(m) {}
  const T value;
  template <size_t Offset, size_t N, typename U, typename Tuple,
            typename Descriptor>
  constexpr void clamp(U& x, Tuple& /*params*/, Descriptor& d, Result* r) const
  {

    U oldX = x;
    x = std::min<U>(x, value);
    if (r && oldX != x)
    {
      r->set(Result::Status::kWarning);
      r->addMessage(d.template get<N>().name, " value (", oldX,
                    ") above absolute maximum (", x, ')');
    }
  }
};

template <int... Is>
struct LowerLimitImpl : public Relational
{
  template <size_t Offset, size_t N, typename T, typename Tuple,
            typename Descriptor>
  void clamp(T& v, Tuple& params, Descriptor& d, Result* r) const
  {
    T oldV = v;

    v = std::max<T>({v, std::get<Is + Offset>(params).get()...});

    if (r && oldV != v)
    {
      r->set(Result::Status::kWarning);
      std::array<T, sizeof...(Is)> constraintValues{
          {std::get<Is + Offset>(params).get()...}};
      index minPos = std::distance(
          constraintValues.begin(),
          std::min_element(constraintValues.begin(), constraintValues.end()));
      std::array<std::string_view, sizeof...(Is)> constraintNames{
          {d.template get<Is + Offset>().name...}};
      r->addMessage(d.template get<N>().name, " value (", oldV,
                    ") below parameter ", constraintNames[asUnsigned(minPos)],
                    " (", v, ')');
    }
  }
};

template <int... Is>
struct UpperLimitImpl : public Relational
{
  template <size_t Offset, size_t N, typename T, typename Tuple,
            typename Descriptor>
  void clamp(T& v, Tuple& params, Descriptor& d, Result* r) const
  {
    T oldV = v;

    v = std::min<T>({v, std::get<Is + Offset>(params)...});

    if (r && oldV != v)
    {
      r->set(Result::Status::kWarning);
      std::array<T, sizeof...(Is)> constraintValues{
          {std::get<Is + Offset>(params)...}};
      index maxPos = std::distance(
          constraintValues.begin(),
          std::max_element(constraintValues.begin(), constraintValues.end()));
      std::array<std::string_view, sizeof...(Is)> constraintNames{
          {d.template get<Is + Offset>().name...}};
      r->addMessage(d.template get<N>().name, " value, ", oldV,
                    ", above parameter ", constraintNames[asUnsigned(maxPos)],
                    " (", v, ')');
    }
  }
};

template <int FFTIndex>
struct FrameSizeUpperLimitImpl : public Relational
{
  template <size_t Offset, size_t N, typename T, typename Tuple,
            typename Descriptor>
  void clamp(T& v, Tuple& params, Descriptor& d, Result* r) const
  {
    T     oldV = v;
    index frameSize = std::get<FFTIndex + Offset>(params).get().frameSize();
    v = std::min<T>(v, frameSize);

    if (r && oldV != v)
    {
      r->set(Result::Status::kWarning);
      r->addMessage(d.template get<N>().name, " value (", oldV,
                    ") above spectral frame size (", v, ')');
    }
  }
};

template <int MaxFFTSizeIndex>
struct MaxFrameSizeUpperLimitImpl : public Relational
{
  template <size_t Offset, size_t N, typename T, typename Tuple,
            typename Descriptor>
  void clamp(T& v, Tuple& params, Descriptor& d, Result* r) const
  {
    T     oldV = v;
    index frameSize = (std::get<MaxFFTSizeIndex + Offset>(params) + 1) / 2;
    v = std::min<T>(v, frameSize);

    if (r && oldV != v)
    {
      r->set(Result::Status::kWarning);
      r->addMessage(d.template get<N>().name, " value (", oldV,
                    ") above maximum spectral frame size (", v, ')');
    }
  }
};

template <int Lower>
struct FrameSizeLowerLimitImpl : public Relational
{
  template <size_t Offset, size_t N, typename Tuple, typename Descriptor>
  void clamp(FFTParams& v, Tuple& params, Descriptor& d, Result* r) const
  {
    FFTParams oldV = v;
    index     frameSize = v.frameSize();
    frameSize = std::max<index>(std::get<Lower + Offset>(params), frameSize);
    assert((frameSize - 1) >= 0 &&
           (frameSize - 1) <= asSigned(std::numeric_limits<uint32_t>::max()));
    intptr_t newsize =
        2 * FFTParams::nextPow2(static_cast<uint32_t>(frameSize - 1), true);
    if (v.fftRaw() == -1)
      v.setWin(newsize);
    else
      v.setFFT(newsize);

    if (r && oldV != v)
    {
      r->set(Result::Status::kWarning);
      r->addMessage(d.template get<N>().name, " value (", oldV.frameSize(),
                    ") below minimum spectral frame size (", v.frameSize(),
                    ')');
    }
  }
};


} // namespace impl

template <typename T>
auto constexpr Min(const T x)
{
  return impl::MinImpl<T>(x);
}

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
  constexpr void clamp(type& v, Tuple& allParams, Descriptor&, Result*) const
  {
    auto& vals = v.value;
    auto& inParams = std::get<N>(allParams).get();
    // For now I know that array size is 2, just upper and lower vals
    // TODO: make generic for any old monotonic array of freq-amp pairs, should
    // we need it

    // Clip freqs to [0,1]
    vals[0].first = std::max<double>(std::min<double>(vals[0].first, 1), 0);
    vals[1].first = std::max<double>(std::min<double>(vals[1].first, 1), 0);

    inParams.lowerChanged = vals[0].first != inParams.oldLower;
    inParams.upperChanged = vals[1].first != inParams.oldUpper;

    if (v.lowerChanged && !v.upperChanged && vals[0].first > vals[1].first)
      vals[0].first = vals[1].first;
    if (v.upperChanged && !v.lowerChanged && vals[0].first > vals[1].first)
      vals[1].first = vals[0].first;
    // If everything changed (i.e. object creation) and in the wrong order, just
    // swap 'em
    if (v.lowerChanged && v.upperChanged && vals[0].first > vals[1].first)
      std::swap(vals[0], vals[1]);

    inParams.oldLower = vals[0].first;
    inParams.oldUpper = vals[1].first;
  }
};

struct PowerOfTwo
{
  template <size_t Offset, size_t N, typename Tuple, typename Descriptor>
  constexpr void clamp(LongUnderlyingType& x, Tuple& /*params*/, Descriptor& d,
                       Result* r) const
  {

    int                exp = 0;
    double             base = std::frexp(x, &exp);
    LongUnderlyingType res = base > 0.5 ? (1 << exp) : (1 << (exp - 1));

    if (r && res != x)
    {
      r->set(Result::Status::kWarning);
      r->addMessage(d.template get<N>().name, " value (", x,
                    ") adjusted to power of two (", res, ')');
    }
    x = res;
  }
};

struct Odd
{
  template <size_t Offset, size_t N, typename Tuple, typename Descriptor>
  constexpr void clamp(LongUnderlyingType& x, Tuple& /*params*/, Descriptor& d,
                       Result* r) const
  {
    LongUnderlyingType val = x % 2 ? x : x + 1;
    if (r && val != x)
    {
      r->set(Result::Status::kWarning);
      r->addMessage(d.template get<N>().name, " value (", x,
                    ") adjusted to next odd number (", val, ')');
    }
    x = val;
  }

  template <size_t Offset, size_t N, typename Tuple, typename Descriptor>
  constexpr void clamp(LongRuntimeMaxParam& x, Tuple& /*params*/, Descriptor& d,
                       Result* r) const
  {
    index val = x();
    val = val % 2 ? val : val + 1;
    if (r && val != x())
    {
      r->set(Result::Status::kWarning);
      r->addMessage(d.template get<N>().name, " value (", x(),
                    ") adjusted to next odd number (", val, ')');
    }
    x.set(val);
  }
};

template <int FFTIndex>
auto constexpr FrameSizeUpperLimit()
{
  return impl::FrameSizeUpperLimitImpl<FFTIndex>{};
}

template <int MaxFFTIndex>
auto constexpr MaxFrameSizeUpperLimit()
{
  return impl::MaxFrameSizeUpperLimitImpl<MaxFFTIndex>{};
}

template <int Lower>
auto constexpr FrameSizeLowerLimit()
{
  return impl::FrameSizeLowerLimitImpl<Lower>{};
}

template <typename Constraint>
struct CanIncreaseValue : std::false_type
{};

template <>
struct CanIncreaseValue<Odd> : std::true_type
{};

template <>
struct CanIncreaseValue<PowerOfTwo> : std::true_type
{};

template <typename... Ts>
constexpr auto GetIncreasingConstraints(const std::tuple<Ts...> constraints)
{
  return std::apply(
      [](Ts...) {
        return std::tuple_cat(
            std::conditional_t<CanIncreaseValue<Ts>::value, std::tuple<Ts>,
                               std::tuple<>>{}...);
      },
      constraints);
}

} // namespace client
} // namespace fluid
