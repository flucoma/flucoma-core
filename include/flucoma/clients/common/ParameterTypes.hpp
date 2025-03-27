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

#include "BufferAdaptor.hpp"
#include "ParameterTrackChanges.hpp"
#include "Result.hpp"
#include "../../data/FluidIndex.hpp"
#include "../../data/FluidMemory.hpp"
#include <memory>
#include <bitset> 
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include <set>

namespace fluid {
namespace client {

using FloatUnderlyingType = double;
using LongUnderlyingType = index; // signed int equal to pointer size, k thx
using EnumUnderlyingType = index;
using BufferUnderlyingType = std::shared_ptr<BufferAdaptor>;
using InputBufferUnderlyingType = std::shared_ptr<const BufferAdaptor>;
using StringUnderlyingType = rt::string;
using FloatArrayUnderlyingType = std::vector<FloatUnderlyingType>;
using LongArrayUnderlyingType = FluidTensor<LongUnderlyingType, 1>;
using BufferArrayUnderlyingType = std::vector<BufferUnderlyingType>;
using MagnitudePairsUnderlyingType = std::vector<std::pair<double, double>>;

template <bool b>
struct Fixed
{
  static bool constexpr value{b};
};

struct Primary {};

struct ParamTypeBase
{
  constexpr ParamTypeBase(const char* n, const char* display)
      : name(n), displayName(display)
  {}
  const char* name;
  const char* displayName;
};

struct FloatT : ParamTypeBase
{
  using type = FloatUnderlyingType;
  constexpr FloatT(const char* name, const char* displayName,
                   const type defaultVal)
      : ParamTypeBase(name, displayName), defaultValue(defaultVal)
  {}
  const index fixedSize = 1;
  const type  defaultValue;
};

struct LongT : ParamTypeBase
{
  using type = LongUnderlyingType;
  constexpr LongT(const char* name, const char* displayName,
                  const type defaultVal)
      : ParamTypeBase(name, displayName), defaultValue(defaultVal)
  {}
  const index fixedSize = 1;
  const type  defaultValue;
};


struct BufferT : ParamTypeBase
{
  using type = BufferUnderlyingType;
  constexpr BufferT(const char* name, const char* displayName)
      : ParamTypeBase(name, displayName)
  {}
  const index          fixedSize = 1;
  const std::nullptr_t defaultValue{nullptr};
}; // no non-relational conditions for buffer?

struct InputBufferT : ParamTypeBase
{
  using type = InputBufferUnderlyingType;
  //  using BufferT::BufferT;
  constexpr InputBufferT(const char* name, const char* displayName)
      : ParamTypeBase(name, displayName)
  {}
  const index          fixedSize = 1;
  const std::nullptr_t defaultValue{nullptr};
};

struct EnumT : ParamTypeBase
{
  using type = EnumUnderlyingType;
  template <index... N>
  constexpr EnumT(const char* name, const char* displayName, type defaultVal,
                  const char (&... string)[N])
      : ParamTypeBase(name, displayName), strings{string...}, fixedSize(1),
        numOptions(sizeof...(N)), defaultValue(defaultVal)
  {
    static_assert(sizeof...(N) > 0, "Fluid Param: No enum strings supplied!");
    static_assert(sizeof...(N) <= 16,
                  "Fluid Param: : Maximum 16 things in an Enum param");
  }
  const char* strings[16]; // unilateral descision klaxon: if you have more than
                           // 16 things in an Enum, you need to rethink
  const index fixedSize;
  const index numOptions;
  const type  defaultValue;

  struct EnumConstraint
  {
    template <size_t Offset, size_t N, typename Tuple, typename Descriptor>
    constexpr void clamp(EnumUnderlyingType& v, Tuple& /*allParams*/,
                         Descriptor&         d, Result*) const
    {
      auto& e = d.template get<N>();
      v = std::max<EnumUnderlyingType>(
          0, std::min<EnumUnderlyingType>(v, e.numOptions - 1));
    }
  };
};


struct ChoicesT: ParamTypeBase
{
  using type = std::bitset<16>; 
    
  template <index... N>
  constexpr ChoicesT(const char* name, const char* displayName,
                  const char (&... string)[N])
      : ParamTypeBase(name, displayName), strings{string...}, fixedSize(1),
        numOptions(sizeof...(N)), defaultValue((1 << numOptions) - 1)
  {
    static_assert(sizeof...(N) > 0, "Fluid Param: No choice strings supplied!");
    static_assert(sizeof...(N) <= 16,
                  "Fluid Param: : Maximum 16 things in an choice param");
  }
  const char* strings[16]; // unilateral decision klaxon: if you have more than
                           // 16 things in an Enum, you need to rethink
  const index fixedSize;
  const index numOptions;
  const type  defaultValue;

  index lookup(std::string name)
  {
    auto start = std::begin(strings);
    auto end = start + numOptions;
    auto pos = std::find(start,end, name);
    return pos != end? std::distance(start, pos) : -1;
  }
};

// can I avoid making this constexpr and thus using std::string? Let's see;
struct StringT : ParamTypeBase
{
  using type = StringUnderlyingType;
  constexpr StringT(const char* name, const char* displayName)
      : ParamTypeBase(name, displayName)
  {}
  const char* defaultValue = "";
  const index fixedSize = 1;
};

struct FloatArrayT : ParamTypeBase
{
  using type = FloatArrayUnderlyingType;
  template <std::size_t N>
  FloatArrayT(const char* name, const char* displayName,
              type::value_type (&/*defaultValues*/)[N])
      : ParamTypeBase(name, displayName)
  {}
  const index fixedSize;
};

struct LongArrayT : ParamTypeBase
{

  using type = LongArrayUnderlyingType;
  constexpr LongArrayT(
      const char* name, const char* displayName,
      const std::initializer_list<typename type::type> defaultValues)
      : ParamTypeBase(name, displayName), defaultValue(defaultValues)
  {}
  const index                                      fixedSize{1};
  const std::initializer_list<typename type::type> defaultValue;
};

struct BufferArrayT : ParamTypeBase
{

  using type = BufferArrayUnderlyingType;
  BufferArrayT(const char* name, const char* displayName, const index size)
      : ParamTypeBase(name, displayName), fixedSize(size)
  {}
  const index fixedSize;
};

// Pair of frequency amplitude pairs for HPSS threshold
struct FloatPairsArrayT : ParamTypeBase
{
  struct FloatPairsArrayType
  {

    constexpr FloatPairsArrayType(const double x0, const double y0,
                                  const double x1, const double y1)
        : value{{{x0, y0}, {x1, y1}}}
    {}

    constexpr FloatPairsArrayType(
        const std::array<std::pair<FloatUnderlyingType, FloatUnderlyingType>, 2>
            x)
        : value(x)
    {}

    FloatPairsArrayType(const FloatPairsArrayType& x) = default;
    FloatPairsArrayType& operator=(const FloatPairsArrayType&) = default;

    FloatPairsArrayType(FloatPairsArrayType&& x) noexcept
    {
      *this = std::move(x);
    }

    FloatPairsArrayType& operator=(FloatPairsArrayType&& x) noexcept
    {
      value = x.value;
      lowerChanged = x.lowerChanged;
      upperChanged = x.upperChanged;
      oldLower = x.oldLower;
      oldUpper = x.oldUpper;
      return *this;
    }

    std::array<std::pair<FloatUnderlyingType, FloatUnderlyingType>, 2> value;
    bool   lowerChanged{false};
    bool   upperChanged{false};
    double oldLower{0};
    double oldUpper{0};
  };

  //  static constexpr TypeTa
  using type = FloatPairsArrayType;

  constexpr FloatPairsArrayT(const char* name, const char* displayName)
      : ParamTypeBase(name, displayName)
  {}
  const index               fixedSize{4};
  const FloatPairsArrayType defaultValue{0.0, 1.0, 1.0, 1.0};
};

class FFTParams
{
public:
  constexpr FFTParams(intptr_t win, intptr_t hop, intptr_t fft, intptr_t max = -1)
      : mWindowSize{win}, mHopSize{hop}, mFFTSize{fft}, mMaxFFTSize{max},
        trackWin{win}, trackHop{hop}, trackFFT{fft}
  {}
  
  constexpr FFTParams(const FFTParams& p) noexcept = default;
  constexpr FFTParams(FFTParams&& p) noexcept = default;

  // Assignment shouldn't reset the trackers
  constexpr FFTParams& operator=(FFTParams&& p) noexcept
  {
    mWindowSize = p.mWindowSize;
    mHopSize = p.mHopSize;
    mFFTSize = p.mFFTSize;
    mMaxFFTSize = p.mMaxFFTSize;
    return *this;
  }

  constexpr FFTParams operator=(const FFTParams& p) noexcept
  {
    mWindowSize = p.mWindowSize;
    mHopSize = p.mHopSize;
    mFFTSize = p.mFFTSize;
    mMaxFFTSize = p.mMaxFFTSize;
    return *this;
  }

  index max() const noexcept
  {
    return mMaxFFTSize < 0 ? fftSize() : mMaxFFTSize;
  }

  index fftSize() const noexcept
  {
    assert(mWindowSize >= 0 &&
           asUnsigned(mWindowSize) <= std::numeric_limits<uint32_t>::max());
    return mFFTSize < 0 ? nextPow2(static_cast<uint32_t>(mWindowSize), true)
                        : mFFTSize;
  }
  
  intptr_t maxRaw() const noexcept { return mMaxFFTSize; }
  intptr_t fftRaw() const noexcept { return mFFTSize; }
  intptr_t hopRaw() const noexcept { return mHopSize; }
  intptr_t winSize() const noexcept { return mWindowSize; }
  intptr_t hopSize() const noexcept
  {
    return mHopSize > 0 ? mHopSize : mWindowSize >> 1;
  }
  
  intptr_t frameSize() const { return (fftSize() >> 1) + 1; }
  intptr_t maxFrameSize() const { return (max() >> 1) + 1; }
  
  static index padding(const FFTParams& settings, index option)
  {
    using Op = index (*)(const FFTParams&);
    static std::array<Op, 3> options{
        [](const FFTParams&) -> index { return 0; },
        [](const FFTParams& x) { return x.winSize() >> 1; },
        [](const FFTParams& x) { return x.winSize() - x.hopSize(); }};
    return options[asUnsigned(option)](settings);
  };

  void setWin(intptr_t win) { mWindowSize = win; }
  void setFFT(intptr_t fft) { mFFTSize = fft; }
  void setHop(intptr_t hop) { mHopSize = hop; }

  static index nextPow2(uint32_t x, bool up)
  {
    /// http://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2
    if (!x) return static_cast<index>(x);
    --x;
    x |= (x >> 1);
    x |= (x >> 2);
    x |= (x >> 4);
    x |= (x >> 8);
    x |= (x >> 16);
    return static_cast<index>(up ? ++x : x - (x >> 1));
  }

  bool operator==(const FFTParams& x)
  {
    return mWindowSize == x.mWindowSize && mHopSize == x.mHopSize &&
           mFFTSize == x.mFFTSize;
  }

  bool operator!=(const FFTParams& x) { return !(*this == x); }

  struct FFTSettingsConstraint
  {
    template <index Offset, size_t N, typename Tuple, typename Descriptor>
    constexpr void clamp(FFTParams& v, Tuple& allParams, Descriptor&,
                         Result* r) const
    {
      FFTParams input = v;
      auto&     inParams = std::get<N>(allParams).get();

      bool winChanged = inParams.trackWin.changed(v.winSize());
      bool fftChanged = inParams.trackFFT.changed(v.fftRaw());
      bool hopChanged = inParams.trackHop.changed(v.hopRaw());

      if (winChanged) v.setWin(std::max<intptr_t>(v.winSize(), 4));
      if (fftChanged && v.fftRaw() >= 0)
        v.setFFT(std::max<intptr_t>(v.fftRaw(), 4));

      if (winChanged && !fftChanged)
        v.setWin(v.fftRaw() < 0 ? v.winSize()
                                : std::min<intptr_t>(v.winSize(), v.fftRaw()));

      if (fftChanged)
      {
        if (v.fftRaw() < 0)
          v.setFFT(-1);
        else
        {
          // This is all about making drag behaviour in GUI elements sensible
          // If we drag down we want it to leap down by powers of 2, but with a
          // lower bound at th nearest power of 2 >= winSize
          bool  up = inParams.trackFFT.template direction<0>() > 0;
          index fft = v.fftRaw();
          assert(fft <= asSigned(std::numeric_limits<uint32_t>::max()));
          fft = (fft & (fft - 1)) == 0
                    ? fft
                    : v.nextPow2(static_cast<uint32_t>(v.fftRaw()), up);
          assert(v.winSize() >= 0 &&
                 v.winSize() <= asSigned(std::numeric_limits<uint32_t>::max()));
          fft = std::max<index>(
              fft, v.nextPow2(static_cast<uint32_t>(v.winSize()), true));
          v.setFFT(fft);

        }
      }

      if (hopChanged) v.setHop(v.hopRaw() <= 0 ? -1 : v.hopRaw());

      if (v.mMaxFFTSize > 0)
        v.mMaxFFTSize = static_cast<index>(
            v.nextPow2(static_cast<uint32_t>(v.mMaxFFTSize), true));
      else
        v.mMaxFFTSize = v.nextPow2(static_cast<uint32_t>(v.fftSize()), true);

      index clippedFFT = v.mMaxFFTSize > 0 ? std::min(v.fftSize(),v.mMaxFFTSize) : v.fftSize();

      bool fftSizeWasClipped{clippedFFT != v.fftSize()};
      if (fftSizeWasClipped)
      {
        v.setWin(std::min<intptr_t>(v.winSize(), clippedFFT));
        v.setFFT(v.fftRaw() < 0 ? v.fftRaw() : clippedFFT);
      }

      inParams.trackWin.changed(v.winSize());
      inParams.trackFFT.changed(v.fftRaw());
      inParams.trackHop.changed(v.hopRaw());

      if (v != input && r)
      {
        r->set(Result::Status::kWarning);
        if (v.winSize() != input.winSize())
          r->addMessage("Window size constrained to ", v.winSize());
        if (v.fftRaw() != input.fftRaw())
          r->addMessage("FFT size adjusted to ", v.fftRaw());
        if (fftSizeWasClipped)
          r->addMessage("FFT and / or window clipped to maximum (", clippedFFT,
                        ")");
      }
    }
  };

private:
  intptr_t mWindowSize;
  intptr_t mHopSize;
  intptr_t mFFTSize;
  intptr_t mMaxFFTSize;

  ParameterTrackChanges<intptr_t> trackWin;
  ParameterTrackChanges<intptr_t> trackHop;
  ParameterTrackChanges<intptr_t> trackFFT;
};

struct FFTParamsT : ParamTypeBase
{
  using type = FFTParams;

  constexpr FFTParamsT(const char* name, const char* displayName,
                       index winDefault, index hopDefault, index fftDefault)
      : ParamTypeBase(name, displayName), defaultValue{winDefault, hopDefault,
                                                       fftDefault,-1}
  {}

  const index fixedSize = 4;
  const type  defaultValue;
};

class LongRuntimeMaxParam {
public:
  
  constexpr LongRuntimeMaxParam(index val, index max)
    : mValue(val),
      mMax(max), mInitialValue{mValue}
  {}

  constexpr LongRuntimeMaxParam(index val): LongRuntimeMaxParam(val,-1){}
  //todo do I need this?
  LongRuntimeMaxParam(std::reference_wrapper<index> val): LongRuntimeMaxParam(val.get(),-1){}

  index operator()() const  { return mValue; }
  operator fluid::index() const {return mValue; }
  index max() const { return mMax < 0 ? mInitialValue : mMax; }
  index maxRaw() const { return mMax; }
  
  void set(index val)
  {
    mValue = val;
  }
  
  void clamp() { mValue = std::min(mValue,max());   }
  
  struct RuntimeMaxConstraint
  {
    template <index Offset, size_t N, typename Tuple, typename Descriptor>
    constexpr void clamp(LongRuntimeMaxParam& v, Tuple&, Descriptor& d,
                         Result* r) const
    {
       index oldValue = v;
       v.clamp();
       if(r && oldValue != v)
       {
          r->set(Result::Status::kWarning);
          r->addMessage(d.template get<N>().name, " value, ", oldValue,
                    ", above user defined maximum ", v.max());
       }
    }
  };
  
  friend bool operator<(const LongRuntimeMaxParam& l, const LongRuntimeMaxParam& r)
  {
    return l() < r();
  }
  
  friend bool operator>(const LongRuntimeMaxParam& l, const LongRuntimeMaxParam& r)
  {
    return r() < l();
  }
  
  friend bool operator<=(const LongRuntimeMaxParam& l, const LongRuntimeMaxParam& r)
  {
    return !(l() > r());
  }
  
  friend bool operator>=(const LongRuntimeMaxParam& l, const LongRuntimeMaxParam& r)
  {
    return !(l() < r());
  }
    
private:
  index mValue;
  index mMax;
  index mInitialValue;
};

struct LongRuntimeMaxT: ParamTypeBase
{
  using type = LongRuntimeMaxParam;
  constexpr LongRuntimeMaxT(const char* name, const char* displayName,
                           index defaultValue)
       : ParamTypeBase(name, displayName), defaultValue(defaultValue,-1)
  {}
  const index fixedSize = 2;
  const type  defaultValue;
};

template <typename T, typename Fixed, typename... Constraints>
using ParamSpec = std::tuple<T, std::tuple<Constraints...>, Fixed>;

template <typename IsFixed = Fixed<false>, typename... Constraints>
constexpr ParamSpec<FloatT, IsFixed, Constraints...>
FloatParam(const char* name, const char* displayName,
           const FloatT::type defaultValue, Constraints... c)
{
  return {FloatT(name, displayName, defaultValue), std::make_tuple(c...),
          IsFixed{}};
}

template <typename IsFixed = Fixed<false>, typename... Constraints>
constexpr ParamSpec<LongT, IsFixed, Constraints...>
LongParam(const char* name, const char* displayName,
          const LongT::type defaultValue, Constraints... c)
{
  return {LongT(name, displayName, defaultValue), std::make_tuple(c...),
          IsFixed{}};
}

template <typename IsFixed = Fixed<false>, typename... Constraints>
constexpr ParamSpec<BufferT, IsFixed, Constraints...>
BufferParam(const char* name, const char* displayName, const Constraints... c)
{
  return {BufferT(name, displayName), std::make_tuple(c...), IsFixed{}};
}

template <typename IsFixed = Fixed<false>, typename... Constraints>
constexpr ParamSpec<InputBufferT, IsFixed, Constraints...>
InputBufferParam(const char* name, const char* displayName,
                 const Constraints... c)
{
  return {InputBufferT(name, displayName), std::make_tuple(c...), IsFixed{}};
}

template <typename IsFixed = Fixed<false>, size_t... N>
constexpr ParamSpec<EnumT, IsFixed, EnumT::EnumConstraint>
EnumParam(const char* name, const char* displayName,
          const EnumT::type defaultVal, const char (&... strings)[N])
{
  return {EnumT(name, displayName, defaultVal, strings...),
          std::make_tuple(EnumT::EnumConstraint()), IsFixed{}};
}

template <typename IsFixed = Fixed<false>, size_t... N>
constexpr ParamSpec<ChoicesT, IsFixed>
ChoicesParam(const char* name, const char* displayName, const char (&... strings)[N])
{
  return {ChoicesT(name, displayName, strings...), std::make_tuple(),
          Fixed<false>{}};
}


template <typename IsFixed = Fixed<false>, size_t N, typename... Constraints>
constexpr ParamSpec<FloatArrayT, IsFixed, Constraints...>
FloatArrayParam(const char* name, const char* displayName,
                FloatArrayT::type::value_type (&defaultValues)[N],
                Constraints... c)
{
  return {FloatArrayT(name, displayName, defaultValues), std::make_tuple(c...),
          IsFixed{}};
}

template <typename IsFixed = Fixed<false>, typename... Constraints>
constexpr ParamSpec<LongArrayT, IsFixed, Constraints...> LongArrayParam(
    const char* name, const char* displayName,
    const std::initializer_list<LongArrayT::type::type> defaultValues,
    const Constraints... c)
{
  return {LongArrayT(name, displayName, defaultValues), std::make_tuple(c...),
          IsFixed{}};
}

template <typename IsFixed = Fixed<false>, typename... Constraints>
constexpr ParamSpec<BufferArrayT, IsFixed, Constraints...>
BufferArrayParam(const char* name, const char* displayName,
                 const Constraints... c)
{
  return {BufferArrayT(name, displayName, 0), std::make_tuple(c...), IsFixed{}};
}

template <typename IsFixed = Fixed<false>, typename... Constraints>
constexpr ParamSpec<FloatPairsArrayT, IsFixed, Constraints...>
FloatPairsArrayParam(const char* name, const char* displayName,
                     const Constraints... c)
{
  return {FloatPairsArrayT(name, displayName), std::make_tuple(c...),
          IsFixed{}};
}

template <typename... Constraints>
constexpr ParamSpec<FFTParamsT, Fixed<false>,
                    FFTParams::FFTSettingsConstraint,
                    Constraints...>
FFTParam(const char* name, const char* displayName, index winDefault,
         index hopDefault, index fftDefault, const Constraints... c)
{
  return {FFTParamsT(name, displayName, winDefault, hopDefault, fftDefault),
          std::tuple_cat(
              std::make_tuple(FFTParams::FFTSettingsConstraint()),
              std::make_tuple(c...)),
          Fixed<false>{}};
}

template<typename IsPrimary = Fixed<false>, typename...Constraints>
constexpr ParamSpec<LongRuntimeMaxT,IsPrimary,LongRuntimeMaxParam::RuntimeMaxConstraint,Constraints...>
LongParamRuntimeMax(const char* name, const char* displayName, index defaultValue,  const Constraints&&...c)
{
  return { LongRuntimeMaxT(name, displayName,defaultValue),
            std::make_tuple(LongRuntimeMaxParam::RuntimeMaxConstraint(), std::forward<const Constraints>(c)...),
            IsPrimary()};
}

template <typename IsFixed = Fixed<false>, typename... Constraints>
constexpr ParamSpec<StringT, IsFixed, Constraints...>
StringParam(const char* name, const char* displayName, const Constraints... c)
{
  return {StringT(name, displayName), std::make_tuple(c...), IsFixed{}};
}

namespace impl {
template <typename T>
struct ParamLiterals
{
  using type = typename T::type;
  static std::array<type, 1> getLiteral(const type& p) { return {{p}}; }
};

template <>
struct ParamLiterals<FloatPairsArrayT>
{
  using type = FloatUnderlyingType;

  static std::array<type, 4> getLiteral(const FloatPairsArrayT::type& p)
  {
    auto v = p.value;
    return {{v[0].first, v[0].second, v[1].first, v[1].second}};
  }
};

template <>
struct ParamLiterals<FFTParamsT>
{
  using type = LongUnderlyingType;

  static std::array<type, 4> getLiteral(const FFTParams& p)
  {
    return {{p.winSize(), p.hopRaw(), p.fftRaw()}};
  }
};

template <>
struct ParamLiterals<LongRuntimeMaxT>
{
   using type = index;
   
   static std::array<index,2> getLiteral(const LongRuntimeMaxParam& p)
   {
      return {{p(),p.max()}};
   }
};

} // namespace impl

template <typename T, size_t N>
class ParamLiteralConvertor
{

public:
  using ValueType = typename T::type;
  using LiteralType = typename impl::ParamLiterals<T>::type;
  using ArrayType = std::array<LiteralType, N>;

  void set(const ValueType& v)
  {
    mArray = impl::ParamLiterals<T>::getLiteral(v);
  }
  ValueType    value() { return make(std::make_index_sequence<N>()); }
  LiteralType& operator[](index idx) { return mArray[asUnsigned(idx)]; }

private:
  template <size_t... Is>
  ValueType make(std::index_sequence<Is...>)
  {
    return {mArray[Is]...};
  }

  ArrayType mArray;
};

template <typename T>
std::ostream& operator<<(std::ostream& o, const std::unique_ptr<T>& p)
{
  return o << p.get();
}

template <typename T, typename U>
std::ostream& operator<<(std::ostream& o, const std::unique_ptr<T, U>& p)
{
  return o << p.get();
}

} // namespace client
} // namespace fluid
