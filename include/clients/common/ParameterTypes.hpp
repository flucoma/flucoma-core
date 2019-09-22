#pragma once

#include "BufferAdaptor.hpp"
#include "ParameterTrackChanges.hpp"
#include "Result.hpp"

#include <memory>
#include <tuple>
#include <utility>
#include <vector>


namespace fluid {
namespace client {

using FloatUnderlyingType          = double;
using LongUnderlyingType           = intptr_t; // signed int equal to pointer size, k thx
using EnumUnderlyingType           = intptr_t;
using BufferUnderlyingType         = std::shared_ptr<BufferAdaptor>;
using InputBufferUnderlyingType    = std::shared_ptr<const BufferAdaptor>;
using FloatArrayUnderlyingType     = std::vector<FloatUnderlyingType>;
using LongArrayUnderlyingType      = std::vector<LongUnderlyingType>;
using BufferArrayUnderlyingType    = std::vector<BufferUnderlyingType>;
using MagnitudePairsUnderlyingType = std::vector<std::pair<double, double>>;

template <bool b>
struct Fixed
{
  static bool constexpr value{b};
};

struct ParamTypeBase
{
  constexpr ParamTypeBase(const char *n, const char *display)
      : name(n)
      , displayName(display)
  {}
  const char *name;
  const char *displayName;
};

struct FloatT : ParamTypeBase
{
  using type  = FloatUnderlyingType;
  constexpr FloatT(const char *name, const char *displayName, const type defaultVal)
      : ParamTypeBase(name, displayName)
      , defaultValue(defaultVal)
  {}
  const std::size_t fixedSize = 1;
  const type        defaultValue;
};

struct LongT : ParamTypeBase
{
  using type  = LongUnderlyingType;
  constexpr LongT(const char *name, const char *displayName, const type defaultVal)
      : ParamTypeBase(name, displayName)
      , defaultValue(defaultVal)
  {}
  const std::size_t fixedSize = 1;
  const type        defaultValue;
};

struct BufferT : ParamTypeBase
{
  using type  = BufferUnderlyingType;
  constexpr BufferT(const char *name, const char *displayName)
      : ParamTypeBase(name, displayName)
  {}
  const std::size_t    fixedSize = 1;
  const std::nullptr_t defaultValue{nullptr};
}; // no non-relational conditions for buffer?

struct InputBufferT:ParamTypeBase
{
  using type = InputBufferUnderlyingType;
  //  using BufferT::BufferT;
  constexpr InputBufferT(const char *name, const char *displayName)
  : ParamTypeBase(name, displayName)
  {}
  const std::size_t    fixedSize = 1;
  const std::nullptr_t defaultValue{nullptr};
};

struct EnumT : ParamTypeBase
{
  using type  = EnumUnderlyingType;
  template <std::size_t... N>
  constexpr EnumT(const char *name, const char *displayName, type defaultVal, const char (&... string)[N])
      : strings{string...}
      , ParamTypeBase(name, displayName)
      , fixedSize(1)
      , numOptions(sizeof...(N))
      , defaultValue(defaultVal)
  {
    static_assert(sizeof...(N) > 0, "Fluid Param: No enum strings supplied!");
    static_assert(sizeof...(N) <= 16, "Fluid Param: : Maximum 16 things in an Enum param");
  }
  const char *strings[16]; // unilateral descision klaxon: if you have more than 16 things in an Enum, you need to rethink
  const std::size_t fixedSize;
  const std::size_t numOptions;
  const type        defaultValue;
  
  struct EnumConstraint
  {
    template <size_t Offset, size_t N, typename Tuple, typename Descriptor>
    constexpr void clamp(EnumUnderlyingType &v, Tuple &allParams, Descriptor &d, Result *r) const
    {
      auto& e = d.template get<N>();
      v = std::max<EnumUnderlyingType>(0,std::min<EnumUnderlyingType>(v, e.numOptions - 1));
    }
  };
};

struct FloatArrayT : ParamTypeBase
{
  using type  = FloatArrayUnderlyingType;
  template <std::size_t N>
  FloatArrayT(const char *name, const char *displayName, type::value_type (&defaultValues)[N])
      : ParamTypeBase(name, displayName)
  {}
  const std::size_t fixedSize;
};

struct LongArrayT : ParamTypeBase
{
  
  using type  = LongArrayUnderlyingType;
  template <std::size_t N>
  LongArrayT(const char *name, const char *displayName, type::value_type (&defaultValues)[N])
      : ParamTypeBase(name, displayName)
  {}
  const std::size_t fixedSize;
};

struct BufferArrayT : ParamTypeBase
{
  
  using type  = BufferArrayUnderlyingType;
  BufferArrayT(const char *name, const char *displayName, const size_t size)
      : ParamTypeBase(name, displayName)
      , fixedSize(size)
  {}
  const std::size_t fixedSize;
};

// Pair of frequency amplitude pairs for HPSS threshold
struct FloatPairsArrayT : ParamTypeBase
{
  struct FloatPairsArrayType
  {
  
    constexpr FloatPairsArrayType(const double x0, const double y0, const double x1, const double y1): value{{{x0,y0},{x1,y1}}}
    {}
  
    constexpr FloatPairsArrayType(const std::array<std::pair<FloatUnderlyingType, FloatUnderlyingType>,2> x): value{x}
    {}
  
    FloatPairsArrayType(const FloatPairsArrayType& x) = default;
    FloatPairsArrayType& operator=(const FloatPairsArrayType&)=default;
    
    FloatPairsArrayType(FloatPairsArrayType&& x) noexcept { *this = std::move(x); }

    FloatPairsArrayType& operator=(FloatPairsArrayType&& x) noexcept
    {
      value = x.value;
      lowerChanged = x.lowerChanged;
      upperChanged = x.upperChanged;
      oldLower = x.oldLower;
      oldUpper = x.oldUpper;
      return *this;
    }
  
    std::array<std::pair<FloatUnderlyingType, FloatUnderlyingType>,2> value;
    bool   lowerChanged{false};
    bool   upperChanged{false};
    double oldLower{0};
    double oldUpper{0};
  };
    
  //  static constexpr TypeTa
  using type = FloatPairsArrayType;

  constexpr FloatPairsArrayT(const char *name, const char *displayName)
      : ParamTypeBase(name, displayName)
  {}
  const std::size_t         fixedSize{4};
  const FloatPairsArrayType defaultValue{0.0, 1.0, 1.0, 1.0};
};

// My name's the C++ linker, and I'm a bit of a knob (fixed in C++17)
//constexpr std::initializer_list<std::pair<double, double>> FloatPairsArrayT::defaultValue;

template <bool>
struct ConstrainMaxFFTSize;

template <>
struct ConstrainMaxFFTSize<false>
{
  template <intptr_t N, typename T>
  size_t clamp(intptr_t x, T &constraints) const
  {
    return x;
  }
};

template <>
struct ConstrainMaxFFTSize<true>
{
  template <intptr_t N, typename T>
  size_t clamp(intptr_t x, T &constraints) const
  {
    return std::min<intptr_t>(x, std::get<N>(constraints));
  }
};

class FFTParams
{
public:
  constexpr FFTParams(intptr_t win, intptr_t hop, intptr_t fft)
    : mWindowSize{win}
    , mHopSize{hop}
    , mFFTSize{fft}
    , trackWin{win}
    , trackHop{hop}
    , trackFFT{fft}
  {}

  constexpr FFTParams(const FFTParams& p) = default;
  constexpr FFTParams(FFTParams&& p) noexcept = default;
    
  // Assignment should not change the trackers
    
  FFTParams& operator = (FFTParams&& p) noexcept
  {
    mWindowSize = p.mWindowSize;
    mHopSize = p.mHopSize;
    mFFTSize = p.mFFTSize;
    
    return *this;
  }
    
  FFTParams& operator = (const FFTParams& p)
  {
    mWindowSize = p.mWindowSize;
    mHopSize = p.mHopSize;
    mFFTSize = p.mFFTSize;
    
    return *this;
  }
    
  size_t fftSize() const noexcept { return mFFTSize < 0 ? nextPow2(mWindowSize, true) : mFFTSize; }
  intptr_t   fftRaw() const noexcept { return mFFTSize; }
  intptr_t   hopRaw() const noexcept { return mHopSize; }
  intptr_t winSize() const noexcept { return mWindowSize; }
  intptr_t hopSize() const noexcept { return mHopSize > 0 ? mHopSize : mWindowSize >> 1; }
  intptr_t frameSize() const { return (fftSize() >> 1) + 1; }

  void setWin(intptr_t win) { mWindowSize = win; }
  void setFFT(intptr_t fft) { mFFTSize = fft; }
  void setHop(intptr_t hop) { mHopSize = hop; }

  static intptr_t nextPow2(uint32_t x, bool up) 
  {
    /// http://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2
    if (!x) return x;
    --x;
    x |= (x >> 1);
    x |= (x >> 2);
    x |= (x >> 4);
    x |= (x >> 8);
    x |= (x >> 16);
    return up ? ++x : x - (x >> 1);
  }

  bool operator==(const FFTParams &x)
  {
    return mWindowSize == x.mWindowSize && mHopSize == x.mHopSize && mFFTSize == x.mFFTSize;
  }

  bool operator!=(const FFTParams &x) { return !(*this == x); }

  template <int MaxFFTIndex = -1>
  struct FFTSettingsConstraint
  {
    template <size_t Offset, size_t N, typename Tuple, typename Descriptor>
    constexpr void clamp(FFTParams &v, Tuple &allParams, Descriptor &d, Result *r) const 
    {
      FFTParams input = v;
      auto& inParams = std::get<N>(allParams);
        
      bool winChanged = inParams.trackWin.changed(v.winSize());
      bool fftChanged = inParams.trackFFT.changed(v.fftRaw());
      bool hopChanged = inParams.trackHop.changed(v.hopRaw());

      if (winChanged) v.setWin(std::max<intptr_t>(v.winSize(), 4));
      if (fftChanged && v.fftRaw() >= 0) v.setFFT(std::max<intptr_t>(v.fftRaw(), 4));

      if (winChanged && !fftChanged) v.setWin(v.fftRaw() < 0 ? v.winSize() : std::min<intptr_t>(v.winSize(), v.fftRaw()));

      if (fftChanged)
      {
        if (v.fftRaw() < 0)
          v.setFFT(-1);
        else
        {
          // This is all about making drag behaviour in GUI elements sensible
          // If we drag down we want it to leap down by powers of 2, but with a lower bound
          // at th nearest power of 2 >= winSize
          bool up = inParams.trackFFT.template direction<0>() > 0;
          int fft = v.fftRaw();
          fft = (fft & (fft - 1)) == 0 ? fft : v.nextPow2(v.fftRaw(), up);
          fft = std::max<int>(fft, v.nextPow2(v.winSize(), true));
          v.setFFT(fft);
//          v.setFFT(std::max(v.fftRaw(), v.nextPow2(v.winSize(), true)));
        }
      }

      if (hopChanged) v.setHop(v.hopRaw() <= 0 ? -1 : v.hopRaw());

      //      //If both have changed at once (e.g. startup), then we need to prioritse something
      //      if(winChanged && fftChanged && v.fftRaw() > 0)
      //          v.setFFT(v.fftRaw() < 0 ? -1 : v.nextPow2(std::max<intptr_t>(v.winSize(), inParams.fftRaw()),trackFFT.template
      //          direction<0>() > 0));
      //
      constexpr bool HasMaxFFT = MaxFFTIndex > 0;
      constexpr intptr_t I     = MaxFFTIndex + Offset;

      // Now check (optionally) against MaxFFTSize
      size_t clippedFFT = std::max<intptr_t>(ConstrainMaxFFTSize<HasMaxFFT>{}.template clamp<I, Tuple>(v.fftSize(), allParams),4);
            
      bool   fftSizeWasClipped{clippedFFT != v.fftSize()};
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
        if (v.winSize() != input.winSize()) r->addMessage("Window size constrained to ", v.winSize());
        if (v.fftRaw() != input.fftRaw()) r->addMessage("FFT size adjusted to ", v.fftRaw());
        if (fftSizeWasClipped) r->addMessage("FFT and / or window clipped to maximum (", clippedFFT, ")");
      }
    }
  };

private:
  intptr_t mWindowSize;
  intptr_t mHopSize;
  intptr_t mFFTSize;
  bool mHopChanged{false};

  ParameterTrackChanges<intptr_t> trackWin;
  ParameterTrackChanges<intptr_t> trackHop;
  ParameterTrackChanges<intptr_t> trackFFT;
};

struct FFTParamsT : ParamTypeBase
{
  using type = FFTParams;

  constexpr FFTParamsT(const char *name, const char *displayName, int winDefault, int hopDefault, int fftDefault)
      : ParamTypeBase(name, displayName)
      , defaultValue{winDefault, hopDefault, fftDefault}
  {}

  const std::size_t fixedSize = 3;
  const type        defaultValue;
};

template <typename T, typename Fixed, typename... Constraints>
using ParamSpec = std::tuple<T, std::tuple<Constraints...>, Fixed>;

template <typename IsFixed = Fixed<false>, typename... Constraints>
constexpr ParamSpec<FloatT, IsFixed, Constraints...> FloatParam(const char *name, const char *displayName,
                                                                const FloatT::type defaultValue, Constraints... c)
{
  return {FloatT(name, displayName, defaultValue), std::make_tuple(c...), IsFixed{}};
}

template <typename IsFixed = Fixed<false>, typename... Constraints>
constexpr ParamSpec<LongT, IsFixed, Constraints...> LongParam(const char *name, const char *displayName,
                                                              const LongT::type defaultValue, Constraints... c)
{
  return {LongT(name, displayName, defaultValue), std::make_tuple(c...), IsFixed{}};
}

template <typename IsFixed = Fixed<false>, typename... Constraints>
constexpr ParamSpec<BufferT, IsFixed, Constraints...> BufferParam(const char *name, const char *displayName,
                                                                  const Constraints... c)
{
  return {BufferT(name, displayName), std::make_tuple(c...), IsFixed{}};
}

template <typename IsFixed = Fixed<false>, typename... Constraints>
constexpr ParamSpec<InputBufferT, IsFixed, Constraints...> InputBufferParam(const char *name, const char *displayName,
                                                                  const Constraints... c)
{
  return {InputBufferT(name, displayName), std::make_tuple(c...), IsFixed{}};
}

template <typename IsFixed = Fixed<false>, size_t... N >
constexpr ParamSpec<EnumT, IsFixed,EnumT::EnumConstraint> EnumParam(const char *name, const char *displayName, const EnumT::type defaultVal,
                                              const char (&... strings)[N])
{
  return {EnumT(name, displayName, defaultVal, strings...), std::make_tuple(EnumT::EnumConstraint()), IsFixed{}};
}

template <typename IsFixed = Fixed<false>, size_t N, typename... Constraints>
constexpr ParamSpec<FloatArrayT, IsFixed, Constraints...> FloatArrayParam(const char *name, const char *displayName,
                                                                          FloatArrayT::type::value_type (&defaultValues)[N],
                                                                          Constraints... c)
{
  return {FloatArrayT(name, displayName, defaultValues), std::make_tuple(c...), IsFixed{}};
}

template <typename IsFixed = Fixed<false>, size_t N, typename... Constraints>
constexpr ParamSpec<LongArrayT, IsFixed, Constraints...> LongArrayParam(const char *name, const char *displayName,
                                                                        LongArrayT::type::value_type (&defaultValues)[N],
                                                                        const Constraints... c)
{
  return {LongArrayT(name, displayName, defaultValues), std::make_tuple(c...), IsFixed{}};
}

template <typename IsFixed = Fixed<false>, typename... Constraints>
constexpr ParamSpec<BufferArrayT, IsFixed, Constraints...> BufferArrayParam(const char *name, const char *displayName,
                                                                            const Constraints... c)
{
  return {BufferArrayT(name, displayName, 0), std::make_tuple(c...), IsFixed{}};
}

template <typename IsFixed = Fixed<false>, typename... Constraints>
constexpr ParamSpec<FloatPairsArrayT, IsFixed, Constraints...>
FloatPairsArrayParam(const char *name, const char *displayName, const Constraints... c)
{
  return {FloatPairsArrayT(name, displayName), std::make_tuple(c...), IsFixed{}};
}

template <intptr_t MaxFFTIndex = -1, typename... Constraints>
constexpr ParamSpec<FFTParamsT, Fixed<false>, FFTParams::FFTSettingsConstraint<MaxFFTIndex>, Constraints...>
FFTParam(const char *name, const char *displayName, int winDefault, int hopDefault, int fftDefault, const Constraints... c)
{
  return {FFTParamsT(name, displayName, winDefault, hopDefault, fftDefault),
          std::tuple_cat(std::make_tuple(FFTParams::FFTSettingsConstraint<MaxFFTIndex>()),
                         std::make_tuple(c...)),
          Fixed<false>{}};
}

namespace impl
{
  template<typename T>
  struct ParamLiterals
  {
    using type = typename T::type;
    static std::array<type, 1> getLiteral(const type& p) { return {p}; }
  };
  
  template<>
  struct ParamLiterals<FloatPairsArrayT>
  {
    using type = FloatUnderlyingType;
    
    static std::array<type, 4> getLiteral(const FloatPairsArrayT::type& p)
    {
        auto v = p.value;
        return { v[0].first, v[0].second, v[1].first, v[1].second };
    }
  };
  
  template<>
  struct ParamLiterals<FFTParamsT>
  {
    using type = LongUnderlyingType;
    
    static std::array<type, 3> getLiteral(const FFTParams& p)
    {
      return { p.winSize(), p.hopRaw(), p.fftRaw()};
    }
  };
}

template <typename T, size_t N>
class ParamLiteralConvertor
{
    
public:
    
  using ValueType = typename T::type;
  using LiteralType = typename impl::ParamLiterals<T>::type;
  using ArrayType = std::array<LiteralType, N>;
    
  void set(const ValueType &v) { mArray = impl::ParamLiterals<T>::getLiteral(v); }
  ValueType value() { return make(std::make_index_sequence<N>()); }
  LiteralType& operator[](size_t idx) { return mArray[idx]; }
    
private:
    
  template <size_t... Is>
  ValueType make(std::index_sequence<Is...>) { return {mArray[Is]...}; }
    
  ArrayType mArray;
};

template <typename T>
std::ostream &operator<<(std::ostream &o, const std::unique_ptr<T> &p)
{
  return o << p.get();
}

template <typename T, typename U>
std::ostream &operator<<(std::ostream &o, const std::unique_ptr<T, U> &p)
{
  return o << p.get();
}

} // namespace client
} // namespace fluid
