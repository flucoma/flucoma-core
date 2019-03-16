#pragma once

#include <clients/common/BufferAdaptor.hpp>
#include <clients/common/ParameterTrackChanges.hpp>
#include <clients/common/Result.hpp>
#include <tuple>
#include <utility>
#include <vector>
namespace fluid {
namespace client {

enum class TypeTag { kFloat, kLong, kBuffer, kEnum, kFloatArray, kLongArray, kBufferArray };

using FloatUnderlyingType          = double;
using LongUnderlyingType           = intptr_t; // signed int equal to pointer size, k thx
using EnumUnderlyingType           = intptr_t;
using BufferUnderlyingType         = std::unique_ptr<BufferAdaptor>;
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
  static constexpr TypeTag typeTag = TypeTag::kFloat;
  using type                       = FloatUnderlyingType;
  constexpr FloatT(const char *name, const char *displayName, const type defaultVal)
      : ParamTypeBase(name, displayName)
      , defaultValue(defaultVal)
  {}
  const std::size_t fixedSize = 1;
  const type        defaultValue;
};

struct LongT : ParamTypeBase
{
  static constexpr TypeTag typeTag = TypeTag::kLong;
  using type                       = LongUnderlyingType;
  constexpr LongT(const char *name, const char *displayName, const type defaultVal)
      : ParamTypeBase(name, displayName)
      , defaultValue(defaultVal)
  {}
  const std::size_t fixedSize = 1;
  const type        defaultValue;
};

struct BufferT : ParamTypeBase
{
  static constexpr TypeTag typeTag = TypeTag::kBuffer;
  using type                       = BufferUnderlyingType;
  constexpr BufferT(const char *name, const char *displayName)
      : ParamTypeBase(name, displayName)
  {}
  const std::size_t    fixedSize = 1;
  const std::nullptr_t defaultValue{nullptr};
}; // no non-relational conditions for buffer?

struct EnumT : ParamTypeBase
{
  static constexpr TypeTag typeTag = TypeTag::kEnum;
  using type                       = EnumUnderlyingType;
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
};

struct FloatArrayT : ParamTypeBase
{
  static constexpr TypeTag typeTag = TypeTag::kFloatArray;
  using type                       = FloatArrayUnderlyingType;

  template <std::size_t N>
  FloatArrayT(const char *name, const char *displayName, type::value_type (&defaultValues)[N])
      : ParamTypeBase(name, displayName)
  {}
  const std::size_t fixedSize;
};

struct LongArrayT : ParamTypeBase
{
  static constexpr TypeTag typeTag = TypeTag::kLongArray;
  using type                       = LongArrayUnderlyingType;
  template <std::size_t N>
  LongArrayT(const char *name, const char *displayName, type::value_type (&defaultValues)[N])
      : ParamTypeBase(name, displayName)
  {}
  const std::size_t fixedSize;
};

struct BufferArrayT : ParamTypeBase
{
  static constexpr TypeTag typeTag = TypeTag::kBufferArray;
  using type                       = BufferArrayUnderlyingType;
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
    
    constexpr FloatPairsArrayType(FloatPairsArrayType&& x) { *this = std::move(x); }

    FloatPairsArrayType& operator=(FloatPairsArrayType&& x)
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
  template <long N, typename T>
  size_t clamp(long x, T &constraints) const
  {
    return x;
  }
};

template <>
struct ConstrainMaxFFTSize<true>
{
  template <long N, typename T>
  size_t clamp(long x, T &constraints) const
  {
    return std::min<long>(x, std::get<N>(constraints).get());
  }
};

class FFTParams
{
public:
  constexpr FFTParams(long win, long hop, long fft)
    : mWindowSize{win}
    , mHopSize{hop}
    , mFFTSize{fft}
    , trackWin{win}
    , trackHop{hop}
    , trackFFT{fft}
  {}

  constexpr FFTParams(const FFTParams& p) = default;
  constexpr FFTParams(FFTParams&& p) = default;
    
  // Assignment should not change the trackers
    
  FFTParams& operator = (FFTParams&& p)
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
  long   fftRaw() const noexcept { return mFFTSize; }
  long   hopRaw() const noexcept { return mHopSize; }
  size_t winSize() const noexcept { return mWindowSize; }
  size_t hopSize() { return mHopSize > 0 ? mHopSize : mWindowSize >> 1; }
  size_t frameSize() const { return (fftSize() >> 1) + 1; }

  void setWin(long win) { mWindowSize = win; }
  void setFFT(long fft) { mFFTSize = fft; }
  void setHop(long hop) { mHopSize = hop; }

  long nextPow2(u_int32_t x, bool up) const
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
    template <size_t Offset, size_t N, typename Tuple>
    constexpr void clamp(FFTParams &v, Tuple &allParams, Result *r) const
    {
      FFTParams input = v;

      bool winChanged = v.trackWin.changed(v.winSize());
      bool fftChanged = v.trackFFT.changed(v.fftRaw());
      bool hopChanged = v.trackHop.changed(v.hopRaw());

      if (winChanged) v.setWin(std::max(v.winSize(), 4ul));

      if (winChanged && !fftChanged) v.setWin(v.fftRaw() < 0 ? v.winSize() : std::min<size_t>(v.winSize(), v.fftRaw()));

      if (fftChanged)
      {
        if (v.fftRaw() < 0)
          v.setFFT(-1);
        else
        {
          // This is all about making drag behaviour in GUI elements sensible
          // If we drag down we want it to leap down by powers of 2, but with a lower bound
          // at th nearest power of 2 >= winSize
          bool up = v.trackFFT.template direction<0>() > 0;
          v.setFFT(v.nextPow2(v.fftRaw(), up));
          v.setFFT(std::max(v.fftRaw(), v.nextPow2(v.winSize(), true)));
        }
      }

      if (hopChanged) v.setHop(v.hopRaw() <= 0 ? -1 : v.hopRaw());

      //      //If both have changed at once (e.g. startup), then we need to prioritse something
      //      if(winChanged && fftChanged && v.fftRaw() > 0)
      //          v.setFFT(v.fftRaw() < 0 ? -1 : v.nextPow2(std::max<long>(v.winSize(), v.fftRaw()),trackFFT.template
      //          direction<0>() > 0));
      //
      constexpr bool HasMaxFFT = MaxFFTIndex > 0;
      constexpr long I         = MaxFFTIndex + Offset;

      // Now check (optionally) against MaxFFTSize
      size_t clippedFFT = ConstrainMaxFFTSize<HasMaxFFT>{}.template clamp<I, Tuple>(v.fftSize(), allParams);
      bool   fftSizeWasClipped{clippedFFT != v.fftSize()};
      if (fftSizeWasClipped)
      {
        v.setWin(std::min(v.winSize(), clippedFFT));
        v.setFFT(v.fftRaw() < 0 ? v.fftRaw() : clippedFFT);
      }

      v.trackWin.changed(v.winSize());
      v.trackFFT.changed(v.fftRaw());
      v.trackHop.changed(v.hopRaw());

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
  long mWindowSize;
  long mHopSize;
  long mFFTSize;
  bool mHopChanged{false};

  ParameterTrackChanges<int> trackWin;
  ParameterTrackChanges<int> trackHop;
  ParameterTrackChanges<int> trackFFT;
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

template <typename IsFixed = Fixed<false>, size_t... N>
constexpr ParamSpec<EnumT, IsFixed> EnumParam(const char *name, const char *displayName, const EnumT::type defaultVal,
                                              const char (&... strings)[N])
{
  return {EnumT(name, displayName, defaultVal, strings...), std::make_tuple(), IsFixed{}};
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

template <long MaxFFTIndex = -1, typename... Constraints>
constexpr ParamSpec<FFTParamsT, Fixed<false>, FFTParams::FFTSettingsConstraint<MaxFFTIndex>, Constraints...>
FFTParam(const char *name, const char *displayName, int winDefault, int hopDefault, int fftDefault, const Constraints... c)
{
  return {FFTParamsT(name, displayName, winDefault, hopDefault, fftDefault),
          std::tuple_cat(std::make_tuple(FFTParams::FFTSettingsConstraint<MaxFFTIndex>()),
                         std::make_tuple(c...)),
          Fixed<false>{}};
}

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

template <typename T>
class ParameterValue
{
public:
  using ParameterType = T;
  using type          = typename T::type;

  ParameterValue(const T descriptor, type &&v)
      : mDescriptor(descriptor)
      , mValue(std::move(v))
  {}

  ParameterValue(const T descriptor)
      : mDescriptor(descriptor)
      , mValue(mDescriptor.defaultValue)
  {}

  ParameterValue(ParameterValue &&) = default;
  ParameterValue &operator=(ParameterValue &&) = default;

  bool        enabled() const noexcept { return true; }
  bool        changed() const noexcept { return mChanged; }
  const char *name() const noexcept { return mDescriptor.name; }
  const T     descriptor() const noexcept { return mDescriptor; }

  type &get() noexcept { return mValue; }

  void set(type &&value)
  {
    mValue   = std::move(value);
    mChanged = true;
  }

  void reset()
  {
    mValue   = mDescriptor.defaultValue;
    mChanged = true;
  }

private:
  const T mDescriptor;

protected:
  bool mChanged{false};
  type mValue;
};

} // namespace client
} // namespace fluid
