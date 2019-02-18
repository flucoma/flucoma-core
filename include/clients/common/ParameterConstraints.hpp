#pragma once

#include "ParameterTypes.hpp"
#include "Result.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
namespace fluid {
namespace client {


namespace impl {

auto makeOdd = [](auto a) { return [=] { return a % 2 ? a - 1 : a; }; };

template <typename T> struct MinImpl {
  constexpr MinImpl(const T m) : value(m) {}
  const T value;
  template <size_t N, typename U, typename Tuple> constexpr void clamp(U &x, Tuple& params, Result* r) {
    U oldX = x;
    x = std::max<U>(x, value);
    if(r && oldX != x)
    {
      r->set(Result::Status::kWarning);
      r->addMessage(std::get<N>(params).first.name());
      r->addMessage(" value, "); r->addMessage(oldX);r->addMessage(", below absolute minimum ");r->addMessage(x);
    }
  }
};

template <typename T> struct MaxImpl {
  constexpr MaxImpl(const T m) : value(m) {}
  const T value;
  template <size_t N, typename U, typename Tuple> constexpr void clamp(U &x, Tuple& params, Result* r) {
    
    U oldX = x;
    x = std::min<U>(x, value);
    if(r && oldX != x)
    {
      r->set(Result::Status::kWarning);
      r->addMessage(std::get<N>(params).first.name());
      r->addMessage(" value ("); r->addMessage(oldX);r->addMessage(") above absolute maximum (");r->addMessage(x);r->addMessage(')');
    }
  }
};

template <int... Is> struct LowerLimitImpl {
  template <size_t N, typename T, typename Tuple> void clamp(T &v, Tuple& params, Result* r) {
    
    T oldV = v;
    
    v = std::max<T>({v, std::get<Is>(params).first.get()...});
    
    if(r && oldV != v)
    {
      r->set(Result::Status::kWarning);
      std::array<T,sizeof...(Is)> constraintValues {std::get<Is>(params).first.get()...};
      size_t minPos = std::distance(constraintValues.begin(), std::min_element(constraintValues.begin(), constraintValues.end()));
      std::array<const char*,sizeof...(Is)> constraintNames {std::get<Is>(params).first.name()...};
      r->addMessage(std::get<N>(params).first.name());
      r->addMessage(" value ("); r->addMessage(oldV);r->addMessage(") below parameter ");r->addMessage(constraintNames[minPos]);
      r->addMessage(" ("); r->addMessage(v);r->addMessage(')');
    }
  }
};

template <int... Is> struct UpperLimitImpl {
  template <size_t N, typename T, typename Tuple> void clamp(T &v, Tuple& params, Result* r) {
    
    T oldV = v;
    
    v = std::min<T>({v, std::get<Is>(params).first.get()...});
    
    if(r && oldV != v)
    {
      r->set(Result::Status::kWarning);
      std::array<T,sizeof...(Is)> constraintValues {std::get<Is>(params).first.get()...};
      size_t maxPos = std::distance(constraintValues.begin(), std::max_element(constraintValues.begin(), constraintValues.end()));
      std::array<const char*,sizeof...(Is)> constraintNames  {std::get<Is>(params).first.name()...};
      r->addMessage(std::get<N>(params).first.name());
      r->addMessage(" value, "); r->addMessage(oldV);r->addMessage(", above parameter ");r->addMessage(constraintNames[maxPos]);
      r->addMessage(" ("); r->addMessage(v);r->addMessage(')');
    }
  }
};

template <int WinIndex, int FFTIndex>
struct FrameSizeUpperLimitImpl
{
  template<size_t N, typename T, typename Tuple> void clamp(T& v, Tuple& params, Result* r)
  {
    T oldV = v;
    size_t fftSize = std::get<FFTIndex>(params).first.get();
    fftSize = fftSize == -1 ? std::get<WinIndex>(params).first.get() : fftSize;
    v = std::min<T>(v, fftSize / 2 + 1);
    
    if(r && oldV != v)
    {
      r->set(Result::Status::kWarning);
      r->addMessage(std::get<N>(params).first.name());
      r->addMessage(" value ("); r->addMessage(oldV);r->addMessage(") above spectral frame size (");r->addMessage(v);r->addMessage(')');
    }
  }
};


template<int WinSizeIndex>
struct WinLowerLimitImpl{
  template<size_t N, typename T, typename Tuple> void clamp(T& FFTSize, Tuple& params, Result* r)
  {
    size_t oldFFTSize = FFTSize;
    size_t winSize = std::get<WinSizeIndex>(params).first.get();
    FFTSize = FFTSize == -1 ? FFTSize : std::max<size_t>(winSize,FFTSize);
    if(r && oldFFTSize != FFTSize)
    {
      r->set(Result::Status::kWarning);
      r->addMessage(std::get<N>(params).first.name());
      r->addMessage(" value ("); r->addMessage(oldFFTSize);r->addMessage(") below window size (");r->addMessage(winSize);r->addMessage(')');
    }
  }
};

template<int FFTIndex>
struct FFTUpperLimitImpl{
  template<size_t N, typename T, typename Tuple> void clamp(T& winSize, Tuple& params, Result* r)
  {
    size_t oldWinSize = winSize;
    size_t fftSize = std::get<FFTIndex>(params).first.get();
    winSize = fftSize == -1 ? winSize : std::min<size_t>(winSize,fftSize);
    if(r && oldWinSize != winSize)
    {
      r->set(Result::Status::kWarning);
      r->addMessage(std::get<N>(params).first.name());
      r->addMessage(" value ("); r->addMessage(oldWinSize);r->addMessage(") above fft size size (");r->addMessage(winSize);r->addMessage(')');
    }
  }
};


} // namespace impl

template <typename T> auto constexpr Min(const T x) {
  return impl::MinImpl<T>(x);
};

template <typename T> auto constexpr Max(const T x) {
  return impl::MaxImpl<T>(x);
}

template <int... Is> auto constexpr LowerLimit() {
  return impl::LowerLimitImpl<Is...>{};
}

template <int... Is> auto constexpr UpperLimit() {
  return impl::UpperLimitImpl<Is...>{};
}


struct FrequencyAmpPairConstraint
{
  using type = typename FloatPairsArrayT::type;
  
  constexpr FrequencyAmpPairConstraint(){}
  
  template <size_t N, typename Tuple> constexpr void clamp(type &v, Tuple&, Result* r)
  {
    //For now I know that array size is 2, just upper and lower vals
    //TODO: make generic for any old monotonic array of freq-amp pairs, should we need it
    
    //Clip freqs to [0,1]
    v[0].first = std::max<double>(std::min<double>(v[0].first,1),0);
    v[1].first = std::max<double>(std::min<double>(v[1].first,1),0); 
    
    lowerChanged = v[0].first == oldLower;
    upperChanged = v[1].first == oldUpper;
    
    if(lowerChanged && !upperChanged && v[0].first > v[1].first) v[0].first = v[1].first;
    if(upperChanged && !lowerChanged && v[0].first > v[1].first) v[1].first = v[0].first;
    //If everything changed (i.e. object creation) and in the wrong order, just swap 'em
    if(lowerChanged && upperChanged && v[0].first > v[1].first) std::swap(v[0],v[1]);
    
    oldLower = v[0].first;
    oldUpper = v[1].first;
    
    
  }
private:
  bool lowerChanged {false};
  bool upperChanged {false};
  double oldLower{0};
  double oldUpper{0};
};

struct PowerOfTwo {
  template <size_t N, typename Tuple> constexpr LongUnderlyingType clamp(LongUnderlyingType x, Tuple& params, Result* r) {
    
    int exp = 0;
    double base = std::frexp(x, &exp);
    LongUnderlyingType res =  base > 0.5 ? (1 << exp) : (1 << (exp - 1));
    
    if(r && res != x)
    {
      r->set(Result::Status::kWarning);
      r->addMessage(std::get<N>(params).first.name());
      r->addMessage(" value ("); r->addMessage(x);r->addMessage(") adjusted to power of two (");r->addMessage(res);r->addMessage(')');
    }
    return res;
  }

};

struct Odd {
  template <size_t N, typename Tuple> constexpr LongUnderlyingType clamp(LongUnderlyingType x, Tuple& params, Result* r) {
    return x % 2 ? x : x + 1; 
  }
};

template <int WinIndex, int FFTIndex> auto constexpr FrameSizeUpperLimit()
{
  return impl::FrameSizeUpperLimitImpl<WinIndex,FFTIndex>{};
}

template<int FFTIndex> auto constexpr FFTUpperLimit()
{
  return impl::FFTUpperLimitImpl<FFTIndex>{}; 
}

template<int WinSizeIndex> auto constexpr WinLowerLimit()
{
  return impl::WinLowerLimitImpl<WinSizeIndex>{};
}


} // namespace client
} // namespace fluid

