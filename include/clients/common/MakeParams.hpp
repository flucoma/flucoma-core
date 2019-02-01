#pragma once

#include <clients/common/ParameterTypes.hpp>
#include <clients/common/ParameterConstraints.hpp>
#include <tuple>

namespace fluid {
  namespace client {
    
    template<typename...Args>
    auto constexpr MakeParams(const Args&&...args){ return std::make_tuple(std::forward<Args>(args)...); }
      
    template<int WinDefault = 1024, int HopDefault = 512, int FFTDefault = -1,typename Tuple>
    auto constexpr AddSTFTParams(const Tuple& params)
    {
  
      auto constexpr stftParams = std::make_tuple(
              LongParam("winSize", "Window Size", WinDefault, Min(4), FFTUpperLimit<std::tuple_size<Tuple>::value + 2>(), UpperLimit<std::tuple_size<Tuple>::value + 3>()),
              LongParam("hopSize", "Hop Size", HopDefault),
              LongParam("fftSize", "FFT Size", FFTDefault, LowerLimit<std::tuple_size<Tuple>::value>(), PowerOfTwo()),
              LongParam<Fixed<true>>("maxWinSize", "Maxiumm Window Size", 16384));
  
      return std::tuple_cat(params, stftParams);
    }
  }
}
