#pragma once

#include <clients/common/ParameterTypes.hpp>
#include <clients/common/ParameterConstraints.hpp>
#include <tuple>

namespace fluid {
  namespace client {
    
    template<typename...Args>
    auto constexpr MakeParams(const Args&&...args){ return std::make_tuple(std::forward<Args>(args)...); }
      
    template<typename Tuple>
    auto constexpr AddSTFTParams(const Tuple& params, const std::array<int,3> STFTDefaults = {1024, 512, -1})
    {
  
      return std::tuple_cat(params,std::make_tuple(
              LongParam("winSize", "Window Size", STFTDefaults[0], Min(4), FFTUpperLimit<std::tuple_size<Tuple>::value + 2>(), UpperLimit<std::tuple_size<Tuple>::value + 3>()),
              LongParam("hopSize", "Hop Size", STFTDefaults[1]),
              LongParam("fftSize", "FFT Size", STFTDefaults[2], LowerLimit<std::tuple_size<Tuple>::value>(), PowerOfTwo()),
              LongParam("maxWinSize", "Maxiumm Window Size", 16384)));
    }
  }
}
