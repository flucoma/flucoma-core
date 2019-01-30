#pragma once

#include <tuple>

namespace fluid
{
namespace client
{
namespace impl
{
  template<size_t WinParamIndex,size_t HopParamIndex, size_t FFTParamIndex,typename Client>
  std::tuple<size_t, size_t, size_t> deriveSTFTParams(Client& x)
  {
      size_t winSize = x.template get<WinParamIndex>();
      size_t hopSize = x.template changed<HopParamIndex>() ? x.template get<HopParamIndex>()
                                                      : winSize / 2;
      size_t fftSize =
          x.template get<FFTParamIndex>() != -1 ? x.template get<FFTParamIndex>() : winSize;
    
    return {winSize,hopSize,fftSize};
  }
}
}
}

