/*
@file: BaseSTFTClient

Test class for STFT pass-through
*/
#pragma once

#include "BufferedProcess.hpp"
#include <algorithms/public/STFT.hpp>
#include <clients/common/FluidBaseClient.hpp>
#include <clients/common/ParameterConstraints.hpp>
#include <clients/common/ParameterTypes.hpp>
#include <data/FluidTensor.hpp>
#include <data/TensorTypes.hpp>

#include <complex>
#include <string>
#include <tuple>
#include <vector>

namespace fluid {
namespace client {

enum STFTParamIndex { kWinsize, kHopsize, kFFTSize, kMaxWin };

auto constexpr STFTParams = std::make_tuple(
    LongParam("winSize", "Window Size", 1024, Min(4)),
    LongParam("hopSize", "Hop Size", 512),
    LongParam("fftSize", "FFT Size", -1, LowerLimit<kWinsize>(), PowerOfTwo()),
    LongParam("maxWinSize", "Maxiumm Window Size", 16384));

using Param_t = decltype(STFTParams);

template <typename T, typename U = T>
class BaseSTFTClient : public FluidBaseClient<Param_t> {

  using HostVector = HostVector<U>;

public:
  BaseSTFTClient(BaseSTFTClient &) = delete;
  BaseSTFTClient operator=(BaseSTFTClient &) = delete;

  BaseSTFTClient() : FluidBaseClient<Param_t>(STFTParams) {
    audioChannelsIn(1);
    audioChannelsOut(1);
  }

  void process(std::vector<HostVector> &input,
               std::vector<HostVector> &output) {

    if (!input[0].data() || !output[0].data())
      return;
    // Here we do an STFT and its inverse
    mSTFTBufferedProcess.process(
        *this, input, output,
        [](ComplexVector in, ComplexVector out) { out = in; });
  }

private:
  STFTBufferedProcess<T,U, BaseSTFTClient, kMaxWin, kWinsize, kHopsize,
                      kFFTSize, true>
      mSTFTBufferedProcess;
};
} // namespace client
} // namespace fluid

