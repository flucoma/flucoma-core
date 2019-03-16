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
#include <clients/common/ParameterSet.hpp>
#include <data/FluidTensor.hpp>
#include <data/TensorTypes.hpp>

namespace fluid {
namespace client {

enum STFTParamIndex { kFFT, kMaxFFT };

auto constexpr STFTParams = defineParameters(
    FFTParam<kMaxFFT>("fft","FFT Settings", 1024, -1, -1),
    LongParam<Fixed<true>>("maxFFTSize", "Maxiumm FFT Size", 16384));

template <typename T>
class BaseSTFTClient : public FluidBaseClient<decltype(STFTParams), STFTParams>, public AudioIn, public AudioOut
{
  using HostVector = HostVector<T>;

public:

  BaseSTFTClient(ParamSetInitType& p) : FluidBaseClient(p), mSTFTBufferedProcess{get<kMaxFFT>(),1,1}
  {
    FluidBaseClient::audioChannelsIn(1);
    FluidBaseClient::audioChannelsOut(1);
  }

  size_t latency() { return get<kFFT>().winSize(); }

  void process(std::vector<HostVector> &input,
               std::vector<HostVector> &output) {

    if (!input[0].data() || !output[0].data())
      return;
    // Here we do an STFT and its inverse
    mSTFTBufferedProcess.process(mParams, input, output,
        [](ComplexMatrixView in, ComplexMatrixView out) { out = in; });
  }

private:
  STFTBufferedProcess<ParamSetInitType, T, kFFT, true> mSTFTBufferedProcess;
};
} // namespace client
} // namespace fluid
