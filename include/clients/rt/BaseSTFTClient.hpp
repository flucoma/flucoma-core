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

template <typename Params, typename T, typename U = T>
class BaseSTFTClient : public FluidBaseClient<Params>, public AudioIn, public AudioOut
{

  using HostVector = HostVector<U>;
  using B =  FluidBaseClient<Params>;

  Params& mParams;

public:

  BaseSTFTClient(Params& p) : FluidBaseClient<Params>(p), mParams(p), mSTFTBufferedProcess{param<kMaxFFT>(p),1,1}
  {
    B::audioChannelsIn(1);
    B::audioChannelsOut(1);
  }

  size_t latency() { return param<kFFT>(mParams).winSize(); }

  void process(std::vector<HostVector> &input,
               std::vector<HostVector> &output) {

    if (!input[0].data() || !output[0].data())
      return;
    // Here we do an STFT and its inverse
    mSTFTBufferedProcess.process(mParams, input, output,
        [](ComplexMatrixView in, ComplexMatrixView out) { out = in; });
  }

private:
  STFTBufferedProcess<Params,  U, kFFT, true> mSTFTBufferedProcess;
};
} // namespace client
} // namespace fluid
