/*
@file: BaseSTFTClient

Test class for STFT pass-through
*/
#pragma once

#include "BufferedProcess.hpp"
#include "../common/FluidBaseClient.hpp"
#include "../common/ParameterConstraints.hpp"
#include "../common/ParameterTypes.hpp"
#include "../common/ParameterSet.hpp"
#include "../../algorithms/public/STFT.hpp"
#include "../../data/FluidTensor.hpp"
#include "../../data/TensorTypes.hpp"

namespace fluid {
namespace client {

enum STFTParamIndex { kFFT, kMaxFFT };

auto constexpr STFTParams = defineParameters(
    FFTParam<kMaxFFT>("fftSettings","FFT Settings", 1024, -1, -1),
    LongParam<Fixed<true>>("maxFFTSize", "Maxiumm FFT Size", 16384, Min(4), PowerOfTwo{})
  );


class BaseSTFTClient : public FluidBaseClient<decltype(STFTParams), STFTParams>, public AudioIn, public AudioOut
{

public:

  BaseSTFTClient(ParamSetViewType& p) : FluidBaseClient(p), mSTFTBufferedProcess{static_cast<size_t>(get<kMaxFFT>()),1,1}
  {
    FluidBaseClient::audioChannelsIn(1);
    FluidBaseClient::audioChannelsOut(1);
  }

  size_t latency() { return get<kFFT>().winSize(); }

  template <typename T>
  void process(std::vector<HostVector<T>> &input, std::vector<HostVector<T>> &output, FluidContext& c,
               bool reset = false) {

    if (!input[0].data() || !output[0].data())
      return;
    // Here we do an STFT and its inverse
    mSTFTBufferedProcess.process(mParams, input, output, c, reset,
        [](ComplexMatrixView in, ComplexMatrixView out) { out = in; });
  }

private:
  STFTBufferedProcess<ParamSetViewType, kFFT, true> mSTFTBufferedProcess;
};
} // namespace client
} // namespace fluid
