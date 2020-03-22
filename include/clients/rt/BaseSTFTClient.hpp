/*
Part of the Fluid Corpus Manipulation Project (http://www.flucoma.org/)
Copyright 2017-2019 University of Huddersfield.
Licensed under the BSD-3 License.
See license.md file in the project root for full license information.
This project has received funding from the European Research Council (ERC)
under the European Union’s Horizon 2020 research and innovation programme
(grant agreement No 725899).
*/
#pragma once

#include "../common/BufferedProcess.hpp"
#include "../common/FluidBaseClient.hpp"
#include "../common/ParameterConstraints.hpp"
#include "../common/ParameterSet.hpp"
#include "../common/ParameterTypes.hpp"
#include "../../algorithms/public/STFT.hpp"
#include "../../data/FluidTensor.hpp"
#include "../../data/TensorTypes.hpp"

namespace fluid {
namespace client {

enum STFTParamIndex { kFFT, kMaxFFT };

extern auto constexpr STFTParams = defineParameters(
    FFTParam<kMaxFFT>("fftSettings", "FFT Settings", 1024, -1, -1),
    LongParam<Fixed<true>>("maxFFTSize", "Maxiumm FFT Size", 16384, Min(4),
                           PowerOfTwo{}));

template <typename T>
class BaseSTFTClient : public FluidBaseClient<decltype(STFTParams), STFTParams>,
                       public AudioIn,
                       public AudioOut
{
  using HostVector = FluidTensorView<T, 1>;

public:
  BaseSTFTClient(ParamSetViewType& p)
      : FluidBaseClient(p), mSTFTBufferedProcess{get<kMaxFFT>(), 1, 1}
  {
    FluidBaseClient::audioChannelsIn(1);
    FluidBaseClient::audioChannelsOut(1);
  }

  index latency() { return get<kFFT>().winSize(); }
  
  void reset(){ mSTFTBufferedProcess.reset(); }

  void process(std::vector<HostVector>& input, std::vector<HostVector>& output,
               FluidContext& c)
  {

    if (!input[0].data() || !output[0].data()) return;
    // Here we do an STFT and its inverse
    mSTFTBufferedProcess.process(
        mParams, input, output, c, 
        [](ComplexMatrixView in, ComplexMatrixView out) { out = in; });
  }

private:
  STFTBufferedProcess<ParamSetViewType, T, kFFT, true> mSTFTBufferedProcess;
};
} // namespace client
} // namespace fluid
