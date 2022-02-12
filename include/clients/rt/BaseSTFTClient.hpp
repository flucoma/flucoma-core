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
namespace stftpass {

enum STFTParamIndex { kFFT, kMaxFFT };

constexpr auto STFTPassParams = defineParameters(
    FFTParam<kMaxFFT>("fftSettings", "FFT Settings", 1024, -1, -1),
    LongParam<Fixed<true>>("maxFFTSize", "Maxiumm FFT Size", 16384, Min(4),
                           PowerOfTwo{}));

class BaseSTFTClient : public FluidBaseClient, public AudioIn, public AudioOut
{
public:
  using ParamDescType = std::add_const_t<decltype(STFTPassParams)>;

  using ParamSetViewType = ParameterSetView<ParamDescType>;
  std::reference_wrapper<ParamSetViewType> mParams;

  void setParams(ParamSetViewType& p) { mParams = p; }

  template <size_t N>
  auto& get() const
  {
    return mParams.get().template get<N>();
  }

  static constexpr auto& getParameterDescriptors() { return STFTPassParams; }


  BaseSTFTClient(ParamSetViewType& p)
      : mParams(p), mSTFTBufferedProcess{get<kMaxFFT>(), 1, 1}
  {
    audioChannelsIn(1);
    audioChannelsOut(1);
  }

  index latency() { return get<kFFT>().winSize(); }

  void reset() { mSTFTBufferedProcess.reset(); }

  template <typename T>
  void process(std::vector<HostVector<T>>& input,
               std::vector<HostVector<T>>& output, FluidContext& c)
  {

    if (!input[0].data() || !output[0].data()) return;
    // Here we do an STFT and its inverse
    mSTFTBufferedProcess.process(
        mParams, input, output, c,
        [](ComplexMatrixView in, ComplexMatrixView out) { out <<= in; });
  }

private:
  STFTBufferedProcess<ParamSetViewType, kFFT, true> mSTFTBufferedProcess;
};
} // namespace stftpass

using RTSTFTPassClient = ClientWrapper<stftpass::BaseSTFTClient>;

} // namespace client
} // namespace fluid
