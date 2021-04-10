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

#include "../common/AudioClient.hpp"
#include "../common/BufferedProcess.hpp"
#include "../common/FluidBaseClient.hpp"
#include "../common/FluidNRTClientWrapper.hpp"
#include "../common/ParameterConstraints.hpp"
#include "../common/ParameterSet.hpp"
#include "../common/ParameterTypes.hpp"
#include "../../algorithms/public/SpectralShape.hpp"
#include "../../data/TensorTypes.hpp"
#include <tuple>

namespace fluid {
namespace client {

using algorithm::SpectralShape;

class SpectralShapeClient : public FluidBaseClient,
                            public AudioIn,
                            public ControlOut
{

  enum SpectralShapeParamIndex { kFFT, kMaxFFTSize };

public:
  using ParamDescType = 
  std::add_const_t<decltype(defineParameters(FFTParam<kMaxFFTSize>("fftSettings", "FFT Settings",
                                             1024, -1, -1),
                       LongParam<Fixed<true>>("maxFFTSize", "Maxiumm FFT Size",
                                              16384, Min(4), PowerOfTwo{})))>; 

  using ParamSetViewType = ParameterSetView<ParamDescType>;
  std::reference_wrapper<ParamSetViewType> mParams;

  void setParams(ParamSetViewType& p) { mParams = p; }

  template <size_t N> 
  auto& get() const
  {
    return mParams.get().template get<N>();
  }

  static constexpr auto getParameterDescriptors()
  { 
    return defineParameters(FFTParam<kMaxFFTSize>("fftSettings", "FFT Settings",
                                             1024, -1, -1),
                       LongParam<Fixed<true>>("maxFFTSize", "Maxiumm FFT Size",
                                              16384, Min(4), PowerOfTwo{})); 
  }


  SpectralShapeClient(ParamSetViewType& p)
      : mParams(p),
        mSTFTBufferedProcess(get<kMaxFFTSize>(), 1, 0), mAlgorithm{
                                                            get<kMaxFFTSize>()}
  {
    audioChannelsIn(1);
    controlChannelsOut(7);
    mDescriptors = FluidTensor<double, 1>(7);
  }

  template <typename T>
  void process(std::vector<HostVector<T>>& input,
               std::vector<HostVector<T>>& output, FluidContext& c)
  {
    using std::size_t;

    if (!input[0].data() || !output[0].data()) return;
    assert(FluidBaseClient::controlChannelsOut() && "No control channels");
    assert(output.size() >= asUnsigned(FluidBaseClient::controlChannelsOut()) &&
           "Too few output channels");

    if (mTracker.changed(get<kFFT>().frameSize(), sampleRate()))
    {
      mMagnitude.resize(get<kFFT>().frameSize());
      mBinHz = sampleRate() / get<kFFT>().fftSize();
    }

    mSTFTBufferedProcess.processInput(
        mParams, input, c, [&](ComplexMatrixView in) {
          algorithm::STFT::magnitude(in.row(0), mMagnitude);
          mAlgorithm.processFrame(mMagnitude, mDescriptors);
        });

    for (int i = 0; i < 7; ++i)
    {
      // TODO: probably move this logic to algorithm
      if (i == 0 || i == 1 || i == 4)
        output[asUnsigned(i)](0) = static_cast<T>(mBinHz * mDescriptors(i));
      else
        output[asUnsigned(i)](0) = static_cast<T>(mDescriptors(i));
    }
  }

  index latency() { return get<kFFT>().winSize(); }

  void reset() { mSTFTBufferedProcess.reset(); }

  index controlRate() { return get<kFFT>().hopSize(); }

private:
  ParameterTrackChanges<index, double>        mTracker;
  STFTBufferedProcess<ParamSetViewType, kFFT> mSTFTBufferedProcess;

  SpectralShape          mAlgorithm;
  FluidTensor<double, 1> mMagnitude;
  FluidTensor<double, 1> mDescriptors;
  double                 mBinHz;
};

using RTSpectralShapeClient = ClientWrapper<SpectralShapeClient>;

auto constexpr NRTSpectralShapeParams = makeNRTParams<SpectralShapeClient>(
    InputBufferParam("source", "Source Buffer"),
    BufferParam("features", "Features Buffer"));

using NRTSpectralShapeClient =
    NRTControlAdaptor<SpectralShapeClient, decltype(NRTSpectralShapeParams),
                      NRTSpectralShapeParams, 1, 1>;

using NRTThreadedSpectralShapeClient =
    NRTThreadingAdaptor<NRTSpectralShapeClient>;

} // namespace client
} // namespace fluid
