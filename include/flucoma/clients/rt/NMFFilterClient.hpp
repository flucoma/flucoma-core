/*
Part of the Fluid Corpus Manipulation Project (http://www.flucoma.org/)
Copyright University of Huddersfield.
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
#include "../common/ParameterTrackChanges.hpp"
#include "../common/ParameterTypes.hpp"
#include "../../algorithms/public/NMF.hpp"
#include "../../algorithms/public/RatioMask.hpp"

namespace fluid {
namespace client {
namespace nmffilter {

enum NMFFilterIndex { kFilterbuf, kMaxRank, kIterations, kFFT };

constexpr auto NMFFilterParams = defineParameters(
    InputBufferParam("bases", "Bases Buffer"),
    LongParamRuntimeMax<Primary>("maxComponents", "Maximum Number of Components", 20, Min(1)),
    LongParam("iterations", "Number of Iterations", 10, Min(1)),
    FFTParam("fftSettings", "FFT Settings", 1024, -1, -1));

class NMFFilterClient : public FluidBaseClient, public AudioIn, public AudioOut
{

public:
  using ParamDescType = decltype(NMFFilterParams);

  using ParamSetViewType = ParameterSetView<ParamDescType>;
  std::reference_wrapper<ParamSetViewType> mParams;

  void setParams(ParamSetViewType& p) { mParams = p; }

  template <size_t N>
  auto& get() const
  {
    return mParams.get().template get<N>();
  }

  static constexpr auto& getParameterDescriptors() { return NMFFilterParams; }

  NMFFilterClient(ParamSetViewType& p, FluidContext& c)
      : mParams{p}, mSTFTProcessor{get<kFFT>(), 1, get<kMaxRank>().max(),
                                   c.hostVectorSize(), c.allocator()},
        mMask{get<kMaxRank>().max(), get<kFFT>().maxFrameSize(), c.allocator()}
  {
    audioChannelsIn(1);
    audioChannelsOut(get<kMaxRank>().max());
    setInputLabels({"audio input"});
    setOutputLabels({"filtered input"});
  }

  index latency() const { return get<kFFT>().winSize(); }

  void reset(FluidContext&) { mSTFTProcessor.reset(); }

  template <typename T>
  void process(std::vector<HostVector<T>>& input,
               std::vector<HostVector<T>>& output, FluidContext& c)
  {
    if (!input[0].data()) return;
    assert(audioChannelsOut() && "No control channels");
    assert(output.size() >= asUnsigned(audioChannelsOut()) &&
           "Too few output channels");

    if (get<kFilterbuf>().get())
    {

      auto  filterBuffer = BufferAdaptor::ReadAccess(get<kFilterbuf>().get());
      auto& fftParams = get<kFFT>();

      if (!filterBuffer.valid()) { return; }

      index rank = std::min<index>(filterBuffer.numChans(), get<kMaxRank>().max());

      if (filterBuffer.numFrames() != fftParams.frameSize()) { return; }

      if (mTrackValues.changed(rank, fftParams.frameSize()))
      {
        tmpFilt.resize(rank, fftParams.frameSize());
        tmpMagnitude.resize(1, fftParams.frameSize());
        tmpOut.resize(rank);
        tmpEstimate.resize(1, fftParams.frameSize());
        tmpSource.resize(1, fftParams.frameSize());
      }

      for (index i = 0; i < tmpFilt.rows(); ++i)
        tmpFilt.row(i) <<= filterBuffer.samps(i);

      //      controlTrigger(false);
      mSTFTProcessor.process(
          get<kFFT>(), input, output, c,
          [&](ComplexMatrixView in, ComplexMatrixView out) {
            algorithm::STFT::magnitude(in, tmpMagnitude);
            mNMF.processFrame(tmpMagnitude.row(0), tmpFilt, tmpOut,
                              get<kIterations>(), tmpEstimate.row(0), c.allocator());
            mMask.init(tmpEstimate);
            for (index i = 0; i < rank; ++i)
            {
              algorithm::NMF::estimate(tmpFilt, RealMatrixView(tmpOut), i,
                                       tmpSource);
              mMask.process(in, RealMatrixView(tmpSource), 1,
                            ComplexMatrixView{out.row(i)});
            }
          });
    }
  }

private:
  ParameterTrackChanges<index, index>               mTrackValues;
  STFTBufferedProcess<true> mSTFTProcessor;

  algorithm::NMF       mNMF;
  algorithm::RatioMask mMask;

  RealMatrix a;
  RealMatrix tmpFilt;
  RealMatrix tmpMagnitude;
  RealVector tmpOut;
  RealMatrix tmpEstimate;
  RealMatrix tmpSource;
};
} // namespace nmffilter

using RTNMFFilterClient = ClientWrapper<nmffilter::NMFFilterClient>;

} // namespace client
} // namespace fluid
