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

namespace fluid {
namespace client {
namespace nmfmatch {

enum NMFMatchParamIndex {
  kFilterbuf,
  kMaxRank,
  kIterations,
  kFFT
};

constexpr auto NMFMatchParams = defineParameters(
    InputBufferParam("bases", "Bases Buffer"),
    LongParamRuntimeMax<Primary>("maxComponents", "Maximum Number of Components", 20,
                           Min(1)),
    LongParam("iterations", "Number of Iterations", 10, Min(1)),
    FFTParam("fftSettings", "FFT Settings", 1024, -1, -1));

class NMFMatchClient : public FluidBaseClient, public AudioIn, public ControlOut
{
public:
  using ParamDescType = decltype(NMFMatchParams);

  using ParamSetViewType = ParameterSetView<ParamDescType>;
  std::reference_wrapper<ParamSetViewType> mParams;

  void setParams(ParamSetViewType& p) { mParams = p; }

  template <size_t N>
  auto& get() const
  {
    return mParams.get().template get<N>();
  }

  static constexpr auto& getParameterDescriptors() { return NMFMatchParams; }

  NMFMatchClient(ParamSetViewType& p, FluidContext& c)
      : mParams(p),
        mFilter(get<kMaxRank>().max(),get<kFFT>().maxFrameSize(),c.allocator()),
        mMagnitude(1, get<kFFT>().maxFrameSize(),c.allocator()),
        mActivations(get<kMaxRank>().max(), c.allocator()),
        mSTFTProcessor(get<kFFT>(), 1, 0, c.hostVectorSize(), c.allocator())
  {
    audioChannelsIn(1);
    controlChannelsOut({1, get<kMaxRank>(),get<kMaxRank>().max()});
    setInputLabels({"audio input"});
    setOutputLabels({"activation amount for each component"});
  }

  index latency() const { return get<kFFT>().winSize(); }

  void reset(FluidContext&) { mSTFTProcessor.reset(); }

  template <typename T>
  void process(std::vector<HostVector<T>>& input,
               std::vector<HostVector<T>>& output, FluidContext& c)
  {
    if (!input[0].data()) return;
    assert(FluidBaseClient::controlChannelsOut().size && "No control channels");
    assert(output[0].size() >= controlChannelsOut().size &&
           "Too few output channels");

    if (get<kFilterbuf>().get())
    {

      auto  filterBuffer = BufferAdaptor::ReadAccess(get<kFilterbuf>().get());
      auto& fftParams = get<kFFT>();

      if (!filterBuffer.valid()) { return; }

      index rank = std::min<index>(filterBuffer.numChans(), get<kMaxRank>());
      index frameSize = fftParams.frameSize();

      if (filterBuffer.numFrames() != frameSize) { return; }
      
      if (mTrackValues.changed(rank, frameSize))
      {
        controlChannelsOut({1, rank});
      }

      auto mags = mMagnitude(Slice(0),Slice(0,frameSize));
      auto filter = mFilter(Slice(0,rank),Slice(0,frameSize));
      auto activations = mActivations(Slice(0,rank));
      
      for (index i = 0; i < filter.rows(); ++i)
        filter.row(i) <<= filterBuffer.samps(i);

      mSTFTProcessor.processInput(get<kFFT>(), input, c, [&](ComplexMatrixView in) {
        algorithm::STFT::magnitude(in, mags);
        mNMF.processFrame(mags.row(0), filter, activations,
            10, FluidTensorView<double,1>{nullptr, 0, 0}, c.allocator());
      });

      output[0](Slice(0,rank)) <<= activations;
      output[0](Slice(rank,get<kMaxRank>().max() - rank)).fill(0);
    }
  }

private:
  ParameterTrackChanges<index, index> mTrackValues;
  algorithm::NMF                      mNMF;
  FluidTensor<double, 2>              mFilter;
  FluidTensor<double, 2>              mMagnitude;
  FluidTensor<double, 1>              mActivations;

  STFTBufferedProcess<false> mSTFTProcessor;
};
} // namespace nmfmatch

using RTNMFMatchClient = ClientWrapper<nmfmatch::NMFMatchClient>;

} // namespace client
} // namespace fluid
