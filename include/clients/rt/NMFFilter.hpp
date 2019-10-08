#pragma once

#include "BufferedProcess.hpp"
#include "../common/ParameterTypes.hpp"
#include "../common/ParameterConstraints.hpp"
#include "../common/ParameterSet.hpp"
#include "../common/FluidBaseClient.hpp"
#include "../common/ParameterSet.hpp"
#include "../common/ParameterTrackChanges.hpp"
#include "../../algorithms/public/NMF.hpp"
#include "../../algorithms/public/RatioMask.hpp"

namespace fluid {
namespace client {

enum NMFFilterIndex{kFilterbuf,kMaxRank,kIterations,kFFT,kMaxFFTSize};

auto constexpr NMFFilterParams = defineParameters(
  InputBufferParam("bases", "Bases Buffer"),
  LongParam<Fixed<true>>("maxComponents","Maximum Number of Components",20,Min(1)),
  LongParam("iterations", "Number of Iterations", 10, Min(1)),
  FFTParam<kMaxFFTSize>("fftSettings","FFT Settings",1024, -1,-1),
  LongParam<Fixed<true>>("maxFFTSize", "Maxiumm FFT Size", 16384, Min(4), PowerOfTwo{})
);


template <typename T>
class NMFFilter : public FluidBaseClient<decltype(NMFFilterParams), NMFFilterParams>, public AudioIn, public AudioOut
{
  using HostVector = FluidTensorView<T,1>;
public:

  NMFFilter(ParamSetViewType& p) : FluidBaseClient(p), mSTFTProcessor(get<kMaxFFTSize>(),1,get<kMaxRank>())
  {
    audioChannelsIn(1);
    audioChannelsOut(get<kMaxRank>());
  }

  size_t latency() { return get<kFFT>().winSize(); }

  void process(std::vector<HostVector> &input, std::vector<HostVector> &output, FluidContext& c, bool reset = false)
  {
    if(!input[0].data()) return;
    assert(audioChannelsOut() && "No control channels");
    assert(output.size() >= audioChannelsOut() && "Too few output channels");

    if (get<kFilterbuf>().get()) {

      auto filterBuffer = BufferAdaptor::ReadAccess(get<kFilterbuf>().get());
      auto& fftParams = get<kFFT>();

      if (!filterBuffer.valid()) {
        return ;
      }

      size_t rank  = std::min<size_t>(filterBuffer.numChans(),get<kMaxRank>());

      if (filterBuffer.numFrames() != fftParams.frameSize())
      {
        return;
      }

      if(mTrackValues.changed(rank, fftParams.frameSize()))
      {
        tmpFilt.resize(rank,fftParams.frameSize());
        tmpMagnitude.resize(1,fftParams.frameSize());
        tmpOut.resize(rank);
        tmpEstimate.resize(1,fftParams.frameSize());
        tmpSource.resize(1,fftParams.frameSize());
        mNMF.init(rank, get<kIterations>());
      }

      for (size_t i = 0; i < tmpFilt.rows(); ++i)
        tmpFilt.row(i) = filterBuffer.samps(i);

//      controlTrigger(false);
      mSTFTProcessor.process(mParams, input, output, c, reset,
        [&](ComplexMatrixView in,ComplexMatrixView out)
        {
          algorithm::STFT::magnitude(in, tmpMagnitude);
          mNMF.processFrame(tmpMagnitude.row(0), tmpFilt, tmpOut,get<kIterations>(),tmpEstimate.row(0));
          mMask.init(tmpEstimate,1);
          for(size_t i = 0; i < rank; ++i)
          {
            algorithm::NMF::estimate(tmpFilt,RealMatrixView(tmpOut),i,tmpSource);
            mMask.process(in,RealMatrixView{tmpSource},ComplexMatrixView{out.row(i)});
          }
        });
    }
  }

private:
  ParameterTrackChanges<size_t,size_t> mTrackValues;
  STFTBufferedProcess<ParamSetViewType, T, kFFT,true> mSTFTProcessor;
  //std::unique_ptr<algorithm::NMF> mNMF;
  algorithm::NMF mNMF{get<kMaxRank>()};
  algorithm::RatioMask mMask;
  RealMatrix a;

  RealMatrix tmpFilt;
  RealMatrix tmpMagnitude;
  RealVector tmpOut;
  RealMatrix tmpEstimate;
  RealMatrix tmpSource;

  size_t mNBins{0};
  size_t mRank{0};
};
} // namespace client
} // namespace fluid
