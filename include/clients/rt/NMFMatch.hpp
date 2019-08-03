#pragma once

#include "BufferedProcess.hpp"
#include "../common/ParameterTypes.hpp"
#include "../common/ParameterConstraints.hpp"
#include "../common/ParameterSet.hpp"
#include "../common/FluidBaseClient.hpp"
#include "../common/ParameterSet.hpp"
#include "../common/ParameterTrackChanges.hpp"
#include "../../algorithms/public/NMF.hpp"

namespace fluid {
namespace client {

enum NMFMatchParamIndex{kFilterbuf,kMaxRank,kIterations,kFFT,kMaxFFTSize};

auto constexpr NMFMatchParams = defineParameters(
  InputBufferParam("bases", "Bases Buffer"),
  LongParam<Fixed<true>>("maxComponents","Maximum Number of Components",20,Min(1)),
  LongParam("iterations", "Number of Iterations", 10, Min(1)),
  FFTParam<kMaxFFTSize>("fftSettings","FFT Settings",1024, -1,-1),
  LongParam<Fixed<true>>("maxFFTSize", "Maxiumm FFT Size", 16384, Min(4), PowerOfTwo{})
);


template <typename T>
class NMFMatch : public FluidBaseClient<decltype(NMFMatchParams), NMFMatchParams>, public AudioIn, public ControlOut
{
  using HostVector = HostVector<T>;
public:

  NMFMatch(ParamSetViewType& p) : FluidBaseClient(p), mSTFTProcessor(get<kMaxFFTSize>(),1,0)
  {
    audioChannelsIn(1);
    controlChannelsOut(get<kMaxRank>());
  }

  size_t latency() { return get<kFFT>().winSize(); }

  void process(std::vector<HostVector> &input, std::vector<HostVector> &output, FluidContext& c, bool reset = false)
  {
    if(!input[0].data()) return;
    assert(FluidBaseClient::controlChannelsOut() && "No control channels");
    assert(output.size() >= FluidBaseClient::controlChannelsOut() && "Too few output channels");

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
        mNMF.reset(new algorithm::NMF(rank, get<kIterations>()));
      }

      for (size_t i = 0; i < tmpFilt.rows(); ++i)
        tmpFilt.row(i) = filterBuffer.samps(i);

//      controlTrigger(false);
      mSTFTProcessor.processInput(mParams, input, c, reset,
        [&](ComplexMatrixView in)
        {
          algorithm::STFT::magnitude(in, tmpMagnitude);
         mNMF->processFrame(tmpMagnitude.row(0), tmpFilt, tmpOut);
//          controlTrigger(true);
        });

        for(size_t i = 0; i < rank; ++i)
          output[i](0) = tmpOut(i);
    }
  }

private:
  ParameterTrackChanges<size_t,size_t> mTrackValues;
  STFTBufferedProcess<ParamSetViewType, T, kFFT,false> mSTFTProcessor;
  std::unique_ptr<algorithm::NMF> mNMF;

  FluidTensor<double, 2> tmpFilt;
  FluidTensor<double, 2> tmpMagnitude;
  FluidTensor<double, 1> tmpOut;

  size_t mNBins{0};
  size_t mRank{0};
};
} // namespace client
} // namespace fluid
