#pragma once

#include <clients/common/ParameterTypes.hpp>
#include <clients/common/ParameterConstraints.hpp>
#include <clients/common/ParameterSet.hpp>
#include <clients/common/FluidBaseClient.hpp>
#include <clients/common/ParameterSet.hpp>
#include <clients/rt/BufferedProcess.hpp>
#include <algorithms/public/NMF.hpp>
#include <clients/common/ParameterTrackChanges.hpp>
namespace fluid {
namespace client {

enum NMFMatchParamIndex{kFilterbuf,kMaxRank,kIterations,kFFT,kMaxFFTSize};

auto constexpr NMFMatchParams = defineParameters(
  BufferParam("dictBuf", "Dictionaries Buffer"),
  LongParam<Fixed<true>>("maxRank","Maximum Rank",20,Min(1)),
  LongParam("nIter", "Iterations", 10, Min(1)),
  FFTParam<kMaxFFTSize>("fft","FFT Settings",1024, -1,-1),
  LongParam<Fixed<true>>("maxFFTSize", "Maxiumm FFT Size", 16384)
);


template <typename T>
class NMFMatch : public FluidBaseClient<decltype(NMFMatchParams), NMFMatchParams>, public AudioIn, public ControlOut {
  using HostVector = HostVector<T>;
public:

  NMFMatch(ParamSetType& p) : FluidBaseClient(p), mSTFTProcessor(param<kMaxFFTSize>(p),1,0)
  {
    FluidBaseClient::audioChannelsIn(1);
    FluidBaseClient::controlChannelsOut(param<kMaxRank>(p));
  }

  size_t latency() { return param<kFFT>(mParams).winSize(); }

  void process(std::vector<HostVector> &input, std::vector<HostVector> &output)
  {
    if(!input[0].data()) return;// {Result::Status::kOk,""};
    assert(FluidBaseClient::controlChannelsOut() && "No control channels");
    assert(output.size() >= FluidBaseClient::controlChannelsOut() && "Too few output channels");

    if (param<kFilterbuf>(mParams).get()) {

      auto filterBuffer = BufferAdaptor::Access(param<kFilterbuf>(mParams).get());
      auto& fftParams = param<kFFT>(mParams);

      if (!filterBuffer.valid()) {
        return ;//{Result::Status::kError,"Filter buffer invalid"};
      }

      size_t rank  = std::min<size_t>(filterBuffer.numChans(),param<kMaxRank>(mParams));

      if (filterBuffer.numFrames() != fftParams.frameSize())
      {
        return;// {Result::Status::kError, "NMFMatch: Filters buffer needs to be (fftsize / 2 + 1) frames"};
      }

      if(mTrackValues.changed(rank, fftParams.frameSize()))
      {
        tmpFilt.resize(rank,fftParams.frameSize());
        tmpMagnitude.resize(1,fftParams.frameSize());
        tmpOut.resize(rank);
        mNMF.reset(new algorithm::NMF(rank, param<kIterations>(mParams)));
      }

      for (size_t i = 0; i < tmpFilt.rows(); ++i)
        tmpFilt.row(i) = filterBuffer.samps(0, i);

//      controlTrigger(false);
      mSTFTProcessor.processInput(mParams, input,
        [&](ComplexMatrixView in)
        {
          algorithm::STFT::magnitude(in, tmpMagnitude);
          mNMF->processFrame(tmpMagnitude.row(0), tmpFilt, tmpOut);
//          controlTrigger(true);
        });

        for(size_t i = 0; i < rank; ++i)
          output[i](0) = tmpOut(i);
    }
    //return;// {Result::Status::kOk};
  }

private:
  ParameterTrackChanges<size_t,size_t> mTrackValues;
  STFTBufferedProcess<ParamSetType, T, kFFT,false> mSTFTProcessor;
  std::unique_ptr<algorithm::NMF> mNMF;

  FluidTensor<double, 2> tmpFilt;
  FluidTensor<double, 2> tmpMagnitude;
  FluidTensor<double, 1> tmpOut;

  size_t mNBins{0};
  size_t mRank{0};
};
} // namespace client
} // namespace fluid
