#pragma once

#include <algorithms/public/TransientSegmentation.hpp>
#include <clients/common/ParameterTypes.hpp>
#include <clients/common/ParameterConstraints.hpp>
#include <clients/common/ParameterTrackChanges.hpp>
#include <clients/common/FluidBaseClient.hpp>
#include <clients/common/AudioClient.hpp>
#include <clients/rt/BufferedProcess.hpp>

#include <tuple>

namespace fluid {
namespace client {

enum TransientParamIndex {
  kOrder,
  kBlockSize,
  kPadding,
  kSkew,
  kThreshFwd,
  kThreshBack,
  kWinSize,
  kDebounce,
  kMinSeg
};

auto constexpr TransientParams = std::make_tuple(
    LongParam("order", "Order", 50, Min(20), LowerLimit<kWinSize>(),UpperLimit<kBlockSize>()),
    LongParam("blockSize", "Block Size", 256, Min(100), LowerLimit<kOrder>()),
    LongParam("padding", "Padding", 128, Min(0)),
    FloatParam("skew", "Skew", 0, Min(-10), Max(10)),
    FloatParam("threshFwd", "Forward Threshold", 3, Min(0)),
    FloatParam("threshBack", "Backward Threshold", 1.1, Min(0)),
    LongParam("winSize", "Window Size", 14, Min(0), UpperLimit<kOrder>()),
    LongParam("debounce", "Debounce", 25, Min(0)),
    LongParam("minSegment","Minimum Segment",50)
);
  
using ParamT = decltype(TransientParams);

template <typename T, typename U = T>
class TransientsSlice : public FluidBaseClient<ParamT>, public AudioIn, public AudioOut

{
  using HostVector = HostVector<U>;


public:

  TransientsSlice(): FluidBaseClient<ParamT>(TransientParams)
  {
    audioChannelsIn(1);
    audioChannelsOut(1);
  }

  
  void process(std::vector<HostVector>& input,
               std::vector<HostVector>& output) {

   
    if(!input[0].data() || !output[0].data())
      return;
    
    static constexpr unsigned iterations = 3;
    static constexpr bool refine = false;
    static constexpr double robustFactor = 3.0;

    std::size_t order = get<kOrder>();
    std::size_t blockSize = get<kBlockSize>();
    std::size_t padding = get<kPadding>();
    std::size_t hostVecSize = input[0].size();
    std::size_t maxWin = 2*blockSize + padding;

    if (!mExtractor.get() || mTrackValues.changed(order, blockSize, padding, hostVecSize)) {
      mExtractor.reset(new algorithm::TransientSegmentation(order, iterations, robustFactor));
      mExtractor->prepareStream(blockSize, padding);
      
      //TODO factor this whole mess away into BufferedProcess
      
      mBufferedProcess.maxSize(maxWin, audioChannelsIn(), audioChannelsOut());
      mInputBuffer.setSize(maxWin);
      mOutputBuffer.setSize(maxWin);
      mInputBuffer.setHostBufferSize(hostVecSize);
      mOutputBuffer.setHostBufferSize(hostVecSize);
      mInputBuffer.reset(audioChannelsIn());
      mOutputBuffer.reset(audioChannelsOut());
      mBufferedProcess.setBuffers(mInputBuffer, mOutputBuffer);
      mBufferedProcess.hostSize(hostVecSize);
    }

    double skew = std::pow(2, get<kSkew>());
    double threshFwd = get<kThreshFwd>();
    double thresBack = get<kThreshBack>();
    size_t halfWindow = std::round(get<kWinSize>() / 2);
    size_t debounce = get<kDebounce>();
    size_t minSeg = get<kMinSeg>();


  
    mExtractor->setDetectionParameters(skew, threshFwd, thresBack, halfWindow,
                                       debounce, minSeg);
  
    RealMatrix in(1,hostVecSize);

    in.row(0) = input[0]; //need to convert float->double in some hosts
    mInputBuffer.push(in);
  

    mBufferedProcess.process(mExtractor->inputSize(), mExtractor->hopSize(), [this](RealMatrixView in, RealMatrixView out)
    {
      mExtractor->process(in.row(0), out.row(0));
    });
    
    RealMatrix out(1, hostVecSize);
    mOutputBuffer.pull(out);

    if(output[0].data()) output[0] = out.row(0);
//    if(output[1].data()) output[1] = out.row(1);
  }

  long latency()
  {
    return get<kPadding>() + get<kOrder>() + get<kBlockSize>();
  }


private:

  ParameterTrackChanges<size_t,size_t,size_t,size_t> mTrackValues;
  std::unique_ptr<algorithm::TransientSegmentation> mExtractor;
  FluidSource<double> mInputBuffer;
  FluidSink<double> mOutputBuffer;
  BufferedProcess mBufferedProcess;
  FluidTensor<T, 1> mTransients;
  size_t mHostSize{0};
  size_t mOrder{0};
  size_t mBlocksize{0};
  size_t mPadding{0};
};

} // namespace client
} // namespace fluid
