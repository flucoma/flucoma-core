#pragma once

#include <algorithms/public/TransientSegmentation.hpp>
#include <clients/common/ParameterTypes.hpp>
#include <clients/common/ParameterConstraints.hpp>
#include <clients/common/ParameterTrackChanges.hpp>
#include <clients/common/ParameterSet.hpp>
#include <clients/common/FluidBaseClient.hpp>
#include <clients/common/AudioClient.hpp>
#include <clients/rt/BufferedProcess.hpp>
#include <clients/nrt/FluidNRTClientWrapper.hpp>
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

auto constexpr TransientParams = defineParameters(
    LongParam("order", "Order", 50, Min(20), LowerLimit<kWinSize>(),UpperLimit<kBlockSize>()),
    LongParam("blockSize", "Block Size", 256, Min(100), LowerLimit<kOrder>()),
    LongParam("padding", "Padding", 128, Min(0)),
    FloatParam("skew", "Skew", 0, Min(-10), Max(10)),
    FloatParam("threshFwd", "Forward Threshold", 3, Min(0)),
    FloatParam("threshBack", "Backward Threshold", 1.1, Min(0)),
    LongParam("winSize", "Window Size", 14, Min(0), UpperLimit<kOrder>()),
    LongParam("debounce", "Debounce", 25, Min(0)),
    LongParam("minSlice","Minimum Segment",1000)
);


template <typename Params, typename T, typename U = T>
class TransientsSlice : public FluidBaseClient<Params>, public AudioIn, public AudioOut

{
  using HostVector = HostVector<U>;

public:

  TransientsSlice(Params& p): mParams{p}, FluidBaseClient<Params>{p}
  {
    FluidBaseClient<Params>::audioChannelsIn(1);
    FluidBaseClient<Params>::audioChannelsOut(1);
  }

  void process(std::vector<HostVector>& input,
               std::vector<HostVector>& output) {

    if(!input[0].data() || !output[0].data())
      return;

    static constexpr unsigned iterations = 3;
    static constexpr bool refine = false;
    static constexpr double robustFactor = 3.0;

    std::size_t order = param<kOrder>(mParams);
    std::size_t blockSize = param<kBlockSize>(mParams);
    std::size_t padding = param<kPadding>(mParams);
    std::size_t hostVecSize = input[0].size();
    std::size_t maxWin = 2*blockSize + padding;

    if (!mExtractor.get() || mTrackValues.changed(order, blockSize, padding, hostVecSize)) {
      mExtractor.reset(new algorithm::TransientSegmentation(order, iterations, robustFactor));
      mExtractor->prepareStream(blockSize, padding);
      mBufferedProcess.hostSize(hostVecSize);
      mBufferedProcess.maxSize(maxWin, FluidBaseClient<Params>::audioChannelsIn(), FluidBaseClient<Params>::audioChannelsOut());

    }

    double skew = std::pow(2, param<kSkew>(mParams));
    double threshFwd = param<kThreshFwd>(mParams);
    double thresBack = param<kThreshBack>(mParams);
    size_t halfWindow = std::round(param<kWinSize>(mParams) / 2);
    size_t debounce = param<kDebounce>(mParams);
    size_t minSeg = param<kMinSeg>(mParams);

    mExtractor->setDetectionParameters(skew, threshFwd, thresBack, halfWindow,
                                       debounce, minSeg);

    RealMatrix in(1,hostVecSize);

    in.row(0) = input[0]; //need to convert float->double in some hosts
    mBufferedProcess.push(RealMatrixView(in));

    mBufferedProcess.process(mExtractor->inputSize(), mExtractor->hopSize(), [this](RealMatrixView in, RealMatrixView out)
    {
      mExtractor->process(in.row(0), out.row(0));
    });

    RealMatrix out(1, hostVecSize);
    mBufferedProcess.pull(RealMatrixView(out));

    if(output[0].data()) output[0] = out.row(0);
  }

  long latency()
  {
    return param<kPadding>(*this) +param<kBlockSize>(*this) -  param<kOrder>(*this);
  }

private:

  ParameterTrackChanges<size_t,size_t,size_t,size_t> mTrackValues;
  std::unique_ptr<algorithm::TransientSegmentation> mExtractor;
  BufferedProcess mBufferedProcess;
  FluidTensor<T, 1> mTransients;
  size_t mHostSize{0};
  size_t mOrder{0};
  size_t mBlocksize{0};
  size_t mPadding{0};
  Params& mParams;
};


template <typename Params, typename T, typename U>
using NRTTransientSlice = NRTSliceAdaptor<TransientsSlice,Params,T,U,1,1>;

auto constexpr NRTTransientSliceParams = impl::makeNRTParams({BufferParam("srcBuf", "Source Buffer")}, {BufferParam("idxBuf","Indexes Buffer")}, TransientParams);


} // namespace client
} // namespace fluid
