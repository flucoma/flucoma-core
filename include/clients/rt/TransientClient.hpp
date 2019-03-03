#pragma once


#include <algorithms/public/TransientExtraction.hpp>
#include <clients/common/FluidBaseClient.hpp>
#include <clients/common/ParameterConstraints.hpp>
#include <clients/common/ParameterTypes.hpp>
#include <clients/common/ParameterTrackChanges.hpp>
#include <clients/rt/BufferedProcess.hpp>
#include <clients/common/ParameterSet.hpp>
#include <clients/nrt/FluidNRTClientWrapper.hpp>
#include <complex>
#include <data/TensorTypes.hpp>
#include <string>
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
  kDebounce
};

auto constexpr TransientParams = defineParameters(
    LongParam("order", "Order", 50, Min(20), LowerLimit<kWinSize>(),UpperLimit<kBlockSize>()),
    LongParam("blockSize", "Block Size", 256, Min(100), LowerLimit<kOrder>()),
    LongParam("padding", "Padding", 128, Min(0)),
    FloatParam("skew", "Skew", 0, Min(-10), Max(10)),
    FloatParam("threshFwd", "Forward Threshold", 3, Min(0)),
    FloatParam("threshBack", "Backward Threshold", 1.1, Min(0)),
    LongParam("winSize", "Window Size", 14, Min(0), UpperLimit<kOrder>()),
    LongParam("debounce", "Debounce", 25, Min(0)));

template <typename Params, typename T, typename U = T>
class TransientClient : public FluidBaseClient<Params>, public AudioIn, public AudioOut {

public:

  using HostVector = HostVector<U>;
  using B = FluidBaseClient<Params>;

  TransientClient(Params& p) : FluidBaseClient<Params>(p) {
    B::audioChannelsIn(1);
    B::audioChannelsOut(2);
  }
  
  void process(std::vector<HostVector>& input,
               std::vector<HostVector>& output) {

   
    if(!input[0].data() || (!output[0].data() && !output[1].data()))
      return;

    static constexpr unsigned iterations = 3;
    static constexpr bool refine = false;
    static constexpr double robustFactor = 3.0;

    std::size_t order = param<kOrder>(*this);
    std::size_t blockSize = param<kBlockSize>(*this);
    std::size_t padding = param<kPadding>(*this);
    std::size_t hostVecSize = input[0].size();
    std::size_t maxWin = 2*blockSize + padding;

    if (!mExtractor.get() || !mExtractor.get() || mTrackValues.changed(order, blockSize, padding, hostVecSize)) {
      mExtractor.reset(new algorithm::TransientExtraction(
          order, iterations, robustFactor, refine));
      mExtractor->prepareStream(blockSize, padding);
      mBufferedProcess.hostSize(hostVecSize);
      mBufferedProcess.maxSize(maxWin, B::audioChannelsIn(), B::audioChannelsOut());
    }

    double skew = std::pow(2, param<kSkew>(*this));
    double threshFwd = param<kThreshFwd>(*this);
    double thresBack = param<kThreshBack>(*this);
    size_t halfWindow = std::round(param<kWinSize>(*this) / 2);
    size_t debounce = param<kDebounce>(*this);

    mExtractor->setDetectionParameters(skew, threshFwd, thresBack, halfWindow, debounce);

    RealMatrix in(1,hostVecSize);

    in.row(0) = input[0]; //need to convert float->double in some hosts
    mBufferedProcess.push(RealMatrixView(in));

    mBufferedProcess.process(mExtractor->inputSize(), mExtractor->hopSize(), [this](RealMatrixView in, RealMatrixView out)
    {
      mExtractor->process(in.row(0), out.row(0), out.row(1));
    });

    RealMatrix out(2, hostVecSize);
    mBufferedProcess.pull(RealMatrixView(out));

    if(output[0].data()) output[0] = out.row(0);
    if(output[1].data()) output[1] = out.row(1);
  }

  long latency()
  {
    return param<kPadding>(*this) +param<kBlockSize>(*this) -  param<kOrder>(*this);
  }

private:
  ParameterTrackChanges<size_t,size_t,size_t,size_t> mTrackValues;
  std::unique_ptr<algorithm::TransientExtraction> mExtractor;
  BufferedProcess mBufferedProcess;
  FluidTensor<T, 1> mTransients;
  FluidTensor<T, 1> mRes;
  size_t mHostSize{0};
  size_t mOrder{0};
  size_t mBlocksize{0};
  size_t mPadding{0};
};


template <typename Params, typename T, typename U>
using NRTTransients = NRTStreamAdaptor<TransientClient,Params,T,U,1,2>;

auto constexpr NRTTransientParams = impl::makeNRTParams({BufferParam("srcBuf", "Source Buffer")}, {BufferParam("transBuf","Transients Buffer"),BufferParam("resBuf","Residual Buffer")}, TransientParams);


} // namespace client
} // namespace fluid

