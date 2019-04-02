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
    LongParam("order", "Order", 20, Min(10), LowerLimit<kWinSize>(),UpperLimit<kBlockSize>()),
    LongParam("blockSize", "Block Size", 256, Min(100), LowerLimit<kOrder>()),
    LongParam("padSize", "Padding", 128, Min(0)),
    FloatParam("skew", "Skew", 0, Min(-10), Max(10)),
    FloatParam("threshFwd", "Forward Threshold", 2, Min(0)),
    FloatParam("threshBack", "Backward Threshold", 1.1, Min(0)),
    LongParam("winSize", "Window Size", 14, Min(0), UpperLimit<kOrder>()),
    LongParam("debounce", "Debounce", 25, Min(0)));


template <typename T>
class TransientClient : public FluidBaseClient<decltype(TransientParams), TransientParams>, public AudioIn, public AudioOut
{

public:

  using HostVector = HostVector<T>;

  TransientClient(ParamSetViewType& p) : FluidBaseClient(p) {
    FluidBaseClient::audioChannelsIn(1);
    FluidBaseClient::audioChannelsOut(2);
  }

  void process(std::vector<HostVector>& input, std::vector<HostVector>& output)
  {
    if(!input[0].data() || (!output[0].data() && !output[1].data()))
      return;

    static constexpr unsigned iterations = 3;
    static constexpr bool refine = false;
    static constexpr double robustFactor = 3.0;

    std::size_t order = get<kOrder>();
    std::size_t blockSize = get<kBlockSize>();
    std::size_t padding = get<kPadding>();
    std::size_t hostVecSize = input[0].size();
    std::size_t maxWinIn = 2*blockSize + padding;
    std::size_t maxWinOut = blockSize - order;

    if (!mExtractor.get() || !mExtractor.get() || mTrackValues.changed(order, blockSize, padding, hostVecSize))
    {
      mExtractor.reset(new algorithm::TransientExtraction(order, iterations, robustFactor, refine));
      mExtractor->prepareStream(blockSize, padding);
      mBufferedProcess.hostSize(hostVecSize);
      mBufferedProcess.maxSize(maxWinIn, maxWinOut, FluidBaseClient::audioChannelsIn(), FluidBaseClient::audioChannelsOut());
    }

    double skew = std::pow(2, get<kSkew>());
    double threshFwd = get<kThreshFwd>();
    double thresBack = get<kThreshBack>();
    size_t halfWindow = std::round(get<kWinSize>() / 2);
    size_t debounce = get<kDebounce>();

    mExtractor->setDetectionParameters(skew, threshFwd, thresBack, halfWindow, debounce);

    RealMatrix in(1,hostVecSize);

    in.row(0) = input[0]; //need to convert float->double in some hosts
    mBufferedProcess.push(RealMatrixView(in));

    mBufferedProcess.process(mExtractor->inputSize(), mExtractor->hopSize(), mExtractor->hopSize(), [this](RealMatrixView in, RealMatrixView out)
    {
      mExtractor->process(in.row(0), out.row(0), out.row(1));
    });

    RealMatrix out(2, hostVecSize);
    mBufferedProcess.pull(RealMatrixView(out));

    if(output[0].data()) output[0] = out.row(0);
    if(output[1].data()) output[1] = out.row(1);
  }

  size_t latency()
  {
    return get<kPadding>() + get<kBlockSize>() -  get<kOrder>();
  }

private:
  ParameterTrackChanges<size_t,size_t,size_t,size_t> mTrackValues;
  std::unique_ptr<algorithm::TransientExtraction> mExtractor;
  BufferedProcess mBufferedProcess;
  size_t mHostSize{0};
  size_t mOrder{0};
  size_t mBlocksize{0};
  size_t mPadding{0};
};

auto constexpr NRTTransientParams = makeNRTParams<TransientClient>({BufferParam("source", "Source Buffer")}, {BufferParam("transients","Transients Buffer"), BufferParam("residual","Residual Buffer")});

template <typename T>
using NRTTransients = NRTStreamAdaptor<TransientClient<T>, decltype(NRTTransientParams), NRTTransientParams, 1, 2>;

} // namespace client
} // namespace fluid
