/*
Copyright 2017-2019 University of Huddersfield.
Licensed under the BSD-3 License.
See LICENSE file in the project root for full license information.
This project has received funding from the European Research Council (ERC)
under the European Union’s Horizon 2020 research and innovation programme
(grant agreement No 725899).
*/
#pragma once

#include "../../algorithms/public/TransientExtraction.hpp"
#include "../../data/TensorTypes.hpp"
#include "../common/BufferedProcess.hpp"
#include "../common/FluidBaseClient.hpp"
#include "../common/FluidNRTClientWrapper.hpp"
#include "../common/ParameterConstraints.hpp"
#include "../common/ParameterSet.hpp"
#include "../common/ParameterTrackChanges.hpp"
#include "../common/ParameterTypes.hpp"

#include <complex>
#include <string>
#include <tuple>

namespace fluid {
namespace client {

enum TransientParamIndex
{
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
    LongParam("order", "Order", 20, Min(10), LowerLimit<kWinSize>(),
              UpperLimit<kBlockSize>()),
    LongParam("blockSize", "Block Size", 256, Min(100), LowerLimit<kOrder>()),
    LongParam("padSize", "Padding", 128, Min(0)),
    FloatParam("skew", "Skew", 0, Min(-10), Max(10)),
    FloatParam("threshFwd", "Forward Threshold", 2, Min(0)),
    FloatParam("threshBack", "Backward Threshold", 1.1, Min(0)),
    LongParam("windowSize", "Window Size", 14, Min(0), UpperLimit<kOrder>()),
    LongParam("clumpLength", "Clumping Window Length", 25, Min(0)));


template <typename T>
class TransientClient
    : public FluidBaseClient<decltype(TransientParams), TransientParams>,
      public AudioIn,
      public AudioOut
{

public:
  using HostVector = FluidTensorView<T, 1>;

  TransientClient(ParamSetViewType& p) : FluidBaseClient(p)
  {
    FluidBaseClient::audioChannelsIn(1);
    FluidBaseClient::audioChannelsOut(2);
  }

  void process(std::vector<HostVector>& input, std::vector<HostVector>& output,
               FluidContext& c, bool reset = false)
  {
    using namespace std;
    if (!input[0].data() || (!output[0].data() && !output[1].data())) return;

    static constexpr unsigned iterations = 3;
    static constexpr bool     refine = false;
    static constexpr double   robustFactor = 3.0;

    size_t order = get<kOrder>();
    size_t blockSize = get<kBlockSize>();
    size_t padding = get<kPadding>();
    size_t hostVecSize = input[0].size();
    size_t maxWinIn = 2 * blockSize + padding;
    size_t maxWinOut = blockSize - order;

    if (mTrackValues.changed(order, blockSize, padding, hostVecSize) ||
        !mExtractor.initialized())
    {
      mExtractor.init(order, iterations, robustFactor, refine, blockSize,
                      padding);
      // mExtractor.reset(new algorithm::TransientExtraction(order, iterations,
      // robustFactor, refine)); mExtractor->prepareStream(blockSize, padding);
      mBufferedProcess.hostSize(hostVecSize);
      mBufferedProcess.maxSize(maxWinIn, maxWinOut,
                               FluidBaseClient::audioChannelsIn(),
                               FluidBaseClient::audioChannelsOut());
    }

    double skew = pow(2, get<kSkew>());
    double threshFwd = get<kThreshFwd>();
    double thresBack = get<kThreshBack>();
    size_t halfWindow = round(get<kWinSize>() / 2);
    size_t debounce = get<kDebounce>();

    mExtractor.setDetectionParameters(skew, threshFwd, thresBack, halfWindow,
                                      debounce);

    RealMatrix in(1, hostVecSize);

    in.row(0) = input[0]; // need to convert float->double in some hosts
    mBufferedProcess.push(RealMatrixView(in));

    mBufferedProcess.process(
        mExtractor.inputSize(), mExtractor.hopSize(), mExtractor.hopSize(), c,
        reset, [this](RealMatrixView in, RealMatrixView out) {
          mExtractor.process(in.row(0), out.row(0), out.row(1));
        });

    RealMatrix out(2, hostVecSize);
    mBufferedProcess.pull(RealMatrixView(out));

    if (output[0].data()) output[0] = out.row(0);
    if (output[1].data()) output[1] = out.row(1);
  }

  size_t latency()
  {
    return get<kPadding>() + get<kBlockSize>() - get<kOrder>();
  }

private:
  ParameterTrackChanges<size_t, size_t, size_t, size_t> mTrackValues;
  // std::unique_ptr<algorithm::TransientExtraction> mExtractor;
  algorithm::TransientExtraction mExtractor{get<kOrder>(), 3, 3.0, false};
  BufferedProcess                mBufferedProcess;
  size_t                         mHostSize{0};
  size_t                         mOrder{0};
  size_t                         mBlocksize{0};
  size_t                         mPadding{0};
};

auto constexpr NRTTransientParams = makeNRTParams<TransientClient>(
    {InputBufferParam("source", "Source Buffer")},
    {BufferParam("transients", "Transients Buffer"),
     BufferParam("residual", "Residual Buffer")});

template <typename T>
using NRTTransientsClient =
    NRTStreamAdaptor<TransientClient<T>, decltype(NRTTransientParams),
                     NRTTransientParams, 1, 2>;

template <typename T>
using NRTThreadedTransientsClient = NRTThreadingAdaptor<NRTTransientsClient<T>>;

} // namespace client
} // namespace fluid
