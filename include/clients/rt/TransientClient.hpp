#pragma once

//#include "BaseAudioClient.hpp"
#include "BufferedProcess.hpp"

#include <algorithms/public/TransientExtraction.hpp>
#include <clients/common/FluidBaseClient.hpp>
#include <clients/common/ParameterConstraints.hpp>
#include <clients/common/ParameterTypes.hpp>
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

auto constexpr TransientParams = std::make_tuple(
    LongParam("order", "Order", 50, Min(20), LowerLimit<kWinSize>(),UpperLimit<kBlockSize>()),
    LongParam("blockSize", "Block Size", 256, Min(100), LowerLimit<kOrder>()),
    LongParam("padding", "Padding", 128, Min(0)),
    FloatParam("skew", "Skew", 0, Min(-10), Max(10)),
    FloatParam("threshFwd", "Forward Threshold", 3, Min(0)),
    FloatParam("threshBack", "Backward Threshold", 1.1, Min(0)),
    LongParam("winSize", "Window Size", 14, Min(0), UpperLimit<kOrder>()),
    LongParam("debounce", "Debounce", 25, Min(0)));

using Param_t = decltype(TransientParams);

template <typename T, typename U = T>
class TransientClient : public FluidBaseClient<Param_t>, public AudioIn, public AudioOut {
  using HostVector = HostVector<U>;

public:
  TransientClient(TransientClient &) = delete;
  TransientClient operator=(TransientClient &) = delete;

  TransientClient() : FluidBaseClient<Param_t>(TransientParams) {
    audioChannelsIn(1);
    audioChannelsOut(2);
  }
  
  void process(std::vector<HostVector>& input,
               std::vector<HostVector>& output) {

   
    if(!input[0].data() || (!output[0].data() && !output[1].data()))
      return;
    
    static constexpr unsigned iterations = 3;
    static constexpr bool refine = false;
    static constexpr double robustFactor = 3.0;

    std::size_t order = get<kOrder>();
    std::size_t blockSize = get<kBlockSize>();
    std::size_t padding = get<kPadding>();
    std::size_t hostVecSize = input[0].size();
    std::size_t maxWin = 2*blockSize + padding;

    if (!mExtractor.get() || startupParamsChanged(order, blockSize, padding) || hostSizeChanged(hostVecSize)) {
      mExtractor.reset(new algorithm::TransientExtraction(
          order, iterations, robustFactor, refine));
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


  
    mExtractor->setDetectionParameters(skew, threshFwd, thresBack, halfWindow,
                                       debounce);
  
    Data<RealMatrix> in(1,hostVecSize);

    in.row(0) = input[0]; //need to convert float->double in some hosts
    mInputBuffer.push(in);
  
    
  
    mBufferedProcess.process(mExtractor->inputSize(), mExtractor->hopSize(), [this](RealMatrix in, RealMatrix out)
    {
      mExtractor->process(in.row(0), out.row(0), out.row(1));
    });
    
    Data<RealMatrix> out(2, hostVecSize);
    mOutputBuffer.pull(out); 

    if(output[0].data()) output[0] = out.row(0);
    if(output[1].data()) output[1] = out.row(1);
  }

  long latency()
  {
    return get<kPadding>() + get<kOrder>() + get<kBlockSize>();
  }

private:

  bool hostSizeChanged(size_t hostSize)
  {
    static size_t size = 0;
    bool res = size != hostSize;
    size = hostSize;
    return res;
  }


  bool startupParamsChanged(size_t order, size_t blocksize, size_t padding) {
    static size_t ord = 0;
    static size_t block = 0;
    static size_t pad = 0;

    bool res = (ord != order) || (block != blocksize) || (pad != padding);

    ord = order;
    block = blocksize;
    pad = padding;

    return res;
  }

  std::unique_ptr<algorithm::TransientExtraction> mExtractor;
  FluidSource<double> mInputBuffer;
  FluidSink<double> mOutputBuffer;
  BufferedProcess mBufferedProcess;
  FluidTensor<T, 1> mTransients;
  FluidTensor<T, 1> mRes;
  //  std::vector<client::Instance> mParams;
};

} // namespace client
} // namespace fluid

