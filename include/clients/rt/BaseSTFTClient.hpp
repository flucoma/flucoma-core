/*
@file: BaseSTFTClient

Test class for STFT pass-through
*/
#pragma once

#include "BufferedProcess.hpp"
#include <clients/common/FluidBaseClient.hpp>
#include <clients/common/ParameterTypes.hpp>
#include <clients/common/ParameterConstraints.hpp>
#include <data/FluidTensor.hpp>
#include <algorithms/STFT.hpp>
//#include "clients/common/FluidParams.hpp"
//#include "clients/common/STFTCheckParams.hpp"
#include "clients/rt/BaseAudioClient.hpp"

#include <complex>
#include <string>
#include <tuple>
#include <vector> 

namespace fluid {
namespace client {

enum STFTParamIndex {kWinsize,kHopsize,kFFTSize,kMaxWin};

auto constexpr STFTParams = std::make_tuple(
  LongParam("winSize", "Window Size", 1024, Min(4)),
  LongParam("hopSize", "Hop Size", 512),
  LongParam("fftSize", "FFT Size", -1, LowerLimit<kWinsize>(),PowerOfTwo()),
  LongParam("maxWinSize", "Maxiumm Window Size", 16384)
);


using Param_t = decltype(STFTParams);

template <typename T, typename U = T>
class BaseSTFTClient : public FluidBaseClient<Param_t> {
  
  using View = FluidTensorView<T, 1>;
  using ComplexView = FluidTensorView<std::complex<T>, 1>;

public:
//  static const std::vector<client::Descriptor> &getParamDescriptors() {
//    static std::vector<client::Descriptor> params;
//    if (params.size() == 0) {
//      BaseAudioClient<T, U>::initParamDescriptors(params);
//
//      params.front().setDefault(1024);
//      params[1].setDefault(512);
//
//      params.emplace_back("fftsize", "FFT Size", client::Type::kLong);
//      params.back().setMin(-1).setInstantiation(true).setDefault(-1);
//    }
//
//    return params;
//  }

  BaseSTFTClient(BaseSTFTClient &) = delete;
  BaseSTFTClient operator=(BaseSTFTClient &) = delete;

  BaseSTFTClient(): FluidBaseClient<Param_t>(STFTParams){
    audioChannelsIn(1);
    audioChannelsOut(1);
  }

//  void reset() {
//    size_t winsize = client::lookupParam("winsize", mParams).getLong();
//    size_t hopsize = client::lookupParam("hopsize", mParams).getLong();
//    size_t fftsize = client::lookupParam("fftsize", mParams).getLong();
//
//    mSTFT = std::unique_ptr<algorithm::STFT>(
//        new algorithm::STFT(winsize, fftsize, hopsize));
//    mISTFT = std::unique_ptr<algorithm::ISTFT>(
//        new algorithm::ISTFT(winsize, fftsize, hopsize));
//
//    normWindow = mSTFT->window();
//    normWindow.apply(mISTFT->window(), [](double &x, double &y) { x *= y; });
//
//    BaseAudioClient<T, U>::reset();
//  }

//  std::tuple<bool, std::string> sanityCheck() {
//    //        BaseAudioClient<T,U>::getParams()[0].setLong(client::lookupParam("winsize",
//    //        mParams).getLong());
//    //        BaseAudioClient<T,U>::getParams()[1].setLong(client::lookupParam("hopsize",
//    //        mParams).getLong());
//
//    for (auto &&p : getParams()) {
//      std::tuple<bool, client::Instance::RangeErrorType> res = p.checkRange();
//      if (!std::get<0>(res)) {
//        switch (std::get<1>(res)) {
//        case client::Instance::RangeErrorType::kMin:
//          return {false, "Parameter below minimum"};
//          break;
//        case client::Instance::RangeErrorType::kMax:
//          return {false, "Parameter above maximum"};
//          break;
//        }
//      }
//    }
//
//    std::tuple<bool, std::string> windowCheck =
//        BaseAudioClient<T, U>::sanityCheck();
//    if (!std::get<0>(windowCheck)) {
//      return windowCheck;
//    }
//
//    return client::checkFFTArguments(mParams[0], mParams[1], mParams[2]);
//  }

  // Here we do an STFT and its inverse
  void process(std::vector<View>& input, std::vector<View> output) {
    
    mInputBuffer.push(input[0]);

    mBufferedProcess.maxWindowSize(get<kMaxWin>()); 
    mBufferedProcess.setBuffers(mInputBuffer, mOutputBuffer);
    mBufferedProcess.hostSize(input.rows()); //safe?
    mBufferedProcess.process(get<kWinsize>(), get(kHopsize),[this](View& in, View& out) {
        auto spec = mSTFT->processFrame(in);
        out = mISTFT->processFrame(spec);
    } );
    
    

//    output.row(1) = normWindow;
  }
  // Here we gain compensate for the OLA
//  void postProcess(data_type output) override {
//    output.row(0).apply(output.row(1), [](double &x, double g) {
//      if (x) {
//        x /= g ? g : 1;
//      }
//    });
//  }
//
//  std::vector<client::Instance> &getParams() override { return mParams; }

//private:
//  void newParamSet() {
//    mParams.clear();
//    for (auto &&d : getParamDescriptors())
//      mParams.emplace_back(d);
//  }

  std::unique_ptr<algorithm::STFT> mSTFT;
  std::unique_ptr<algorithm::ISTFT> mISTFT;
  FluidTensor<T, 1> normWindow;
  FluidSource<double> mInputBuffer;
  FluidSink<double> mOutputBuffer;
  BufferedProcess mBufferedProcess;
//  std::vector<client::Instance> mParams;
};
} // namespace client
} // namespace fluid
