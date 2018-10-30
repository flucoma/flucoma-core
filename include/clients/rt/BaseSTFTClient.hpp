/*
@file: BaseSTFTClient

Base class for real-time STFT processes

*/
#pragma once
#include "algorithms/STFT.hpp"
#include "clients/common/FluidParams.hpp"
#include "clients/common/STFTCheckParams.hpp"
#include "clients/rt/BaseAudioClient.hpp"

#include <complex>
#include <string>
#include <tuple>

namespace fluid {
namespace client {

template <typename T, typename U>
class BaseSTFTClient : public BaseAudioClient<T, U> {

  using data_type = FluidTensorView<T, 2>;
  using complex = FluidTensorView<std::complex<T>, 1>;

public:
  static const std::vector<client::Descriptor> &getParamDescriptors() {
    static std::vector<client::Descriptor> params;
    if (params.size() == 0) {
      BaseAudioClient<T, U>::initParamDescriptors(params);

      params.front().setDefault(1024);
      params[1].setDefault(512);

      params.emplace_back("fftsize", "FFT Size", client::Type::kLong);
      params.back().setMin(-1).setInstantiation(true).setDefault(-1);
    }

    return params;
  }

  BaseSTFTClient() = default;
  BaseSTFTClient(BaseSTFTClient &) = delete;
  BaseSTFTClient operator=(BaseSTFTClient &) = delete;

  BaseSTFTClient(size_t maxWindowSize)
      : // algorithm::STFTCheckParams(windowsize,hopsize,fftsize),
        BaseAudioClient<T, U>(maxWindowSize, 1, 1, 2) {
    newParamSet();
  }

  void reset() {
    size_t winsize = client::lookupParam("winsize", mParams).getLong();
    size_t hopsize = client::lookupParam("hopsize", mParams).getLong();
    size_t fftsize = client::lookupParam("fftsize", mParams).getLong();

    mSTFT = std::unique_ptr<algorithm::STFT>(
        new algorithm::STFT(winsize, fftsize, hopsize));
    mISTFT = std::unique_ptr<algorithm::ISTFT>(
        new algorithm::ISTFT(winsize, fftsize, hopsize));

    normWindow = mSTFT->window();
    normWindow.apply(mISTFT->window(), [](double &x, double &y) { x *= y; });

    BaseAudioClient<T, U>::reset();
  }

  std::tuple<bool, std::string> sanityCheck() {
    //        BaseAudioClient<T,U>::getParams()[0].setLong(client::lookupParam("winsize",
    //        mParams).getLong());
    //        BaseAudioClient<T,U>::getParams()[1].setLong(client::lookupParam("hopsize",
    //        mParams).getLong());

    for (auto &&p : getParams()) {
      std::tuple<bool, client::Instance::RangeErrorType> res = p.checkRange();
      if (!std::get<0>(res)) {
        switch (std::get<1>(res)) {
        case client::Instance::RangeErrorType::kMin:
          return {false, "Parameter below minimum"};
          break;
        case client::Instance::RangeErrorType::kMax:
          return {false, "Parameter above maximum"};
          break;
        }
      }
    }

    std::tuple<bool, std::string> windowCheck =
        BaseAudioClient<T, U>::sanityCheck();
    if (!std::get<0>(windowCheck)) {
      return windowCheck;
    }

    return client::checkFFTArguments(mParams[0], mParams[1], mParams[2]);
  }

  // Here we do an STFT and its inverse
  void process(data_type input, data_type output) override {
    complex spec = mSTFT->processFrame(input.row(0));
    output.row(0) = mISTFT->processFrame(spec);
    output.row(1) = normWindow;
  }
  // Here we gain compensate for the OLA
  void postProcess(data_type output) override {
    output.row(0).apply(output.row(1), [](double &x, double g) {
      if (x) {
        x /= g ? g : 1;
      }
    });
  }

  std::vector<client::Instance> &getParams() override { return mParams; }

private:
  void newParamSet() {
    mParams.clear();
    for (auto &&d : getParamDescriptors())
      mParams.emplace_back(d);
  }

  std::unique_ptr<algorithm::STFT> mSTFT;
  std::unique_ptr<algorithm::ISTFT> mISTFT;
  FluidTensor<T, 1> normWindow;
  std::vector<client::Instance> mParams;
};
} // namespace client
} // namespace fluid
