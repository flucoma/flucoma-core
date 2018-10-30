#pragma once

#include "BaseAudioClient.hpp"
#include "algorithms/RTSineExtraction.hpp"
#include "algorithms/STFT.hpp"
#include "clients/common/FluidParams.hpp"
#include "clients/common/STFTCheckParams.hpp"

#include <complex>
#include <string>
#include <tuple>

namespace fluid {
namespace client {
template <typename T, typename U>
class SinesClient : public client::BaseAudioClient<T, U> {
  using data_type = FluidTensorView<T, 2>;
  using complex = FluidTensorView<std::complex<T>, 1>;

public:
  static const std::vector<client::Descriptor> &getParamDescriptors() {
    static std::vector<client::Descriptor> params;
    if (params.size() == 0) {
      params.emplace_back("bandwidth", "Bandwidth", client::Type::kLong);
      params.back().setMin(1).setDefault(76).setInstantiation(true);

      params.emplace_back("threshold", "Threshold", client::Type::kFloat);
      params.back().setMin(0).setMax(1).setDefault(0.7).setInstantiation(false);

      params.emplace_back("mintracklen", "Min Track Length",
                          client::Type::kLong);
      params.back().setMin(0).setDefault(15).setInstantiation(false);

      params.emplace_back("magweight", "Magnitude Weight",
                          client::Type::kFloat);
      params.back().setMin(0).setMax(1).setDefault(0.1).setInstantiation(false);

      params.emplace_back("freqweight", "Frequency Weight",
                          client::Type::kFloat);
      params.back().setMin(0).setMax(1).setDefault(1).setInstantiation(false);

      //        params.emplace_back("winsize","Window Size",
      //        client::Type::Long); params.back().setMin(4).setDefault(4096);
      //
      //        params.emplace_back("hopsize","Hop Size", client::Type::Long);
      //        params.back().setMin(1).setDefault(1024);
      client::BaseAudioClient<T, U>::initParamDescriptors(params);

      params[params.size() - 2].setInstantiation(true); // winsize
      params.back().setInstantiation(true);             // hopsize

      params.emplace_back("fftsize", "FFT Size", client::Type::kLong);
      params.back().setMin(-1).setDefault(8192).setInstantiation(true);
    }

    return params;
  }

  SinesClient() = default;
  SinesClient(SinesClient &) = delete;
  SinesClient operator=(SinesClient &) = delete;

  SinesClient(size_t maxWindowSize)
      : // algorithm::STFTCheckParams(windowsize,hopsize,fftsize),
        client::BaseAudioClient<T, U>(maxWindowSize, 1, 2, 3) {
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

    size_t bandwidth = client::lookupParam("bandwidth", getParams()).getLong();
    double threshold = client::lookupParam("threshold", getParams()).getFloat();
    size_t mintracklen =
        client::lookupParam("mintracklen", getParams()).getLong();
    double magweight = client::lookupParam("magweight", getParams()).getFloat();
    double freqweight =
        client::lookupParam("freqweight", getParams()).getFloat();

    //      (int windowSize, int fftSize, int hopSize, int bandwidth,
    //       double threshold, int minTrackLength, double magWeight,
    //       double freqWeight)

    mSinesExtractor = std::unique_ptr<algorithm::RTSineExtraction>(
        new algorithm::RTSineExtraction(winsize, fftsize, hopsize, bandwidth,
                                        threshold, mintracklen, magweight,
                                        freqweight));

    mNormWindow = mSTFT->window();
    mNormWindow.apply(mISTFT->window(), [](double &x, double &y) { x *= y; });

    mSeparatedSpectra.resize(fftsize / 2 + 1, 2);
    mSines.resize(fftsize / 2 + 1);
    mRes.resize(fftsize / 2 + 1);
    client::BaseAudioClient<T, U>::reset();
  }

  std::tuple<bool, std::string> sanityCheck() {
    //        BaseAudioClient<T,U>::getParams()[0].setLong(client::lookupParam("winsize",
    //        mParams).getLong());
    //        BaseAudioClient<T,U>::getParams()[1].setLong(client::lookupParam("hopsize",
    //        mParams).getLong());
    const std::vector<client::Descriptor> &desc = getParamDescriptors();
    // First, let's make sure that we have a complete of parameters of the right
    // sort
    bool sensible =
        std::equal(mParams.begin(), mParams.end(), desc.begin(),
                   [](const client::Instance &i, const client::Descriptor &d) {
                     return i.getDescriptor() == d;
                   });

    if (!sensible || (desc.size() != mParams.size())) {
      return {false, "Invalid params passed. Were these generated with "
                     "newParameterSet()?"};
    }

    // Now scan everything for range, until we hit a problem
    // TODO Factor into client::instance
    for (auto &&p : mParams) {
      client::Descriptor d = p.getDescriptor();
      bool rangeOk;
      client::Instance::RangeErrorType errorType;
      std::tie(rangeOk, errorType) = p.checkRange();
      if (!rangeOk) {
        std::ostringstream msg;
        msg << "Parameter " << d.getName();
        switch (errorType) {
        case client::Instance::RangeErrorType::kMin:
          msg << " value below minimum (" << d.getMin() << ")";
          break;
        case client::Instance::RangeErrorType::kMax:
          msg << " value above maximum (" << d.getMin() << ")";
          break;
        default:
          assert(false && "This should be unreachable");
        }
        return {false, msg.str()};
      }
    }

    std::tuple<bool, std::string> windowCheck =
        client::BaseAudioClient<T, U>::sanityCheck();
    if (!std::get<0>(windowCheck)) {
      return windowCheck;
    }

    client::Instance &winSize = client::lookupParam("winsize", getParams());
    client::Instance &hopSize = client::lookupParam("hopsize", getParams());
    client::Instance &fftSize = client::lookupParam("fftsize", getParams());

    return client::checkFFTArguments(winSize, hopSize, fftSize);

    return {true, "Groovy"};
  }

  // Here we do an STFT and its inverse
  void process(data_type input, data_type output) override {
    complex spec = mSTFT->processFrame(input.row(0));

    mSinesExtractor->setThreshold(
        client::lookupParam("threshold", getParams()).getFloat());
    mSinesExtractor->setMagWeight(
        client::lookupParam("magweight", getParams()).getFloat());
    mSinesExtractor->setFreqWeight(
        client::lookupParam("freqweight", getParams()).getFloat());
    mSinesExtractor->setMinTrackLength(
        client::lookupParam("mintracklen", getParams()).getLong());

    mSinesExtractor->processFrame(spec, mSeparatedSpectra);

    mSines = mSeparatedSpectra.col(0);
    mRes = mSeparatedSpectra.col(1);
    //      mSeparatedSpectra.row(0) = spec;
    //      mSeparatedSpectra.row(1) = spec;

    output.row(0) = mISTFT->processFrame(mSines);
    output.row(1) = mISTFT->processFrame(mRes);
    output.row(2) = mNormWindow;
  }
  // Here we gain compensate for the OLA
  void postProcess(data_type output) override {
    output.row(0).apply(output.row(2), [](double &x, double g) {
      if (x) {
        x /= g ? g : 1;
      }
    });
    output.row(1).apply(output.row(2), [](double &x, double g) {
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
  std::unique_ptr<algorithm::RTSineExtraction> mSinesExtractor;
  std::unique_ptr<algorithm::ISTFT> mISTFT;
  FluidTensor<T, 1> mNormWindow;
  FluidTensor<std::complex<T>, 2> mSeparatedSpectra;
  FluidTensor<std::complex<T>, 1> mSines;
  FluidTensor<std::complex<T>, 1> mRes;
  std::vector<client::Instance> mParams;
};

} // namespace client
} // namespace fluid
