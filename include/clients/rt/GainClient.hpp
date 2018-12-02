/*
 @file GainClient.hpp

 Simple multi-input client, just does modulation of signal 1 by signal 2, or
 sclar gain change

 */
#ifndef fluid_audio_gainclient_h
#define fluid_audio_gainclient_h

#include "BaseAudioClient.hpp"
#include <clients/common/FluidBaseClient.hpp>
#include <clients/common/ParameterConstraints.hpp>
#include <clients/common/ParameterDescriptorList.hpp>

namespace fluid {
namespace client {

enum GainParamTags { kGain, kWindowSize, kMaxSize };

constexpr auto GainParams = std::make_tuple(
    FloatParam("gain","Gain",1.0),
    LongParam("winSize","Window Size", 512, LowerLimit<GainParamTags>(kMaxSize)),
    LongParam("maxSize","", 8192));


using Params_t = decltype(GainParams);

/**!
 @class GainAudioClient

 Inherits core functionality (incl. variable hop size input ansd output
 buffering) from BaseAudioClient<T>"

 **/
template <typename T, typename U = T>
class GainAudioClient : public FluidBaseClient<Params_t>,
                        public BaseAudioClient<T, U> {
  using tensor_type = fluid::FluidTensor<T, 2>;
  using view_type = fluid::FluidTensorView<T, 1>;

public:
  using Signal = typename BaseAudioClient<T, U>::template Signal<U>;
  using AudioSignal = typename BaseAudioClient<T, U>::AudioSignal;
  using ScalarSignal = typename BaseAudioClient<T, U>::ScalarSignal;

  enum class Params { kGain, kMaxWindow, kWindow, kHop };

  //  static auto descriptors()
  //  {
  //
  //    static auto desc = ParameterDescriptorList<Params>();
  //
  //    if(desc.size() == 0)
  //    {
  //      desc.add("gain", "Gain", false,Float_t("gain"));
  //
  //      desc.add("maxwindow","Maximum Window Size", true,Long_t("maxwindow"));
  //      desc.add("window","Window Size", false, Long_t("windowsize"));
  //      desc.add("hopsize","Hop Size", false, Long_t("hopsize"));
  //
  ////
  ///desc.addRelationalConstraint(Params::kWindow,Params::kMaxWindow,std::less_equal<>());
  ////
  ///desc.addRelationalConstraint(Params::kHop,Params::kWindow,std::less_equal<>());
  //    }
  //
  //    return desc;
  //  }

  //  static std::vector<client::Descriptor> getParamDescriptors() {
  //    static std::vector<client::Descriptor> desc;
  //
  //    if (desc.size() == 0) {
  //
  //      desc.emplace_back("gain", "Gain", client::Type::kFloat);
  //      desc.back().setDefault(1);
  //      BaseAudioClient<T, U>::initParamDescriptors(desc);
  //    }
  //    return desc;
  //  }

  /**
   No default instances, no copying
   **/
//  GainAudioClient() = delete;
  GainAudioClient(GainAudioClient &) = delete;
  GainAudioClient operator=(GainAudioClient &) = delete;

  /**
   Construct with a (maximum) chunk size and some input channels
   **/
  GainAudioClient(/*size_t maxChunkSize*/)
      : FluidBaseClient<Params_t>(GainParams), BaseAudioClient<T, U>(
                                                   8192, 2, 1) {
  } //, mParams(descriptors().makeInstances()) {}

  using BaseAudioClient<T, U>::channelsIn;

  /**
   Do the processing: this is the function that descendents of BaseAudioClient
   should override

   Takes a view in

   **/
  template <size_t NIns, size_t NOuts>
  void process(view_type (&data) [NIns], view_type (&output) [NOuts]) {
    // Punishment crashes for the sloppy
    // Data is stored with samples laid out in rows, one channel per row
//    assert(output.cols() == data.cols());
//    assert(data.rows() == channelsIn());

    // Copy the input samples
    output[0] = data[0];
    double g = get<kGain>();
    // Apply gain from the second channel
    output[0].apply([g](double &x) { x *= g; });
  }

  void reset() {
    //    mParams.reset();
    //    mScalarGain = client::lookupParam("gain", mParams).getFloat();
    BaseAudioClient<T, U>::reset();
  }

private:
  //  ParameterInstanceList<Params> mParams;
}; // class
} // namespace client
} // namespace fluid

#endif /* fluid_audio_gainclient_h */

