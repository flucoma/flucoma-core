#pragma once

#include "algorithms/NMF.hpp"
#include "algorithms/RatioMask.hpp"
#include "algorithms/STFT.hpp"
#include "clients/common/FluidParams.hpp"
#include "clients/common/STFTCheckParams.hpp"
#include "data/FluidBuffers.hpp"
#include "data/FluidTensor.hpp"

#include <algorithm> //for max_element
#include <sstream>   //for ostringstream
#include <string>
#include <unordered_set>
#include <utility> //for std make_pair
#include <vector>  //for containers of params, and for checking things

using fluid::FluidTensor;
using fluid::algorithm::ISTFT;
using fluid::algorithm::NMF;
using fluid::algorithm::STFT;
using fluid::algorithm::Spectrogram;

namespace fluid {
namespace client {

/**
 Integration class for doing NMF filtering and resynthesis
 **/
class NMFClient {
  using desc_type = client::Descriptor;
  using param_type = client::Instance;

public:
  struct ProcessModel {
    bool fixDictionaries;
    bool fixActivations;
    bool seedDictionaries;
    bool seedActivations;
    bool returnDictionaries;
    bool returnActivations;

    size_t windowSize;
    size_t hopSize;
    size_t fftSize;

    size_t rank;
    size_t iterations;
    bool resynthesise;

    size_t frames;
    size_t offset;

    size_t channels;
    size_t channelOffset;

    client::BufferAdaptor *src = 0;
    client::BufferAdaptor *resynth = 0;
    client::BufferAdaptor *dict = 0;
    client::BufferAdaptor *act = 0;
  };

  static const std::vector<client::Descriptor> &getParamDescriptors() {
    static std::vector<desc_type> params;
    if (params.empty()) {
      params.emplace_back(
          desc_type{"src", "Source Buffer", client::Type::kBuffer});
      params.back().setInstantiation(true);

      params.emplace_back(
          desc_type{"offsetframes", "Source Offset", client::Type::kLong});
      params.back().setInstantiation(true).setMin(0).setDefault(0);

      params.emplace_back(
          desc_type{"numframes", "Source Frames", client::Type::kLong});
      params.back().setInstantiation(true).setMin(-1).setDefault(-1);

      params.emplace_back(desc_type{"offsetchans", "Source Channel Offset",
                                    client::Type::kLong});
      params.back().setInstantiation(true).setMin(0).setDefault(0);

      params.emplace_back(
          desc_type{"numchans", "Source Channels", client::Type::kLong});
      params.back().setInstantiation(true).setMin(-1).setDefault(-1);

      params.emplace_back(
          desc_type{"resynthbuf", "Resynthesis Buffer", client::Type::kBuffer});
      params.back().setInstantiation(false);

      params.emplace_back(
          desc_type{"filterbuf", "Filters Buffer", client::Type::kBuffer});
      params.back().setInstantiation(false);

      params.emplace_back(
          desc_type{"filterupdate", "Filter Update", client::Type::kLong});
      params.back()
          .setInstantiation(false)
          .setMin(0)
          .setMax(2)
          .setDefault(0)
          .setInstantiation(false);

      params.emplace_back(
          desc_type{"envbuf", "Envelopes Buffer", client::Type::kBuffer});
      params.back().setInstantiation(false).setInstantiation(false);

      params.emplace_back(
          desc_type{"envupdate", "Activation Update", client::Type::kLong});
      params.back()
          .setInstantiation(false)
          .setMin(0)
          .setMax(2)
          .setDefault(0)
          .setInstantiation(false);

      params.emplace_back(desc_type{"rank", "Rank", client::Type::kLong});
      params.back().setMin(1).setDefault(1).setInstantiation(false);

      params.emplace_back(
          desc_type{"iterations", "Iterations", client::Type::kLong});
      params.back()
          .setInstantiation(false)
          .setMin(1)
          .setDefault(100)
          .setInstantiation(false);

      params.emplace_back(
          desc_type{"winsize", "Window Size", client::Type::kLong});
      params.back().setMin(4).setDefault(1024).setInstantiation(false);

      params.emplace_back(
          desc_type{"hopsize", "Hop Size", client::Type::kLong});
      params.back().setMin(1).setDefault(256).setInstantiation(false);

      params.emplace_back(
          desc_type{"fftsize", "FFT Size", client::Type::kLong});
      params.back().setMin(-1).setDefault(-1).setInstantiation(false);
    }
    return params;
  }

  /**
   Go over the supplied parameter values and ensure that they are sensible
   No hygiene checking of buffers is done here (like whether they exist). It
  needs to be done in the host code, until I've worked out something cleverer
  **/
  std::tuple<bool, std::string, ProcessModel> sanityCheck() {
    ProcessModel model;
    const std::vector<client::Descriptor> &desc = getParamDescriptors();
    // First, let's make sure that we have a complete of parameters of the right
    // sort
    bool sensible =
        std::equal(mParams.begin(), mParams.end(), desc.begin(),
                   [](const param_type &i, const client::Descriptor &d) {
                     return i.getDescriptor() == d;
                   });

    if (!sensible || (desc.size() != mParams.size())) {
      return {
          false,
          "Invalid params passed. Were these generated with newParameterSet()?",
          model};
    }

    size_t bufCount = 0;
    std::unordered_set<client::BufferAdaptor *> uniqueBuffers;
    // First round of buffer checks
    // Source buffer is mandatory, and should exist

    client::BufferAdaptor::Access src(mParams[0].getBuffer());

    if (!src.valid()) {
      return {false, "Source buffer doesn't exist or can't be accessed.",
              model};
    }

    for (auto &&p : mParams) {
      switch (p.getDescriptor().getType()) {
      case client::Type::kBuffer:
        // If we've been handed a buffer that we're expecting, then it
        // should exist
        if (p.hasChanged() && p.getBuffer()) {
          client::BufferAdaptor::Access b(p.getBuffer());
          if (!b.valid())

          {
            std::ostringstream ss;
            ss << "Buffer given for " << p.getDescriptor().getName()
               << " doesn't exist.";

            return {false, ss.str(), model};
          }
          ++bufCount;
          uniqueBuffers.insert(p.getBuffer());
        }
      default:
        continue;
      }
    }

    if (bufCount < 2) {
      return {false, "Expecting at least two valid buffers", model};
    }

    if (bufCount > uniqueBuffers.size()) {
      return {false,
              "One or more buffers are the same. They all need to be distinct",
              model};
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
          msg << " value below minimum(" << d.getMin() << ")";
          break;
        case client::Instance::RangeErrorType::kMax:
          msg << " value above maximum(" << d.getMin() << ")";
        default:
          assert(false && "This should be unreachable");
        }
        return {false, msg.str(), model};
      }
    }

    // Now make sense of the overwrite parameter
    // if 0 (normal) then we expect to reallocate both dicts and acts and init
    // randomly  if 1 (seeding) then the buffer should be present, allocated, and
    // will mutate  if 2 (matching) then the buffer should be present, allocated
    // and won't mutate  having both == 2 makes no sense

    client::Instance &dictUpdateRule =
        client::lookupParam("filterupdate", mParams);
    client::Instance &actUpdateRule = client::lookupParam("envupdate", mParams);

    if (dictUpdateRule.getLong() == 2 && actUpdateRule.getLong() == 2) {
      return {false,
              "It makes no sense to update neither the dictionaries nor the "
              "activaitons",
              model};
    }

    model.seedDictionaries = (dictUpdateRule.getLong() > 0);
    model.fixDictionaries = (dictUpdateRule.getLong() == 2);
    model.seedActivations = (actUpdateRule.getLong() > 0);
    model.fixActivations = (actUpdateRule.getLong() == 2);

    // Check the size of our buffers

    long srcOffset = client::lookupParam("offsetframes", mParams).getLong();
    long srcFrames = client::lookupParam("numframes", mParams).getLong();
    long srcChanOffset = client::lookupParam("offsetchans", mParams).getLong();
    long srcChans = client::lookupParam("numchans", mParams).getLong();

    // Ensure that the source buffer can deliver
    if (srcFrames > 0 ? (src.numFrames() < (srcOffset + srcFrames))
                      : (src.numFrames() < srcOffset)) {
      return {false,
              "Source buffer not long enough for given offset and frame count",
              model};
    }

    if ((srcChans > 0) ? (src.numChans() < (srcChanOffset + srcChans))
                       : (src.numChans() < srcChanOffset)) {
      return {false,
              "Source buffer doesn't have enough channels for given offset and "
              "channel count",
              model};
    }

    // At this point, we're happy with the source buffer
    model.src = mParams[0].getBuffer();
    model.offset = srcOffset;
    model.frames = srcFrames > 0 ? srcFrames : src.numFrames() - model.offset;
    model.channelOffset = srcChanOffset;
    model.channels =
        srcChans > 0 ? srcChans : src.numChans() - model.channelOffset;

    // Check the FFT args

    client::Instance &windowSize = lookupParam("winsize", mParams);
    client::Instance &fftSize = lookupParam("fftsize", mParams);
    client::Instance &hopSize = lookupParam("hopsize", mParams);

    auto fftOk = client::checkFFTArguments(windowSize, hopSize, fftSize);

    if (!std::get<0>(fftOk)) {
      // Glue the fft error message together with our model
      return std::tuple_cat(fftOk, std::tie(model));
    }

    model.fftSize = fftSize.getLong();
    model.windowSize = windowSize.getLong();
    model.hopSize = hopSize.getLong();
    model.rank = client::lookupParam("rank", mParams).getLong();
    model.iterations = client::lookupParam("iterations", mParams).getLong();

    client::Instance &resynth = client::lookupParam("resynthbuf", mParams);
    client::BufferAdaptor::Access resynthBuf(resynth.getBuffer());

    if (resynth.hasChanged() && (!resynth.getBuffer() || !resynthBuf.valid())) {
      return {false, "Invalid resynthesis buffer supplied", model};
    }

    model.resynthesise = resynth.hasChanged() && resynthBuf.valid();
    if (model.resynthesise)
      model.resynth = resynth.getBuffer();

    client::Instance &dict = client::lookupParam("filterbuf", mParams);
    client::BufferAdaptor::Access dictBuf(dict.getBuffer());

    if (dict.hasChanged() && (!dict.getBuffer() || !dictBuf.valid())) {
      return {false, "Invalid filters buffer supplied", model};
    }
    if (model.fixDictionaries || model.seedDictionaries) {
      if (!dict.hasChanged())
        return {false,
                "No dictionary buffer given, but one needed for seeding or "
                "matching",
                model};
      // Prepared Dictionary buffer needs to be (fftSize/2 + 1) by (rank *
      // srcChans)
      if (dictBuf.numFrames() != (model.fftSize / 2) + 1 ||
          dictBuf.numChans() != model.rank * model.channels)
        return {false,
                "Pre-prepared dictionary buffer must be [(FFTSize / 2) + 1] "
                "frames long, and have [rank] * [channels] channels",
                model};
    }
    model.returnDictionaries = dict.hasChanged();
    model.dict = dict.getBuffer();

    client::Instance &act = client::lookupParam("envbuf", mParams);
    client::BufferAdaptor::Access actBuf(act.getBuffer());

    if (act.hasChanged() && (!act.getBuffer() && !actBuf.valid())) {
      return {false, "Invalid envelope buffer supplied", model};
    }
    if (model.fixActivations || model.seedActivations) {
      if (!act.hasChanged())
        return {false,
                "No dictionary buffer given, but one needed for seeding or "
                "matching",
                model};

      // Prepared activation buffer needs to be (src Frames / hop size + 1) by
      // (rank * srcChans)
      if (actBuf.numFrames() != (model.frames / model.hopSize) + 1 ||
          actBuf.numChans() != model.rank * model.channels) {
        return {false,
                "Pre-prepared activation buffer must be [(num samples / hop "
                "size) + 1] frames long, and have [rank] * [channels] channels",
                model};
      }
    }
    model.returnActivations = act.hasChanged();
    model.act = act.getBuffer();

    // We made it
    return {true, "Everything is lovely", model};
  }

  NMFClient() { newParameterSet(); };
  // no copy this, nor move this

  NMFClient(NMFClient &) = delete;
  NMFClient(NMFClient &&) = delete;
  NMFClient operator=(NMFClient &) = delete;
  NMFClient operator=(NMFClient &&) = delete;

  /**
   You may constrct one by supplying some senisble numbers here
   rank: NMF rank
   iterations: max nmf iterations
   fft_size: power 2 pls
//       **/
  //        NMFClient(ProcessModel model):
  //        mArguments(model)

  ~NMFClient() = default;

  //      /**
  //       Call this before pushing / processing / pulling
  //       to prepare buffers to correct size
  //       **/
  //      void setSourceSize(size_t source_frames)
  //      {
  //        mSource.set_host_buffer_size(source_frames);
  //        mSinkResynth.set_host_buffer_size(source_frames);
  //      }

  /***
   Take some data, NMF it
   ***/
  void process(ProcessModel model) {

    //        mArguments = model;

    client::BufferAdaptor::Access src(model.src);
    client::BufferAdaptor::Access dict(model.dict);
    client::BufferAdaptor::Access act(model.act);
    client::BufferAdaptor::Access resynth(model.resynth);

    if (model.resynthesise)
      resynth.resize(model.frames, model.channels, model.rank);
    if (model.returnDictionaries && !model.seedDictionaries)
      dict.resize(model.fftSize / 2 + 1, model.channels, model.rank);
    if (model.returnActivations && !model.seedActivations)
      act.resize((model.frames / model.hopSize) + 1, model.channels,
                 model.rank);

    algorithm::STFT stft(model.windowSize, model.fftSize, model.hopSize);
    // Copy input buffer
    //        RealMatrix sourceData(src.samps(model.offset, model.frames,
    //        model.channelOffset, model.channels));
    // TODO: get rid of need for this
    // Either: do the whole process loop here via process_frame
    // Or: change sig of stft.process() to take a view
    FluidTensor<double, 1> tmp(model.frames);
    for (size_t i = 0; i < model.channels; ++i) {
      //          tmp = sourceData.col(i);
      tmp = src.samps(model.offset, model.frames, model.channelOffset + i);
      algorithm::Spectrogram spec = stft.process(tmp);

      // For multichannel dictionaries, seed data could be all over the place,
      // so we'll build it up by hand :-/
      FluidTensor<double, 2> seedDicts(0, 0);
      FluidTensor<double, 2> seedActs(0, 0);

      if (model.seedDictionaries)
        seedDicts.resize(model.rank, model.fftSize / 2 + 1);
      if (model.seedActivations)
        seedActs.resize((model.frames / model.hopSize) + 1, model.rank);

      for (size_t j = 0; j < model.rank; ++j) {
        if (model.seedDictionaries)
          seedDicts.row(j) = dict.samps(i, j);
        if (model.seedActivations)
          seedActs.col(j) = act.samps(i, j);
      }

      algorithm::NMF nmf(model.rank, model.iterations, !model.fixDictionaries,
                         !model.fixActivations);
      algorithm::NMFModel m =
          nmf.process(spec.getMagnitude(), seedDicts, seedActs);

      // Write W?
      if (model.returnDictionaries && !model.seedDictionaries) {
        auto dictionaries = m.getW();
        for (size_t j = 0; j < model.rank; ++j)
          dict.samps(i, j) = dictionaries.row(j);
        // model.dict->col(j + (i * model.rank)) = dictionaries.col(j);
      }

      // Write H? Need to normalise also
      if (model.returnActivations && !model.seedActivations) {
        auto activations = m.getH();
        double maxH = *std::max_element(activations.begin(), activations.end());
        double scale = 1. / (maxH);
        for (size_t j = 0; j < model.rank; ++j) {
          auto acts = act.samps(i, j);
          acts = activations.col(j);
          acts.apply([scale](float &x) { x *= scale; });
        }
      }

      if (model.resynthesise) {
        algorithm::RatioMask mask(m.getMixEstimate(), 1);
        algorithm::ISTFT istft(model.windowSize, model.fftSize, model.hopSize);
        for (size_t j = 0; j < model.rank; ++j) {
          auto estimate = m.getEstimate(j);
          algorithm::Spectrogram result(mask.process(spec.mData, estimate));
          auto audio = istft.process(result);
          resynth.samps(i, j) = audio(fluid::Slice(0, model.frames));
        }
      }
    }
  }

  std::vector<client::Instance> &getParams() { return mParams; }

private:
  //      ProcessModel mArguments;
  fluid::algorithm::NMFModel mModel;
  FluidTensor<double, 2> mAudioBuffers;

  void newParameterSet() {
    mParams.clear();
    // Note: I'm pretty sure I want auto's copy behaviour here
    for (auto p : getParamDescriptors())
      mParams.emplace_back(client::Instance(p));
  }

  std::vector<client::Instance> mParams;
};
} // namespace client
} // namespace fluid
