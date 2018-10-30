#pragma once

#include "clients/common/FluidParams.hpp"
#include "data/FluidBuffers.hpp"
#include "data/FluidTensor.hpp"

#include <sstream> //for ostringstream
#include <string>
#include <unordered_set>
#include <utility> //for std make_pair
#include <vector>  //for containers of params, and for checking things

namespace fluid {
namespace client {

/**
 Integration class for doing NMF filtering and resynthesis
 **/
class BufferComposeClient {
  using desc_type = client::Descriptor;
  using param_type = client::Instance;

public:
  struct ProcessModel {
    size_t frames[2];
    size_t offset[2];
    size_t channels[2];
    size_t channelOffset[2];
    double gain[2];
    size_t dstOffset[2];
    size_t dstChannelOffset[2];
    client::BufferAdaptor *src1 = 0;
    client::BufferAdaptor *src2 = 0;
    client::BufferAdaptor *dst = 0;
  };

  static const std::vector<client::Descriptor> &getParamDescriptors() {
    static std::vector<desc_type> params;
    if (params.empty()) {
      params.emplace_back("src", "First Source Buffer", client::Type::kBuffer);
      params.back().setInstantiation(true);

      params.emplace_back("offsetframes1", "Source 1 Offset",
                          client::Type::kLong);
      params.back().setInstantiation(true).setMin(0).setDefault(0);

      params.emplace_back("numframes1", "Source 1 Frames", client::Type::kLong);
      params.back().setInstantiation(true).setMin(-1).setDefault(-1);

      params.emplace_back("offsetchans1", "Source 1 Channel Offset",
                          client::Type::kLong);
      params.back().setInstantiation(true).setMin(0).setDefault(0);

      params.emplace_back("numchans1", "Source 1 Channels",
                          client::Type::kLong);
      params.back().setInstantiation(true).setMin(-1).setDefault(-1);

      params.emplace_back("src1gain", "Source 1 Gain", client::Type::kFloat);
      params.back().setInstantiation(true).setDefault(1);

      params.emplace_back("src1dstoffset", "Source 1 Destination Offset",
                          client::Type::kLong);
      params.back().setInstantiation(true).setMin(0).setDefault(0);

      params.emplace_back("src1dstchanoffset",
                          "Source 1 Destination Channel Offset",
                          client::Type::kLong);
      params.back().setInstantiation(true).setMin(0).setDefault(0);

      params.emplace_back("src2", "Second Source Buffer",
                          client::Type::kBuffer);
      params.back().setInstantiation(true);

      params.emplace_back("offsetframes2", "Source 2 Offset",
                          client::Type::kLong);
      params.back().setInstantiation(true).setMin(0).setDefault(0);

      params.emplace_back("numframes2", "Source 2 Frames", client::Type::kLong);
      params.back().setInstantiation(true).setMin(-1).setDefault(-1);

      params.emplace_back("offsetchans2", "Source 2 Channel Offset",
                          client::Type::kLong);
      params.back().setInstantiation(true).setMin(0).setDefault(0);

      params.emplace_back("numchans2", "Source 2 Channels",
                          client::Type::kLong);
      params.back().setInstantiation(true).setMin(-1).setDefault(-1);

      params.emplace_back("src2gain", "Source 2 Gain", client::Type::kFloat);
      params.back().setInstantiation(true).setDefault(1);

      params.emplace_back("src2dstoffset", "Source 2 Destination Offset",
                          client::Type::kLong);
      params.back().setInstantiation(true).setMin(0).setDefault(0);

      params.emplace_back("src2dstchanoffset",
                          "Source 2 Destination Channel Offset",
                          client::Type::kLong);
      params.back().setInstantiation(true).setMin(0).setDefault(0);

      params.emplace_back("dstbuf", "Destination Buffer",
                          client::Type::kBuffer);
      params.back().setInstantiation(false);
    }
    return params;
  }

  void newParameterSet() {
    mParams.clear();
    mParams.reserve(getParamDescriptors().size());
    // Note: I'm pretty sure I want auto's copy behaviour here
    for (auto p : getParamDescriptors())
      mParams.emplace_back(client::Instance(p));
  }

  /**
   Go over the supplied parameter values and ensure that they are sensible
   Args
   source buffer [offset numframes offsetchan numchan src gain]
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
    client::BufferAdaptor::Access src1(mParams[0].getBuffer());
    client::BufferAdaptor::Access src2(mParams[8].getBuffer());

    if (!(src1.valid() && src2.valid())) {
      return {false,
              "At least one source buffer doesn't exist or can't be accessed.",
              model};
    }

    if (src1.numFrames() == 0 || src2.numFrames() == 0) {
      return {false, "At least one source buffer is empty", model};
    }

    for (auto &&p : mParams) {
      switch (p.getDescriptor().getType()) {
      case client::Type::kBuffer:
        // If we've been handed a buffer that we're expecting, then it
        // should exist
        if (p.hasChanged() && p.getBuffer()) {
          client::BufferAdaptor::Access b(p.getBuffer());
          if (!b.valid()) {
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

    if (bufCount < 3) {
      return {false, "Expecting three valid buffers", model};
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
        default:
          assert(false && "This should be unreachable");
        }
        return {false, msg.str(), model};
      }
    }

    // Check the size of our buffers
    long srcOffset = client::lookupParam("offsetframes1", mParams).getLong();
    long srcFrames = client::lookupParam("numframes1", mParams).getLong();
    long srcChanOffset = client::lookupParam("offsetchans1", mParams).getLong();
    long srcChans = client::lookupParam("numchans1", mParams).getLong();
    long srcDstOffset = client::lookupParam("src1dstoffset", mParams).getLong();
    long srcDstChanOffset =
        client::lookupParam("src1dstchanoffset", mParams).getLong();

    // We're quite relaxed about asking for more frames than the buffer contains
    // (we'll zero pad)  but it seems reasonable the offset should at least point
    // somewhere in the buffer
    if (srcOffset > src1.numFrames()) {
      return {false, "Source 1 frame offset beyond end of buffer", model};
    }

    // We're quite relaxd about asking for more channel than the buffer contains
    // (we'll go modulo)  but it seems reasonable the offset should at least point
    // somewhere in the buffer
    if (srcChanOffset > src1.numChans()) {
      return {false, "Source 1 channel offset beyond end of buffer", model};
    }

    // At this point, we're happy with the source buffer
    model.src1 = mParams[0].getBuffer();
    model.offset[0] = srcOffset;
    model.frames[0] =
        srcFrames > 0 ? srcFrames : src1.numFrames() - model.offset[0];
    model.channelOffset[0] = srcChanOffset;
    model.channels[0] =
        srcChans > 0 ? srcChans : src1.numChans() - model.channelOffset[0];
    model.gain[0] = client::lookupParam("src1gain", mParams).getFloat();
    model.dstOffset[0] = srcDstOffset;
    model.dstChannelOffset[0] = srcDstChanOffset;

    srcOffset = client::lookupParam("offsetframes2", mParams).getLong();
    srcFrames = client::lookupParam("numframes2", mParams).getLong();
    srcChanOffset = client::lookupParam("offsetchans2", mParams).getLong();
    srcChans = client::lookupParam("numchans2", mParams).getLong();
    srcDstOffset = client::lookupParam("src2dstoffset", mParams).getLong();
    srcDstChanOffset =
        client::lookupParam("src2dstchanoffset", mParams).getLong();

    // We're quite relaxed about asking for more frames than the buffer contains
    // (we'll zero pad)  but it seems reasonable the offset should at least point
    // somewhere in the buffer
    if (srcOffset > src2.numFrames()) {
      return {false, "Source 2 frame offset beyond end of buffer", model};
    }

    // We're quite relaxd about asking for more channel than the buffer contains
    // (we'll go modulo)  but it seems reasonable the offset should at least point
    // somewhere in the buffer
    if (srcChanOffset > src2.numChans()) {
      return {false, "Source 2 channel offset beyond end of buffer", model};
    }

    // At this point, we're happy with the source buffer
    model.src2 = mParams[8].getBuffer();
    model.offset[1] = srcOffset;
    model.frames[1] =
        srcFrames > 0 ? srcFrames : src2.numFrames() - model.offset[1];
    model.channelOffset[1] = srcChanOffset;
    model.channels[1] =
        srcChans > 0 ? srcChans : src2.numChans() - model.channelOffset[1];
    model.gain[1] = client::lookupParam("src2gain", mParams).getFloat();
    model.dstOffset[1] = srcDstOffset;
    model.dstChannelOffset[1] = srcDstChanOffset;

    client::Instance &dstParam = client::lookupParam("dstbuf", mParams);

    client::BufferAdaptor::Access dst(dstParam.getBuffer());
    if (!dst.valid()) {
      return {false, "Destination buffer invalid", model};
    }

    model.dst = dstParam.getBuffer();

    // We made it
    return {true, "Everything is lovely", model};
  }

  // No, you may not  copy this, or move this
  BufferComposeClient(BufferComposeClient &) = delete;
  BufferComposeClient(BufferComposeClient &&) = delete;
  BufferComposeClient operator=(BufferComposeClient &) = delete;
  BufferComposeClient operator=(BufferComposeClient &&) = delete;

  BufferComposeClient() { newParameterSet(); }

  void process(ProcessModel model) {
    // chans and frames are cheked and their max is taken
    std::array<size_t, 2> sumChans;
    std::transform(model.channels, model.channels + 2, model.dstChannelOffset,
                   sumChans.begin(), std::plus<size_t>());
    size_t chans = *std::max_element(sumChans.begin(), sumChans.end());

    std::array<size_t, 2> sumFrames;
    std::transform(model.frames, model.frames + 2, model.dstOffset,
                   sumFrames.begin(), std::plus<size_t>());
    size_t frames = *std::max_element(sumFrames.begin(), sumFrames.end());

    FluidTensor<double, 2> dstData(frames, chans);

    // Access sources inside own scope block, so they'll be unlocked before
    // we need to (possibly) resize the desintation buffer which could
    // (possibly)  also be one of the sources
    {
      client::BufferAdaptor::Access src1(model.src1);
      client::BufferAdaptor::Access src2(model.src2);
      // iterates through the copying of the first source
      for (size_t i = model.dstChannelOffset[0], j = 0; j < model.channels[0];
           ++i, ++j) {
        FluidTensor<double, 1> srcChan(
            src1.samps(model.offset[0], model.frames[0],
                       (model.channelOffset[0] + j) % src1.numChans()));
        srcChan.apply([model](double &x) { x *= model.gain[0]; });
        auto dstChan =
            dstData(fluid::Slice(model.dstOffset[0], model.frames[0]),
                    fluid::Slice(i, 1))
                .col(0);
        dstChan = srcChan;
      }

      // iterates through the copying of the second source and sums it to the
      // dest buff
      for (size_t i = model.dstChannelOffset[1], j = 0; j < model.channels[1];
           ++i, ++j) {
        FluidTensor<double, 1> srcChan(
            src2.samps(model.offset[1], model.frames[1],
                       (model.channelOffset[1] + j) % src2.numChans()));
        srcChan.apply([model](double &x) { x *= model.gain[1]; });
        auto dstChan =
            dstData(fluid::Slice(model.dstOffset[1], model.frames[1]),
                    fluid::Slice(i, 1))
                .col(0);
        dstChan.apply(srcChan, [](double &x, double &y) { x += y; });
      }
    }

    client::BufferAdaptor::Access dst(model.dst);
    //        if(dst.numFrames() < frames || dst.numChans() < chans)
    //        {
    dst.resize(frames, chans, 1);
    //        }

    for (size_t i = 0; i < dst.numChans(); ++i) {
      dst.samps(0, frames, i) = dstData.col(i % dstData.cols());
    }
  }

  std::vector<client::Instance> &getParams() { return mParams; }

private:
  std::vector<client::Instance> mParams;
};
} // namespace client
} // namespace fluid
