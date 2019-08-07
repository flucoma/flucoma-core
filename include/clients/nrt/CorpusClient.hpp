#pragma once

#include "data/FluidDataset.hpp"

#include <clients/common/FluidBaseClient.hpp>
#include <clients/common/MessageSet.hpp>
#include <clients/common/OfflineClient.hpp>
#include <clients/common/ParameterSet.hpp>
#include <clients/common/ParameterTypes.hpp>
#include <clients/common/Result.hpp>
#include <clients/nrt/FluidNRTClientWrapper.hpp>
#include <data/FluidTensor.hpp>
#include <data/TensorTypes.hpp>
#include <string>

namespace fluid {
namespace client {

enum { kNDims };

auto constexpr CorpusParams = defineParameters
(
    LongParam<Fixed<true>>("nDims", "Dimension size", 1, Min(1))
);

struct addPoint
{
  template <typename T>
  MessageResult<void> operator()(T &client, std::string label,
                         std::shared_ptr<BufferAdaptor> point)
  {
    return client.addPoint(label, point.get());
  }
};


struct getPoint
{
  template <typename T>
  MessageResult<void> operator()(T &client, std::string label,
                         std::shared_ptr<BufferAdaptor> point)
  {
    return client.getPoint(label, point.get());
  }
};

struct updatePoint
{
  template <typename T>
  MessageResult<void> operator()(T &client, std::string label,
                         std::shared_ptr<BufferAdaptor> point) {
    return client.updatePoint(label, point.get());
  }
};

struct deletePoint
{
  template <typename T>
  MessageResult<void> operator()(T &client, std::string label) {
    return client.deletePoint(label);
  }
};

auto constexpr CorpusMessages = defineMessages
(
    Message<addPoint>("addPoint"),
    Message<getPoint>("getPoint"),
    Message<updatePoint>("updatePoint"),
    Message<deletePoint>("deletePoint")
);


template <typename T>
class CorpusClient
    : public FluidBaseClient<decltype(CorpusParams), CorpusParams,
                             decltype(CorpusMessages), CorpusMessages>,
      OfflineIn,
      OfflineOut {
  
public:
  using string = std::string;

  Result process(FluidContext&) { return {}; }

  CorpusClient(ParamSetViewType &p)
      : FluidBaseClient(p), mDataset(get<kNDims>()) {
    mDims = get<kNDims>();
  }

  MessageResult<void> addPoint(string label, BufferAdaptor *data) {
    BufferAdaptor::Access buf(data);
    if (buf.numFrames() < mDims)
      return {Result::Status::kError, "Incorrect point size"};
    FluidTensor<double, 1> point(mDims);
    point = buf.samps(0, mDims, 0);
    mDataset.print();
    if (point.rows() != mDims)
      return {Result::Status::kError, "Wrong number of dimensions"};
    return mDataset.add(label, point)
               ? MessageResult<void>{Result::Status::kOk}
               : MessageResult<void>{Result::Status::kError, "Label already in dataset"};
  }

  MessageResult<void> getPoint(string label, BufferAdaptor *data) {
    BufferAdaptor::Access buf(data);
    if (buf.numFrames() < mDims)
      return {Result::Status::kError, "Incorrect point size"};
    FluidTensor<double, 1> point(mDims);
    point = buf.samps(0, mDims, 0);
    bool result = mDataset.get(label, point);
    if (result) {
      buf.samps(0, mDims, 0) = point;
      return {Result::Status::kOk};
    } else {
      return {Result::Status::kError, "Point not found"};
    }
  }

  MessageResult<void> updatePoint(string label, BufferAdaptor *data) {
    BufferAdaptor::Access buf(data);
    if (buf.numFrames() < mDims)
      return {Result::Status::kError, "Incorrect point size"};
    FluidTensor<double, 1> point(mDims);
    point = buf.samps(0, mDims, 0);
    return mDataset.update(label, point)
               ? MessageResult<void>{Result::Status::kOk}
               : MessageResult<void>{Result::Status::kError, "Point not found"};
  }

  MessageResult<void> deletePoint(string label) {
    return mDataset.remove(label)
               ? MessageResult<void>{Result::Status::kOk}
               : MessageResult<void>{Result::Status::kError, "Point not found"};
  }

private:
  FluidDataset<double, string, 1> mDataset;
  FluidTensor<string, 1> mLabels;
  FluidTensor<double, 2> mData;
  size_t mDims;
};

template <typename T>
using NRTThreadedCorpus = NRTThreadingAdaptor<CorpusClient<T>>;


} // namespace client
} // namespace fluid
