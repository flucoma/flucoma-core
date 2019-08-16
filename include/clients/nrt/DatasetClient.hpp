#pragma once

#include "CorpusClient.hpp"

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

class DatasetClient : public FluidBaseClient, OfflineIn, OfflineOut
{
    enum { kNDims};
  
public:
  using string = std::string;

  template <typename T>
  Result process(FluidContext&) { return {}; }

  FLUID_DECLARE_PARAMS(
    LongParam<Fixed<true>>("nDims", "Dimension size", 1, Min(1))
  );

  DatasetClient(ParamSetViewType &p) : mParams(p), mDataset(get<kNDims>())
  {
    mDims = get<kNDims>();
  }

  MessageResult<void> addPoint(string label, std::shared_ptr<BufferAdaptor> data) {
    if(!data) return mNoBufferError;
    BufferAdaptor::Access buf(data.get());
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

  MessageResult<void> getPoint(string label, std::shared_ptr<BufferAdaptor> data) const{
    if(!data) return mNoBufferError;
    BufferAdaptor::Access buf(data.get());
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

  MessageResult<void> updatePoint(string label, std::shared_ptr<BufferAdaptor> data) {
    if(!data) return mNoBufferError;
    BufferAdaptor::Access buf(data.get());
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

  FLUID_DECLARE_MESSAGES(
    makeMessage("addPoint",&DatasetClient::addPoint),
    makeMessage("getPoint",&DatasetClient::getPoint),
    makeMessage("updatePoint",&DatasetClient::updatePoint),
    makeMessage("deletePoint",&DatasetClient::deletePoint)
  );

private:
  MessageResult<void> mNoBufferError{Result::Status::kError, "No buffer passed"};
  mutable FluidDataset<double, string, 1> mDataset;
  FluidTensor<string, 1> mLabels;
  FluidTensor<double, 2> mData;
  size_t mDims;
};

using NRTThreadedDataset = NRTThreadingAdaptor<ClientWrapper<DatasetClient>>;


} // namespace client
} // namespace fluid
