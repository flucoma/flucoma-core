#pragma once

#include "clients/common/FluidBaseClient.hpp"
#include "clients/common/MessageSet.hpp"
#include "clients/common/OfflineClient.hpp"
#include "clients/common/ParameterSet.hpp"
#include "clients/common/ParameterTypes.hpp"
#include "clients/common/Result.hpp"
#include "data/FluidDataset.hpp"
#include "data/FluidTensor.hpp"
#include "data/TensorTypes.hpp"
#include <string>

namespace fluid {
namespace client {

enum { kNDims };

auto constexpr CorpusParams = defineParameters
(
    LongParam<Fixed<true>>("nDims", "Dimension size", 0, Min(1))
);


struct addPoint
{
  template <typename T>
  std::string operator()(T &client, std::string label,
                         std::shared_ptr<BufferAdaptor> point)
  {
    Result r = client.addPoint(label, point.get());
    return r.message();
  }
};


struct getPoint
{
  template <typename T>
  std::string operator()(T &client, std::string label,
                         std::shared_ptr<BufferAdaptor> point)
  {
    Result r = client.getPoint(label, point.get());
    return r.message();
  }
};

struct updatePoint
{
  template <typename T>
  std::string operator()(T &client, std::string label,
                         std::shared_ptr<BufferAdaptor> point) {
    Result r = client.updatePoint(label, point.get());
    return r.message();
  }
};

struct deletePoint
{
  template <typename T> std::string operator()(T &client, std::string label) {
    Result r = client.deletePoint(label);
    return r.message();
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

  Result process() { return {}; }

  CorpusClient(ParamSetViewType &p)
      : FluidBaseClient(p), mDataset(get<kNDims>()) {
    mDims = get<kNDims>();
  }

  Result addPoint(string label, BufferAdaptor *data) {
    BufferAdaptor::Access buf(data);
    if (buf.numFrames() < mDims)
      return {Result::Status::kError, "Incorrect point size"};
    FluidTensor<double, 1> point(mDims);
    point = buf.samps(0, mDims, 0);
    mDataset.print();
    if (point.rows() != mDims)
      return {Result::Status::kError, "Wrong number of dimensions"};
    return mDataset.add(label, point)
               ? Result{Result::Status::kOk}
               : Result{Result::Status::kError, "Label already in dataset"};
  }

  Result getPoint(string label, BufferAdaptor *data) {
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

  Result updatePoint(string label, BufferAdaptor *data) {
    BufferAdaptor::Access buf(data);
    if (buf.numFrames() < mDims)
      return {Result::Status::kError, "Incorrect point size"};
    FluidTensor<double, 1> point(mDims);
    point = buf.samps(0, mDims, 0);
    return mDataset.update(label, point)
               ? Result{Result::Status::kOk}
               : Result{Result::Status::kError, "Point not found"};
  }

  Result deletePoint(string label) {
    return mDataset.remove(label)
               ? Result{Result::Status::kOk}
               : Result{Result::Status::kError, "Point not found"};
  }

private:
  FluidDataset<double, string, 1> mDataset;
  FluidTensor<string, 1> mLabels;
  FluidTensor<double, 2> mData;
  int mDims;
};
} // namespace client
} // namespace fluid
