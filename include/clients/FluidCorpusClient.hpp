#pragma once

#include "clients/common/FluidBaseClient.hpp"
#include "clients/common/OfflineClient.hpp"
#include "clients/common/ParameterSet.hpp"
#include "clients/common/ParameterTypes.hpp"
#include "clients/common/Result.hpp"
#include "data/FluidTensor.hpp"
#include "data/TensorTypes.hpp"
#include <string>

namespace fluid {
namespace client {

enum { kNDims };

auto constexpr FluidCorpusParams =
    defineParameters(
      LongParam<Fixed<true>>("nDims", "Dimension size", 0, Min(1))
    );

/*template <typename T>
class FluidCorpusClient
    : public FluidBaseClient<decltype(FluidCorpusParams), FluidCorpusParams>,
      OfflineIn,
      OfflineOut {*/
class FluidCorpusClient{

public:
  using string = std::string;

  /*FluidCorpusClient(ParamSetViewType &p)
      : FluidBaseClient(p) {
        mDims = get<kNDims>();
  }*/
  FluidCorpusClient(int nDims){
    mDims = nDims;
  }

  Result addPoint(string label, RealVectorView data) {
    if(data.rows() != mDims) return {Result::Status::kError, "Wrong number of dimensions" };
    auto pos = std::find(mLabels.begin(), mLabels.end(), label);
    if (pos != mLabels.end()) return {Result::Status::kError, "Label already in dataset" };
    mData.resize(mData.rows() + 1,mDims);
    mLabels.resize(mLabels.cols() + 1);
    mLabels(mLabels.rows() - 1) = label;
    mData.row(mData.rows() - 1) = data;
    return {Result::Status::kOk};
  }

  Result getPoint(string label, RealVectorView data) {
    auto pos = std::find(mLabels.begin(), mLabels.end(), label);
    if (pos == mLabels.end())return {Result::Status::kError, "Point not found" };
    data = mData.row(std::distance(mLabels.begin(), pos));
    return {Result::Status::kOk};
  }

  Result updatePoint(string label, RealVectorView data) {
    auto pos = std::find(mLabels.begin(), mLabels.end(), label);
    if (pos == mLabels.end())return {Result::Status::kError, "Point not found" };
    else mData.row(std::distance(mLabels.begin(), pos)) = data;
    return {Result::Status::kOk};
  }

  Result deletePoint(string label){
    // not implemented
    return {Result::Status::kOk};
  }

private:
  FluidTensor<string, 1> mLabels;
  FluidTensor<double, 2> mData;
  int mDims;
};
}
}
