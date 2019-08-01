#pragma once

#include "clients/common/FluidBaseClient.hpp"
#include "clients/common/OfflineClient.hpp"
#include "clients/common/ParameterSet.hpp"
#include "clients/common/ParameterTypes.hpp"
#include "clients/common/Result.hpp"
#include "data/FluidTensor.hpp"
#include "data/FluidDataset.hpp"
#include "data/TensorTypes.hpp"
#include <string>

namespace fluid {
namespace client {

enum { kNDims };

auto constexpr FluidCorpusParams =
    defineParameters
    (
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
  FluidCorpusClient(int nDims) : mDataset(nDims), mDims(nDims)
  {

  }

  Result addPoint(string label, RealVectorView data)
  {
    if(data.rows() != mDims) return {Result::Status::kError, "Wrong number of dimensions" };
    return mDataset.add(label, data)?
      Result{Result::Status::kOk}:
      Result{Result::Status::kError, "Label already in dataset" };
  }

  Result getPoint(string label, RealVectorView data)
  {
    return mDataset.get(label, data)?
      Result{Result::Status::kOk}:
      Result{Result::Status::kError, "Point not found" };
  }

  Result updatePoint(string label, RealVectorView data)
  {
    return mDataset.update(label, data)?
      Result{Result::Status::kOk}:
      Result{Result::Status::kError, "Point not found" };
  }

  Result deletePoint(string label)
  {
    return mDataset.remove(label)?
      Result{Result::Status::kOk}:
      Result{Result::Status::kError, "Point not found" };
  }

private:

  FluidDataset<double, string, 1> mDataset;
  FluidTensor<string, 1> mLabels;
  FluidTensor<double, 2> mData;
  int mDims;
};
}
}
