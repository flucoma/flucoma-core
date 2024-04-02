/*
Part of the Fluid Corpus Manipulation Project (http://www.flucoma.org/)
Copyright University of Huddersfield.
Licensed under the BSD-3 License.
See license.md file in the project root for full license information.
This project has received funding from the European Research Council (ERC)
under the European Union’s Horizon 2020 research and innovation programme
(grant agreement No 725899).
*/

#pragma once
#include "DataSetClient.hpp"
#include "NRTClient.hpp"
#include "../../algorithms/public/DataSetQuery.hpp"

namespace fluid {
namespace client {
namespace datasetquery {

enum { kName };

constexpr auto DataSetQueryParams = defineParameters();

class DataSetQueryClient : public FluidBaseClient, OfflineIn, OfflineOut
{
public:
  using string = std::string;
  using DataSet = FluidDataSet<string, double, 1>;

  template <typename T>
  Result process(FluidContext&)
  {
    return {};
  }

  using ParamDescType = decltype(DataSetQueryParams);
  using ParamSetViewType = ParameterSetView<ParamDescType>;
  std::reference_wrapper<ParamSetViewType> mParams;

  void setParams(ParamSetViewType& p) { mParams = p; }

  template <size_t N>
  auto& get() const
  {
    return mParams.get().template get<N>();
  }

  static constexpr auto& getParameterDescriptors()
  {
    return DataSetQueryParams;
  }

  DataSetQueryClient(ParamSetViewType& p,  FluidContext&) : mParams(p) {}

  MessageResult<void> addColumn(index column)
  {
    if (column < 0) return Error("invalid index");
    mAlgorithm.addColumn(column);
    return OK();
  }

  MessageResult<void> addRange(index from, index count)
  {
    if (from < 0 || count <= 0) return Error("invalid range");
    mAlgorithm.addRange(from, count);
    return OK();
  }

  MessageResult<void> filter(index column, string comparison, double value)
  {
    if (column < 0) return Error("invalid index");
    if (mAlgorithm.hasAndConditions()) return Error("Filter already set");
    bool result = mAlgorithm.addCondition(column, comparison, value, true);
    if (!result)
      return Error("invalid filter");
    else
      return OK();
  }

  MessageResult<void> andFilter(index column, string comparison, double value)
  {
    if (column < 0) return Error("invalid index");
    if (!mAlgorithm.hasAndConditions()) return Error("Add a filter first");
    bool result = mAlgorithm.addCondition(column, comparison, value, true);
    if (!result)
      return Error("invalid filter");
    else
      return OK();
  }

  MessageResult<void> orFilter(index column, string comparison, double value)
  {
    if (column < 0) return Error("invalid index");
    if (!mAlgorithm.hasAndConditions()) return Error("Add a filter first");
    bool result = mAlgorithm.addCondition(column, comparison, value, false);
    if (!result)
      return Error("invalid filter");
    else
      return OK();
  }


  MessageResult<void> transform(InputDataSetClientRef sourceClient,
                                DataSetClientRef destClient)
  {
    if (mAlgorithm.numColumns() <= 0) return Error("No columns");
    auto srcPtr = sourceClient.get().lock();
    auto destPtr = destClient.get().lock();
    if (!srcPtr || !destPtr) return Error(NoDataSet);
    auto src = srcPtr->getDataSet();
    if (src.size() == 0) return Error(EmptyDataSet);
    if (src.pointSize() <= mAlgorithm.maxColumn()) return Error(WrongPointSize);
    index   resultSize = mAlgorithm.numColumns();
    DataSet empty;
    DataSet result(resultSize);
    mAlgorithm.process(src, empty, result);
    destPtr->setDataSet(result);
    return OK();
  }

  MessageResult<void> transformJoin(InputDataSetClientRef source1Client,
                                    InputDataSetClientRef source2Client,
                                    DataSetClientRef destClient)
  {
    auto src1Ptr = source1Client.get().lock();
    auto src2Ptr = source2Client.get().lock();
    auto destPtr = destClient.get().lock();
    if (!src1Ptr || !src2Ptr || !destPtr) return Error(NoDataSet);
    auto src1 = src1Ptr->getDataSet();
    if (src1.size() == 0) return Error(EmptyDataSet);
    if (src1.pointSize() <= mAlgorithm.maxColumn())
      return Error(WrongPointSize);
    DataSet src2 = src2Ptr->getDataSet();
    if (src2.size() == 0) return Error(EmptyDataSet);
    DataSet result(mAlgorithm.numColumns() + src2.pointSize());
    mAlgorithm.process(src1, src2, result);
    destPtr->setDataSet(result);
    return OK();
  }

  MessageResult<void> clear()
  {
    mAlgorithm.clear();
    return OK();
  }

  MessageResult<void> limit(index rows)
  {
    if (rows <= 0) return Error("invalid value");
    mAlgorithm.limit(rows);
    return OK();
  }

  static auto getMessageDescriptors()
  {
    return defineMessages(
        makeMessage("transform", &DataSetQueryClient::transform),
        makeMessage("transformJoin", &DataSetQueryClient::transformJoin),
        makeMessage("addColumn", &DataSetQueryClient::addColumn),
        makeMessage("addRange", &DataSetQueryClient::addRange),
        makeMessage("filter", &DataSetQueryClient::filter),
        makeMessage("and", &DataSetQueryClient::andFilter),
        makeMessage("or", &DataSetQueryClient::orFilter),
        makeMessage("clear", &DataSetQueryClient::clear),
        makeMessage("limit", &DataSetQueryClient::limit));
  }

private:
  algorithm::DataSetQuery mAlgorithm;
};
} // namespace datasetquery

using NRTThreadedDataSetQueryClient =
    NRTThreadingAdaptor<ClientWrapper<datasetquery::DataSetQueryClient>>;
} // namespace client
} // namespace fluid
