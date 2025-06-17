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

constexpr auto DataSetQueryParams =
    defineParameters(StringParam<Fixed<true>>("name", "Name"));

class DataSetQueryClient : public FluidBaseClient,
                           OfflineIn,
                           OfflineOut,
                           ModelObject,
                           public DataClient<algorithm::DataSetQuery>

{
  enum { kName };

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

  DataSetQueryClient(ParamSetViewType& p, FluidContext&) : mParams(p) {}

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
                                DataSetClientRef      destClient)
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
                                    DataSetClientRef      destClient)
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

  MessageResult<void> limit(index points)
  {
    if (points <= 0) return Error("invalid limit on the number of points");
    mAlgorithm.limit(points);
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

  const algorithm::DataSetQuery& algorithm() const { return mAlgorithm; }

private:
  algorithm::DataSetQuery mAlgorithm;
};

using DSQueryRef = SharedClientRef<const DataSetQueryClient>;

constexpr auto DataSetRTQueryParams = defineParameters(
    DSQueryRef::makeParam("dataSetQuery", "DataSetQuery"),
    InputDataSetClientRef::makeParam("sourceClient", "Source DataSet Name"),
    DataSetClientRef::makeParam("destClient", "Destination DataSet Name"));

class DataSetRTQuery : public FluidBaseClient, ControlIn, ControlOut
{
  enum { kDSQ, kSourceDataSet, kDestDataSet };

public:
  using ParamDescType = decltype(DataSetRTQueryParams);
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
    return DataSetRTQueryParams;
  }

  DataSetRTQuery(ParamSetViewType& p, FluidContext& c) : mParams(p)
  {
    controlChannelsIn(1);
    controlChannelsOut({1, 1});
  }

  template <typename T>
  void process(std::vector<FluidTensorView<T, 1>>& input,
               std::vector<FluidTensorView<T, 1>>& output, FluidContext& c)
  {
    if (input[0](0) > 0)
    {
      output[0](0) = 0; // start with error output as default

      auto DSQptr = get<kDSQ>().get().lock();
      if (!DSQptr)
        return; // c.reportError("FluidDataSetQuery RT Query: No
                // FluidDataSetQuery found");
                //      if (DSQptr->algorithm().numColumns() <= 0)
      //        return; // c.reportError("FluidDataSetQuery RT Query: No colums

      auto sourceDSptr = get<kSourceDataSet>().get().lock();
      if (!sourceDSptr)
        return; // c.reportError("FluidDataSetQuery RT Query: invalid Source
                // FluidDataSet");

      auto destDSptr = get<kDestDataSet>().get().lock();
      if (!destDSptr)
        return; // c.reportError("FluidDataSetQuery RT Query: invalid
                // Destination FluidDataSet");

      output[0](0) = 1; // reaching here means success as a trigger output
    }
  }
    
    index latency() const { return 0; }

};

} // namespace datasetquery

using NRTThreadedDataSetQueryClient =
    NRTThreadingAdaptor<typename datasetquery::DSQueryRef::SharedType>;
using RTDataSetQueryClient = ClientWrapper<datasetquery::DataSetRTQuery>;

} // namespace client
} // namespace fluid
