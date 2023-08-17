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

#include "../../algorithms/public/PolynomialRegressor.hpp"

namespace fluid {
namespace client {
namespace polynomialregressor {

constexpr auto PolynomialRegressorParams = defineParameters(
    StringParam<Fixed<true>>("name", "Name"),
    LongParam("degree", "Degree of fit polynomial", 2, Min(0))
);

class PolynomialRegressorClient : public FluidBaseClient,
                                  OfflineIn,
                                  OfflineOut,
                                  ModelObject,
                                  public DataClient<algorithm::PolynomialRegressor>
{
  enum {
    kName,
    kDegree
  };

public:
  using string = std::string;  
  using BufferPtr = std::shared_ptr<BufferAdaptor>;
  using InputBufferPtr = std::shared_ptr<const BufferAdaptor>;
  using DataSet = FluidDataSet<string, double, 1>;
  using StringVector = FluidTensor<string, 1>;

  using ParamDescType = decltype(PolynomialRegressorParams);
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
    return PolynomialRegressorParams;
  }
  
// c.allocator();
  PolynomialRegressorClient(ParamSetViewType& p, FluidContext& c) : mParams(p) 
  {
    audioChannelsIn(1);
    controlChannelsOut({1, 1});
  }

  template <typename T>
  Result process(FluidContext&)
  {
    return {};
  }

  MessageResult<StringVector> fit(InputDataSetClientRef source,
                            InputDataSetClientRef target)
  {
    auto sourceClientPtr = source.get().lock();
    if (!sourceClientPtr) return Error<string>(NoDataSet);

    auto sourceDataSet = sourceClientPtr->getDataSet();
    if (sourceDataSet.size() == 0) return Error<string>(EmptyDataSet);

    if (sourceDataSet.dims() != 1)
      return Error<string>(DimensionsDontMatch);

    auto targetClientPtr = target.get().lock();
    if (!targetClientPtr) return Error<string>(NoDataSet);
    auto targetDataSet = targetClientPtr->getDataSet();
    if (targetDataSet.size() == 0) return Error<string>(EmptyDataSet);
    if (sourceDataSet.size() != targetDataSet.size())
      return Error<string>(SizesDontMatch);

    if (!mAlgorithm.initialized()) 
      mAlgorithm.init();
    
    auto data = sourceDataSet.getData();
    auto tgt = targetDataSet.getData().col(0);

    string res = " ";

    for (auto& x : data) {
      res += std::to_string(x);
    }

    return res.c_str();
  }

  static auto getMessageDescriptors()
  {
    return defineMessages(
        makeMessage("fit", &PolynomialRegressorClient::fit),
        makeMessage("cols", &PolynomialRegressorClient::dims),
        makeMessage("clear", &PolynomialRegressorClient::clear),
        makeMessage("size", &PolynomialRegressorClient::size));
  }
};

using PolynomialRegressorRef = SharedClientRef<const PolynomialRegressorClient>;

constexpr auto PolynomialRegressorQueryParams = defineParameters(
    PolynomialRegressorRef::makeParam("model", "Source Model"),
    LongParam("degree", "Prediction Polynomial Degree", 2, Min(0) ),
    InputDataSetClientRef::makeParam("dataSet", "DataSet Name"),
    InputBufferParam("inputPointBuffer", "Input Point Buffer"),
    BufferParam("predictionBuffer", "Prediction Buffer"));

class PolynomialRegressorQuery : public FluidBaseClient, ControlIn, ControlOut
{
  enum { kModel, kDegree, kInputBuffer, kOutputBuffer };

public:
  using ParamDescType = decltype(PolynomialRegressorQueryParams);
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
    return PolynomialRegressorQueryParams;
  }

  PolynomialRegressorQuery(ParamSetViewType& p, FluidContext& c) 
      : mParams(p), mRTBuffer(c.allocator())
  {
    controlChannelsIn(1);
    controlChannelsOut({1, 1});
  }

  template <typename T>
  void process(std::vector<FluidTensorView<T, 1>>& input,
               std::vector<FluidTensorView<T, 1>>& output, FluidContext& c)
  {

  }

  index latency() { return 0; }

private:
  RealVector            mRTBuffer;
  InputDataSetClientRef mDataSetClient;
}; 

} // namespace polynomialregressor

using NRTThreadedPolynomialRegressorClient =
    NRTThreadingAdaptor<typename polynomialregressor::PolynomialRegressorRef::SharedType>;

using RTPolynomialRegressorQueryClient =
    ClientWrapper<polynomialregressor::PolynomialRegressorQuery>;

} // namespace client
} // namespace fluid