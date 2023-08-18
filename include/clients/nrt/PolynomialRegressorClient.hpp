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
    LongParam("degree", "Degree of polynomial", 2, Min(0)),
    LongParam("regressors", "Number of regrssors", 1, Min(1))
);

class PolynomialRegressorClient : public FluidBaseClient,
                                  OfflineIn,
                                  OfflineOut,
                                  ModelObject,
                                  public DataClient<algorithm::PolynomialRegressor>
{
  enum {
    kName,
    kDegree,
    kRegressors
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

  void setParams(ParamSetViewType& p) { 
    mParams = p;

    mAlgorithm.setDegree(get<kDegree>());
    mAlgorithm.setSize(get<kRegressors>());
  }

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

  MessageResult<void> fit(InputDataSetClientRef source,
                          InputDataSetClientRef target)
  {
    auto sourceClientPtr = source.get().lock();
    if (!sourceClientPtr) return Error<void>(NoDataSet);

    auto sourceDataSet = sourceClientPtr->getDataSet();
    if (sourceDataSet.size() == 0) return Error<void>(EmptyDataSet);
    if (sourceDataSet.dims() != mAlgorithm.size())
      return Error<void>(DimensionsDontMatch);

    auto targetClientPtr = target.get().lock();
    if (!targetClientPtr) return Error<void>(NoDataSet);

    auto targetDataSet = targetClientPtr->getDataSet();
    if (targetDataSet.size() == 0) return Error<void>(EmptyDataSet);
    if (sourceDataSet.size() != targetDataSet.size())
      return Error<void>(SizesDontMatch);
    if (sourceDataSet.dims() != targetDataSet.dims())
      return Error<void>(SizesDontMatch);

    if (!mAlgorithm.initialized()) 
      mAlgorithm.init(get<kDegree>(), get<kRegressors>());
    
    auto data = sourceDataSet.getData();
    auto tgt = targetDataSet.getData();

    mAlgorithm.calculateRegressionCoefficients(data, tgt);

    return OK();
  }

   MessageResult<void> predict(InputDataSetClientRef src,
                               DataSetClientRef dest)
  {
    index inputSize = mAlgorithm.size();
    index outputSize = mAlgorithm.size();
    auto  srcPtr = src.get().lock();
    auto  destPtr = dest.get().lock();

    if (!srcPtr || !destPtr) return Error(NoDataSet);

    auto srcDataSet = srcPtr->getDataSet();
    if (srcDataSet.size() == 0) return Error(EmptyDataSet);

    if (!mAlgorithm.regressed()) return Error(NoDataFitted);
    if (srcDataSet.dims() != inputSize) return Error(WrongPointSize);

    StringVector ids{srcDataSet.getIds()};
    RealMatrix output(srcDataSet.size(), outputSize);

    mAlgorithm.getMappedSpace(srcDataSet.getData(), output);

    DataSet result(ids, output);
    destPtr->setDataSet(result);

    return OK();
  }

  MessageResult<void> predictPoint(InputBufferPtr in, BufferPtr out) const
  {
    index inputSize = mAlgorithm.size();
    index outputSize = mAlgorithm.size();

    if (!in || !out) return Error(NoBuffer);

    BufferAdaptor::ReadAccess inBuf(in.get());
    BufferAdaptor::Access outBuf(out.get());

    if (!inBuf.exists()) return Error(InvalidBuffer);
    if (!outBuf.exists()) return Error(InvalidBuffer);
    if (inBuf.numFrames() != inputSize) return Error(WrongPointSize);

    if (!mAlgorithm.regressed()) return Error(NoDataFitted);

    Result resizeResult = outBuf.resize(outputSize, 1, inBuf.sampleRate());
    if (!resizeResult.ok()) return Error(BufferAlloc);

    RealMatrix src(inputSize, 1);
    RealMatrix dest(outputSize, 1);
    
    src.col(0) <<= inBuf.samps(0, inputSize, 0);
    mAlgorithm.getMappedSpace(src, dest);
    outBuf.samps(0, outputSize, 0) <<= dest.col(0);

    return OK();
  }

  
  MessageResult<string> print()
  {
    return "PolynomialRegressor " 
          + std::string(get<kName>()) 
          + "\ndegree: "
          + std::to_string(mAlgorithm.dims()) 
          + "\n regressed: " 
          + (mAlgorithm.regressed() ? "true" : "false");
  }

  static auto getMessageDescriptors()
  {
    return defineMessages(
        makeMessage("fit",    &PolynomialRegressorClient::fit),
        makeMessage("dims",   &PolynomialRegressorClient::dims),
        makeMessage("clear",  &PolynomialRegressorClient::clear),
        makeMessage("size",   &PolynomialRegressorClient::size),
        makeMessage("print",  &PolynomialRegressorClient::print),
        makeMessage("predict",&PolynomialRegressorClient::predict),
        makeMessage("predictPoint", &PolynomialRegressorClient::predictPoint),
        makeMessage("load",   &PolynomialRegressorClient::load),
        makeMessage("dump",   &PolynomialRegressorClient::dump),
        makeMessage("write",  &PolynomialRegressorClient::write),
        makeMessage("read",   &PolynomialRegressorClient::read));
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
      : mParams(p)
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
}; 

} // namespace polynomialregressor

using NRTThreadedPolynomialRegressorClient =
    NRTThreadingAdaptor<typename polynomialregressor::PolynomialRegressorRef::SharedType>;

using RTPolynomialRegressorQueryClient =
    ClientWrapper<polynomialregressor::PolynomialRegressorQuery>;

} // namespace client
} // namespace fluid