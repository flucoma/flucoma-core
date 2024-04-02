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
#include "../../algorithms/public/UMAP.hpp"

namespace fluid {
namespace client {
namespace umap {

constexpr auto UMAPParams = defineParameters(
    StringParam<Fixed<true>>("name", "Name"),
    LongParam("numDimensions", "Target Number of Dimensions", 2, Min(1)),
    LongParam("numNeighbours", "Number of Nearest Neighbours", 15, Min(1)),
    FloatParam("minDist", "Minimum Distance", 0.1, Min(0)),
    LongParam("iterations", "Number of Iterations", 200, Min(1)),
    FloatParam("learnRate", "Learning Rate", 0.1, Min(0.0), Max(1.0)));

class UMAPClient : public FluidBaseClient,
                   OfflineIn,
                   OfflineOut,
                   ModelObject,
                   public DataClient<algorithm::UMAP>
{
  enum {
    kName,
    kNumDimensions,
    kNumNeighbors,
    kMinDistance,
    kNumIter,
    kLearningRate
  };

public:
  using string = std::string;
  using BufferPtr = std::shared_ptr<BufferAdaptor>;
  using InputBufferPtr = std::shared_ptr<const BufferAdaptor>;
  using StringVector = FluidTensor<string, 1>;

  using ParamDescType = decltype(UMAPParams);

  using ParamSetViewType = ParameterSetView<ParamDescType>;
  std::reference_wrapper<ParamSetViewType> mParams;

  void setParams(ParamSetViewType& p) { mParams = p; }

  template <size_t N>
  auto& get() const
  {
    return mParams.get().template get<N>();
  }

  static constexpr auto getParameterDescriptors() { return UMAPParams; }

  UMAPClient(ParamSetViewType& p, FluidContext&) : mParams(p) {}

  template <typename T>
  Result process(FluidContext&)
  {
      return{};
  }

  MessageResult<void> fitTransform(InputDataSetClientRef sourceClient,
                                   DataSetClientRef destClient)
  {
    auto srcPtr = sourceClient.get().lock();
    auto destPtr = destClient.get().lock();
    if (!srcPtr || !destPtr) return Error(NoDataSet);
    auto src = srcPtr->getDataSet();
    auto dest = destPtr->getDataSet();
    if (src.size() == 0) return Error(EmptyDataSet);
    if (get<kNumNeighbors>() > src.size())
      return Error("Number of Neighbours is larger than dataset");
    FluidDataSet<string, double, 1> result;
    try
    {
      result = mAlgorithm.train(src, get<kNumNeighbors>(), get<kNumDimensions>(),
                                get<kMinDistance>(), get<kNumIter>(),
                                get<kLearningRate>());
    }
    catch (const std::runtime_error& e) //spectra library will throw if eigen decomp fails
    {
      return {Result::Status::kError, e.what()};
    }
    destPtr->setDataSet(result);
    return OK();
  }

  MessageResult<void> fit(InputDataSetClientRef sourceClient)
  {
    auto srcPtr = sourceClient.get().lock();
    if (!srcPtr) return Error(NoDataSet);
    auto src = srcPtr->getDataSet();
    if (src.size() == 0) return Error(EmptyDataSet);
    if (get<kNumNeighbors>() > src.size())
      return Error("Number of Neighbours is larger than dataset");
    StringVector                    ids{src.getIds()};
    FluidDataSet<string, double, 1> result;
    result = mAlgorithm.train(src, get<kNumNeighbors>(), get<kNumDimensions>(),
                              get<kMinDistance>(), get<kNumIter>(),
                              get<kLearningRate>());
    return OK();
  }

  MessageResult<void> transform(InputDataSetClientRef sourceClient,
                                DataSetClientRef destClient)
  {
    auto srcPtr = sourceClient.get().lock();
    auto destPtr = destClient.get().lock();
    if (!srcPtr || !destPtr) return Error(NoDataSet);
    auto src = srcPtr->getDataSet();
    auto dest = destPtr->getDataSet();
    if (src.size() == 0) return Error(EmptyDataSet);
    if (!mAlgorithm.initialized()) return Error(NoDataFitted);
    if (get<kNumDimensions>() != mAlgorithm.dims())
      return Error("Wrong target number of dimensions");
    if (src.pointSize() != mAlgorithm.inputDims()) return Error(WrongPointSize);
    StringVector                    ids{src.getIds()};
    FluidDataSet<string, double, 1> result;
    result = mAlgorithm.transform(src, get<kNumIter>(), get<kLearningRate>());
    destPtr->setDataSet(result);
    return OK();
  }

  MessageResult<void> transformPoint(InputBufferPtr in, BufferPtr out)
  {
    index inSize = mAlgorithm.inputDims();
    index outSize = mAlgorithm.dims();
    if (!mAlgorithm.initialized()) return Error(NoDataFitted);
    if (get<kNumDimensions>() != outSize)
      return Error("Wrong target number of dimensions");
    InOutBuffersCheck bufCheck(inSize);
    if (!bufCheck.checkInputs(in.get(), out.get()))
      return Error(bufCheck.error());
    BufferAdaptor::Access outBuf(out.get());
    Result resizeResult = outBuf.resize(outSize, 1, outBuf.sampleRate());
    if (!resizeResult.ok()) return Error(BufferAlloc);
    FluidTensor<double, 1> src(inSize);
    FluidTensor<double, 1> dest(outSize);
    src <<= BufferAdaptor::ReadAccess(in.get()).samps(0, inSize, 0);
    mAlgorithm.transformPoint(src, dest);
    outBuf.samps(0, outSize, 0) <<= dest;
    return OK();
  }

  static auto getMessageDescriptors()
  {
    return defineMessages(
        makeMessage("fitTransform", &UMAPClient::fitTransform),
        makeMessage("fit", &UMAPClient::fit),
        makeMessage("transform", &UMAPClient::transform),
        makeMessage("transformPoint", &UMAPClient::transformPoint),
        makeMessage("cols", &UMAPClient::dims),
        makeMessage("clear", &UMAPClient::clear),
        makeMessage("size", &UMAPClient::size),
        makeMessage("load", &UMAPClient::load),
        makeMessage("dump", &UMAPClient::dump),
        makeMessage("write", &UMAPClient::write),
        makeMessage("read", &UMAPClient::read));
  }
};

using UMAPRef = SharedClientRef<const UMAPClient>;

constexpr auto UMAPQueryParams =
    defineParameters(UMAPRef::makeParam("model", "Source Model"),
                     InputBufferParam("inputPointBuffer", "Input Point Buffer"),
                     BufferParam("predictionBuffer", "Prediction Buffer"));

class UMAPQuery : public FluidBaseClient, ControlIn, ControlOut
{
  enum { kModel, kInputBuffer, kOutputBuffer };

public:
  using ParamDescType = decltype(UMAPQueryParams);

  using ParamSetViewType = ParameterSetView<ParamDescType>;
  std::reference_wrapper<ParamSetViewType> mParams;

  void setParams(ParamSetViewType& p) { mParams = p; }

  template <size_t N>
  auto& get() const
  {
    return mParams.get().template get<N>();
  }

  static constexpr auto getParameterDescriptors() { return UMAPQueryParams; }

  UMAPQuery(ParamSetViewType& p, FluidContext&) : mParams(p)
  {
    controlChannelsIn(1);
    controlChannelsOut({1, 1});
  }

  template <typename T>
  void process(std::vector<FluidTensorView<T, 1>>& input,
               std::vector<FluidTensorView<T, 1>>& output, FluidContext&)
  {
    output[0] <<= input[0];
    if (input[0](0) > 0)
    {
      auto UMAPPtr = get<kModel>().get().lock();
      if (!UMAPPtr)
      {
        // report error?
        return;
      }
      algorithm::UMAP const& algorithm = UMAPPtr->algorithm();
      if (!algorithm.initialized()) return;
      index inSize = algorithm.inputDims();
      index outSize = algorithm.dims();
      if (algorithm.dims() != outSize) return;
      InOutBuffersCheck bufCheck(inSize);
      if (!bufCheck.checkInputs(get<kInputBuffer>().get(),
                                get<kOutputBuffer>().get()))
        return;
      auto outBuf = BufferAdaptor::Access(get<kOutputBuffer>().get());
      if (outBuf.samps(0).size() < outSize) return;
      RealVector src(inSize);
      RealVector dest(outSize);
      src <<= BufferAdaptor::ReadAccess(get<kInputBuffer>().get())
                .samps(0, inSize, 0);
      algorithm.transformPoint(src, dest);
      outBuf.samps(0, outSize, 0) <<= dest;
    }
  }

  index latency() const { return 0; }
};

} // namespace umap

using NRTThreadedUMAPClient =
    NRTThreadingAdaptor<typename umap::UMAPRef::SharedType>;

using RTUMAPQueryClient = ClientWrapper<umap::UMAPQuery>;

} // namespace client
} // namespace fluid
