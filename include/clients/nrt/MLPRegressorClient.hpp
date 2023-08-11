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
#include "../../algorithms/public/MLP.hpp"
#include "../../algorithms/public/SGD.hpp"
#include <string>

namespace fluid {
namespace client {
namespace mlpregressor {

constexpr std::initializer_list<index> HiddenLayerDefaults = {3, 3};

constexpr auto MLPRegressorParams = defineParameters(
    StringParam<Fixed<true>>("name", "Name"),
    LongArrayParam("hiddenLayers", "Hidden Layer Sizes", HiddenLayerDefaults),
    EnumParam("activation", "Activation Function", 2, "Identity", "Sigmoid",
              "ReLU", "Tanh"),
    EnumParam("outputActivation", "Output Activation Function", 0, "Identity",
              "Sigmoid", "ReLU", "Tanh"),
    LongParam("tapIn", "Input Tap Index", 0, Min(0)),
    LongParam("tapOut", "Output Tap Index", -1, Min(-1)),
    LongParam("maxIter", "Maximum Number of Iterations", 1000, Min(1)),
    FloatParam("learnRate", "Learning Rate", 0.01, Min(0.0), Max(1.0)),
    FloatParam("momentum", "Momentum", 0.9, Min(0.0), Max(0.99)),
    LongParam("batchSize", "Batch Size", 50, Min(1)),
    FloatParam("validation", "Validation Amount", 0.2, Min(0), Max(0.9)));

class MLPRegressorClient : public FluidBaseClient,
                           OfflineIn,
                           OfflineOut,
                           ModelObject,
                           public DataClient<algorithm::MLP>
{

  enum {
    kName,
    kHidden,
    kActivation,
    kOutputActivation,
    kInputTap,
    kOutputTap,
    kIter,
    kRate,
    kMomentum,
    kBatchSize,
    kVal
  };

public:
  using string = std::string;
  using BufferPtr = std::shared_ptr<BufferAdaptor>;
  using InputBufferPtr = std::shared_ptr<const BufferAdaptor>;
  using IndexVector = FluidTensor<index, 1>;
  using StringVector = FluidTensor<string, 1>;
  using DataSet = FluidDataSet<string, double, 1>;

  using ParamDescType = decltype(MLPRegressorParams);

  using ParamSetViewType = ParameterSetView<ParamDescType>;
  using ParamValues = typename ParamSetViewType::ValueTuple;

  std::reference_wrapper<ParamSetViewType> mParams;

  void setParams(ParamSetViewType& p) { mParams = p; }

  template <size_t N>
  auto& get() const
  {
    return mParams.get().template get<N>();
  }

  static constexpr auto& getParameterDescriptors()
  {
    return MLPRegressorParams;
  }

  MLPRegressorClient(ParamSetViewType& p, FluidContext&) : mParams(p) {}

  template <typename T>
  Result process(FluidContext&)
  {
    return {};
  }

  MessageResult<double> fit(InputDataSetClientRef source,
                            InputDataSetClientRef target)
  {
    auto sourceClientPtr = source.get().lock();
    if (!sourceClientPtr) return Error<double>(NoDataSet);
    auto sourceDataSet = sourceClientPtr->getDataSet();
    if (sourceDataSet.size() == 0) return Error<double>(EmptyDataSet);
    if (mAlgorithm.initialized() && sourceDataSet.dims() != mAlgorithm.dims())
      return Error<double>(DimensionsDontMatch);

    auto targetClientPtr = target.get().lock();
    if (!targetClientPtr) return Error<double>(NoDataSet);
    auto targetDataSet = targetClientPtr->getDataSet();
    if (targetDataSet.size() == 0) return Error<double>(EmptyDataSet);
    if (sourceDataSet.size() != targetDataSet.size())
      return Error<double>(SizesDontMatch);
    index outputAct = get<kOutputActivation>() == -1 ? get<kActivation>()
                                                     : get<kOutputActivation>();
    if (!mAlgorithm.initialized() ||
        mTracker.changed(sourceDataSet.pointSize(), targetDataSet.pointSize(),
                         get<kHidden>(), get<kActivation>(), outputAct))
    {

      mAlgorithm.init(sourceDataSet.pointSize(), targetDataSet.pointSize(),
                      get<kHidden>(), get<kActivation>(), outputAct);
    }

    mAlgorithm.setTrained(false);
    DataSet        result(1);
    auto           data = sourceDataSet.getData();
    auto           tgt = targetDataSet.getData();
    algorithm::SGD sgd;
    double         error =
        sgd.train(mAlgorithm, data, tgt, get<kIter>(), get<kBatchSize>(),
                  get<kRate>(), get<kMomentum>(), get<kVal>());
    return error;
  }

  MessageResult<void> predict(InputDataSetClientRef srcClient,
                              DataSetClientRef      destClient)
  {
    index inputTap = get<kInputTap>();
    index outputTap = get<kOutputTap>();
    if (inputTap >= mAlgorithm.size()) return Error("Input tap too large");
    if (outputTap > mAlgorithm.size()) return Error("Ouput tap too large");
    if (outputTap == 0) return Error("Ouput tap cannot be 0");
    if (outputTap == -1) outputTap = mAlgorithm.size();
    if (outputTap - inputTap <= 0) return Error("Output Tap must be > Input Tap");

    index inputSize = mAlgorithm.inputSize(inputTap);
    index outputSize = mAlgorithm.outputSize(outputTap);
    auto  srcPtr = srcClient.get().lock();
    auto  destPtr = destClient.get().lock();

    if (!srcPtr || !destPtr) return Error(NoDataSet);
    auto srcDataSet = srcPtr->getDataSet();
    if (srcDataSet.size() == 0) return Error(EmptyDataSet);
    if (!mAlgorithm.trained()) return Error(NoDataFitted);
    if (srcDataSet.dims() != inputSize) return Error(WrongPointSize);

    StringVector ids{srcDataSet.getIds()};
    RealMatrix   output(srcDataSet.size(), outputSize);
    mAlgorithm.process(srcDataSet.getData(), output, inputTap, outputTap);
    FluidDataSet<string, double, 1> result(ids, output);
    destPtr->setDataSet(result);
    return OK();
  }

  MessageResult<void> predictPoint(InputBufferPtr in, BufferPtr out)
  {
    index inputTap = get<kInputTap>();
    index outputTap = get<kOutputTap>();
    if (inputTap >= mAlgorithm.size()) return Error("Input tap too large");
    if (outputTap > mAlgorithm.size()) return Error("Ouput tap too large");
    if (outputTap == 0) return Error("Ouput tap should be > 0 or -1");
    if (outputTap == -1) outputTap = mAlgorithm.size();
    if (outputTap - inputTap <= 0) return Error("Output Tap must be > Input Tap");

    index inputSize = mAlgorithm.inputSize(inputTap);
    index outputSize = mAlgorithm.outputSize(outputTap);

    if (!in || !out) return Error(NoBuffer);
    BufferAdaptor::ReadAccess inBuf(in.get());
    BufferAdaptor::Access outBuf(out.get());
    if (!inBuf.exists()) return Error(InvalidBuffer);
    if (!outBuf.exists()) return Error(InvalidBuffer);
    if (inBuf.numFrames() != inputSize) return Error(WrongPointSize);
    if (!mAlgorithm.trained()) return Error(NoDataFitted);

    Result resizeResult = outBuf.resize(outputSize, 1, inBuf.sampleRate());
    if (!resizeResult.ok()) return Error(BufferAlloc);
    RealVector src(inputSize);
    RealVector dest(outputSize);
    src <<= inBuf.samps(0, inputSize, 0);
    mAlgorithm.processFrame(src, dest, inputTap, outputTap);
    outBuf.samps(0, outputSize, 0) <<= dest;
    return OK();
  }

  MessageResult<ParamValues> read(string fileName)
  {
    auto result = DataClient::read(fileName);
    if (result.ok()) return  updateParameters();
    return {result.status(), result.message()};
  }

  MessageResult<ParamValues> load(string s)
  {
    auto result = DataClient::load(s);
    if (result.ok()) return updateParameters();
    return {result.status(), result.message()};
  }

  static auto getMessageDescriptors()
  {
    return defineMessages(
        makeMessage("fit", &MLPRegressorClient::fit),
        makeMessage("predict", &MLPRegressorClient::predict),
        makeMessage("predictPoint", &MLPRegressorClient::predictPoint),
        makeMessage("clear", &MLPRegressorClient::clear),
        makeMessage("cols", &MLPRegressorClient::dims),
        makeMessage("size", &MLPRegressorClient::size),
        makeMessage("load", &MLPRegressorClient::load),
        makeMessage("dump", &MLPRegressorClient::dump),
        makeMessage("write", &MLPRegressorClient::write),
        makeMessage("read", &MLPRegressorClient::read));
  }

private:

  ParameterTrackChanges<index, index, IndexVector, index, index> mTracker;

  MessageResult<ParamValues> updateParameters()
  {
    const index nLayers = mAlgorithm.size();
    ParamValues newParams = mParams.get().toTuple();

    if (nLayers > 1)
    {
      FluidTensor<index, 1> layersParam(nLayers - 1);
      for (index i = 0; i < nLayers - 1; i++)
        layersParam[i] = mAlgorithm.outputSize(i + 1);
      std::get<kHidden>(newParams) = std::move(layersParam);
      std::get<kActivation>(newParams) = mAlgorithm.hiddenActivation();
      std::get<kOutputActivation>(newParams) = mAlgorithm.outputActivation();
      std::get<kInputTap>(newParams) = 0;
      std::get<kOutputTap>(newParams) = -1;
    }

    return newParams;
  }
};

using MLPRegressorRef = SharedClientRef<const MLPRegressorClient>;

constexpr auto MLPRegressorQueryParams =
    defineParameters(MLPRegressorRef::makeParam("model", "Source Model"),
                     LongParam("tapIn", "Input Tap Index", 0, Min(0)),
                     LongParam("tapOut", "Output Tap Index", -1, Min(-1)),
                     InputBufferParam("inputPointBuffer", "Input Point Buffer"),
                     BufferParam("predictionBuffer", "Prediction Buffer"));

class MLPRegressorQuery : public FluidBaseClient, ControlIn, ControlOut
{
  enum { kModel, kInputTap, kOutputTap, kInputBuffer, kOutputBuffer };

public:
  using ParamDescType = decltype(MLPRegressorQueryParams);

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
    return MLPRegressorQueryParams;
  }

  MLPRegressorQuery(ParamSetViewType& p, FluidContext&) : mParams(p)
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
      auto MLPRef = get<kModel>().get().lock();
      if (!MLPRef)
      {
        // report error?
        return;
      }

      algorithm::MLP const& algorithm = MLPRef->algorithm();

      if (!algorithm.trained()) return;
      index inputTap = get<kInputTap>();
      index outputTap = get<kOutputTap>();
      if (inputTap >= algorithm.size() - 1) return;
      if (outputTap >= algorithm.size()) return;
      if (outputTap == 0) return;
      if (outputTap == -1) outputTap = algorithm.size();

      index inputSize = algorithm.inputSize(inputTap);
      index outputSize = algorithm.outputSize(outputTap);

      InOutBuffersCheck bufCheck(inputSize);
      if (!bufCheck.checkInputs(get<kInputBuffer>().get(),
                                get<kOutputBuffer>().get()))
        return;
      auto outBuf = BufferAdaptor::Access(get<kOutputBuffer>().get());
      if (outBuf.samps(0).size() < outputSize) return;

      RealVector src(inputSize);
      RealVector dest(outputSize);
      src <<= BufferAdaptor::ReadAccess(get<kInputBuffer>().get())
                .samps(0, inputSize, 0);
      algorithm.processFrame(src, dest, inputTap, outputTap);
      outBuf.samps(0, outputSize, 0) <<= dest;
    }
  }

  index latency() const { return 0; }
};

} // namespace mlpregressor

using NRTThreadedMLPRegressorClient =
    NRTThreadingAdaptor<typename mlpregressor::MLPRegressorRef::SharedType>;

using RTMLPRegressorQueryClient =
    ClientWrapper<mlpregressor::MLPRegressorQuery>;
} // namespace client
} // namespace fluid
