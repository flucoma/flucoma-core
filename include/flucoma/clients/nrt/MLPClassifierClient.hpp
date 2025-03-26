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
#include "LabelSetClient.hpp"
#include "NRTClient.hpp"
#include "../../algorithms/public/LabelSetEncoder.hpp"
#include "../../algorithms/public/MLP.hpp"
#include "../../algorithms/public/SGD.hpp"
#include <string>

namespace fluid {
namespace client {
namespace mlpclassifier {

struct MLPClassifierData
{
  algorithm::MLP             mlp;
  algorithm::LabelSetEncoder encoder;
  index                      size() const { return mlp.size(); }
  index                      dims() const { return mlp.dims(); }
  void                       clear()
  {
    mlp.clear();
    encoder.clear();
  }
  bool initialized() const { return mlp.initialized(); }
};

void to_json(nlohmann::json& j, const MLPClassifierData& data)
{
  j["mlp"] = data.mlp;
  j["labels"] = data.encoder;
}

bool check_json(const nlohmann::json& j, const MLPClassifierData&)
{
  return fluid::check_json(j, {"mlp", "labels"},
                           {JSONTypes::OBJECT, JSONTypes::OBJECT});
}

void from_json(const nlohmann::json& j, MLPClassifierData& data)
{
  data.mlp = j.at("mlp").get<algorithm::MLP>();
  data.encoder = j.at("labels").get<algorithm::LabelSetEncoder>();
}

constexpr std::initializer_list<index> HiddenLayerDefaults = {3, 3};

constexpr auto MLPClassifierParams = defineParameters(
    StringParam<Fixed<true>>("name", "Name"),
    LongArrayParam("hiddenLayers", "Hidden Layer Sizes", HiddenLayerDefaults),
    EnumParam("activation", "Activation Function", 2, "Identity", "Sigmoid",
              "ReLU", "Tanh"),
    LongParam("maxIter", "Maximum Number of Iterations", 1000, Min(1)),
    FloatParam("learnRate", "Learning Rate", 0.01, Min(0.0), Max(1.0)),
    FloatParam("momentum", "Momentum", 0.5, Min(0.0), Max(0.99)),
    LongParam("batchSize", "Batch Size", 50, Min(1)),
    FloatParam("validation", "Validation Amount", 0.2, Min(0), Max(0.9)));


class MLPClassifierClient : public FluidBaseClient,
                            OfflineIn,
                            OfflineOut,
                            ModelObject,
                            public DataClient<MLPClassifierData>
{
  enum {
    kName,
    kHidden,
    kActivation,
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
  using LabelSet = FluidDataSet<string, string, 1>;

  using ParamDescType = decltype(MLPClassifierParams);

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
    return MLPClassifierParams;
  }

  MLPClassifierClient(ParamSetViewType& p, FluidContext&) : mParams(p) {}

  template <typename T>
  Result process(FluidContext&)
  {
    return {};
  }

  MessageResult<ParamValues> read(string fileName)
  {
    auto result = DataClient::read(fileName);
    if (result.ok()) return updateParameters();
    return {result.status(),result.message()};
  }

  MessageResult<ParamValues> load(string s)
  {
    auto result = DataClient::load(s);
    if (result.ok()) return updateParameters();
    return {result.status(), result.message()};
  }

  MessageResult<double> fit(InputDataSetClientRef source, InputLabelSetClientRef target)
  {
    auto sourceClientPtr = source.get().lock();
    if (!sourceClientPtr) return Error<double>(NoDataSet);
    auto sourceDataSet = sourceClientPtr->getDataSet();
    if (sourceDataSet.size() == 0) return Error<double>(EmptyDataSet);
    if (mAlgorithm.initialized() && sourceDataSet.dims() != mAlgorithm.dims())
      return Error<double>(DimensionsDontMatch);

    auto targetClientPtr = target.get().lock();
    if (!targetClientPtr) return Error<double>(NoLabelSet);
    auto targetDataSet = targetClientPtr->getLabelSet();
    if (targetDataSet.size() == 0) return Error<double>(EmptyLabelSet);
    if (sourceDataSet.size() != targetDataSet.size())
      return Error<double>(SizesDontMatch);

    mAlgorithm.encoder.fit(targetDataSet);

    if (mTracker.changed(sourceDataSet.pointSize(),
                         mAlgorithm.encoder.numLabels(), get<kHidden>(),
                         get<kActivation>()))
    {
      mAlgorithm.mlp.init(sourceDataSet.pointSize(),
                          mAlgorithm.encoder.numLabels(), get<kHidden>(),
                          get<kActivation>(), 1); // sigmoid output
    }
    mAlgorithm.mlp.setTrained(false);
    DataSet result(1);
    auto    data = sourceDataSet.getData();
    auto    tgt = targetDataSet.getData();

    RealMatrix oneHot(targetDataSet.size(), mAlgorithm.encoder.numLabels());
    oneHot.fill(0);
    for (index i = 0; i < targetDataSet.size(); i++)
    {
      mAlgorithm.encoder.encodeOneHot(tgt.row(i)(0), oneHot.row(i));
    }

    algorithm::SGD sgd;
    double         error =
        sgd.train(mAlgorithm.mlp, data, oneHot, get<kIter>(), get<kBatchSize>(),
                  get<kRate>(), get<kMomentum>(), get<kVal>());

    return error;
  }

  MessageResult<void> predict(InputDataSetClientRef  srcClient,
                              LabelSetClientRef destClient)
  {
    auto srcPtr = srcClient.get().lock();
    auto destPtr = destClient.get().lock();
    if (!srcPtr) return Error(NoDataSet);
    if (!destPtr) return Error(NoLabelSet);
    auto srcDataSet = srcPtr->getDataSet();
    if (srcDataSet.size() == 0) return Error(EmptyDataSet);
    if (!mAlgorithm.mlp.trained()) return Error(NoDataFitted);
    if (srcDataSet.dims() != mAlgorithm.dims()) return Error(WrongPointSize);

    StringVector ids{srcDataSet.getIds()};
    RealMatrix   output(srcDataSet.size(), mAlgorithm.encoder.numLabels());
    mAlgorithm.mlp.process(srcDataSet.getData(), output, 0,
                           mAlgorithm.mlp.size());
    LabelSet result(1);
    for (index i = 0; i < srcDataSet.size(); i++)
    {
      StringVector label = {mAlgorithm.encoder.decodeOneHot(output.row(i))};
      result.add(ids(i), label);
    }
    destPtr->setLabelSet(result);
    return OK();
  }

  MessageResult<string> predictPoint(InputBufferPtr in)
  {
    if (!in) return Error<string>(NoBuffer);
    BufferAdaptor::ReadAccess inBuf(in.get());
    if (!inBuf.exists()) return Error<string>(InvalidBuffer);
    if (inBuf.numFrames() != mAlgorithm.mlp.dims())
      return Error<string>(WrongPointSize);
    if (!mAlgorithm.mlp.trained()) return Error<string>(NoDataFitted);

    index      layer = mAlgorithm.mlp.size();
    RealVector src(mAlgorithm.mlp.dims());
    RealVector dest(mAlgorithm.mlp.outputSize(layer));
    src <<= inBuf.samps(0, mAlgorithm.mlp.dims(), 0);
    mAlgorithm.mlp.processFrame(src, dest, 0, layer);
    auto& label = mAlgorithm.encoder.decodeOneHot(dest);
    return label;
  }

  static auto getMessageDescriptors()
  {
    return defineMessages(
        makeMessage("fit", &MLPClassifierClient::fit),
        makeMessage("predict", &MLPClassifierClient::predict),
        makeMessage("predictPoint", &MLPClassifierClient::predictPoint),
        makeMessage("clear", &MLPClassifierClient::clear),
        makeMessage("cols", &MLPClassifierClient::dims),
        makeMessage("size", &MLPClassifierClient::size),
        makeMessage("load", &MLPClassifierClient::load),
        makeMessage("dump", &MLPClassifierClient::dump),
        makeMessage("write", &MLPClassifierClient::write),
        makeMessage("read", &MLPClassifierClient::read));
  }

private:

  ParameterTrackChanges<index, index, IndexVector, index> mTracker;

  MessageResult<ParamValues> updateParameters()
  {
    const index nLayers = mAlgorithm.size();
    ParamValues newParams = mParams.get().toTuple();

    if (nLayers > 1)
    {
      FluidTensor<index, 1> layersParam(nLayers - 1);
      for (index i = 0; i < nLayers - 1; i++)
        layersParam[i] = mAlgorithm.mlp.outputSize(i + 1);
      std::get<kHidden>(newParams) = std::move(layersParam);
      std::get<kActivation>(newParams) = mAlgorithm.mlp.hiddenActivation();
    }    
    return newParams; 
  }

};

using MLPClassifierRef = SharedClientRef<const MLPClassifierClient>;

constexpr auto MLPClassifierQueryParams =
    defineParameters(MLPClassifierRef::makeParam("model", "Source Model"),
                     InputBufferParam("inputPointBuffer", "Input Point Buffer"),
                     BufferParam("predictionBuffer", "Prediction Buffer"));

class MLPClassifierQuery : public FluidBaseClient, ControlIn, ControlOut
{
  enum { kModel, kInputBuffer, kOutputBuffer };

public:
  using ParamDescType = decltype(MLPClassifierQueryParams);

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
    return MLPClassifierQueryParams;
  }

  MLPClassifierQuery(ParamSetViewType& p, FluidContext&) : mParams(p)
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
      auto mlpPtr = get<kModel>().get().lock();
      if (!mlpPtr)
      {
        // report error?
        return;
      }
      MLPClassifierData const& algorithm = mlpPtr->algorithm();

      if (!algorithm.mlp.trained()) return;
      index dims = algorithm.mlp.dims();
      index layer = algorithm.mlp.size();

      InOutBuffersCheck bufCheck(dims);
      if (!bufCheck.checkInputs(get<kInputBuffer>().get(),
                                get<kOutputBuffer>().get()))
        return;
      auto outBuf = BufferAdaptor::Access(get<kOutputBuffer>().get());
      if (outBuf.samps(0).size() != 1) return;

      RealVector src(dims);
      RealVector dest(algorithm.mlp.outputSize(layer));
      src <<= BufferAdaptor::ReadAccess(get<kInputBuffer>().get())
                .samps(0, dims, 0);
      algorithm.mlp.processFrame(src, dest, 0, layer);
      auto& label = algorithm.encoder.decodeOneHot(dest);
      outBuf.samps(0)[0] =
          static_cast<double>(algorithm.encoder.encodeIndex(label));
    }
  }

  index latency() const { return 0; }
};


} // namespace mlpclassifier

using NRTThreadedMLPClassifierClient =
    NRTThreadingAdaptor<typename mlpclassifier::MLPClassifierRef::SharedType>;

using RTMLPClassifierQueryClient =
    ClientWrapper<mlpclassifier::MLPClassifierQuery>;

} // namespace client
} // namespace fluid
