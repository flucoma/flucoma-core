#pragma once

#include <algorithms/public/KDTree.hpp>
#include <algorithms/public/KMeans.hpp>
#include <algorithms/public/SKMeans.hpp>
#include <algorithms/public/Normalization.hpp>
#include <algorithms/public/RobustScaling.hpp>
#include <algorithms/public/PCA.hpp>
#include <algorithms/public/MLP.hpp>
#include <algorithms/public/UMAP.hpp>
#include <algorithms/public/Standardization.hpp>
#include <algorithms/public/LabelSetEncoder.hpp>
#include <data/FluidDataSet.hpp>
#include <data/FluidIndex.hpp>
#include <data/FluidTensor.hpp>
#include <data/TensorTypes.hpp>
#include <fstream>
#include <nlohmann/json.hpp>

namespace fluid {

enum class JSONTypes {
  STRING,
  NUMBER,
  OBJECT,
  ARRAY,
  BOOLEAN
};

bool check_json(const nlohmann::json &j,
  std::vector<std::string> keys, std::vector<JSONTypes> types) {
    if(keys.size()!=types.size()) return false;
    for (index i = 0; i < asSigned(keys.size()); i++) {
      std::string key = keys[asUnsigned(i)];
      if (!j.contains(key))
        return false;
      nlohmann::json value = j[key];
      switch(types[asUnsigned(i)]){
        case JSONTypes::STRING:if(!value.is_string()) return false; break;
        case JSONTypes::NUMBER:if(!value.is_number()) return false; break;
        case JSONTypes::OBJECT:if(!value.is_object()) return false; break;
        case JSONTypes::ARRAY:if(!value.is_array()) return false; break;
        case JSONTypes::BOOLEAN:if(!value.is_boolean()) return false; break;
      }
      if(value.is_null()) return false;
  }
  return true;
}

// FluidTensor

namespace impl {

template <typename T, typename Validator>
void from_json(const nlohmann::json &j, FluidTensor<T, 1>& t, Validator&& validate)
{
  using namespace nlohmann;
  if((t.size() > 1 && !j.is_array()) || (j.size() < asUnsigned(t.size()))) return;
  for(auto&& el:j)if(!validate(el)) return;
  std::transform(j.begin(), j.begin() + t.size(), t.begin(), [](const json& x)
  {return x.get<T>();});
}
} //impl

template <typename T>
void from_json(const nlohmann::json &j, FluidTensor<T, 1> &t) {
  impl::from_json(j, t, [](const auto& x){ return x.is_number(); });
}

void from_json(const nlohmann::json &j, FluidTensor<std::string, 1> &t) {
  impl::from_json(j, t, [](const auto& x){ return x.is_string(); });
}

template <typename T>
void to_json(nlohmann::json &j, const FluidTensorView<T, 1> &t) {
  j = std::vector<T>(t.begin(), t.end());
}

template <typename T>
void to_json(nlohmann::json &j, const FluidTensorView<const T, 1> &t) {
  j = std::vector<T>(t.begin(), t.end());
}

template <typename T>
void from_json(const nlohmann::json &j, FluidTensor<T, 2> &t) {
  if (j.size() > 0) {
    auto result = FluidTensor<T, 2>(asSigned(j.size()), asSigned(j[0].size()));
    FluidTensor<T, 1> tmp(asSigned(j[0].size()));
    for (size_t i = 0; i < j.size(); i++) {
      j.at(i).get_to(tmp);
      result.row(asSigned(i)) <<= tmp;
    }
    t = result;
  }
}

template <typename T>
void to_json(nlohmann::json &j, const FluidTensorView<T, 2> &t) {
  for (index i = 0; i < t.rows(); i++) {
    j.push_back(t.row(i));
  }
}

// FluidDataSet
template <typename T>
void to_json(nlohmann::json &j, const FluidDataSet<std::string, T, 1> &ds) {
  auto ids = ds.getIds();
  auto data = ds.getData();
  j["cols"] = ds.pointSize();
  for (index r = 0; r < ds.size(); r++) {
    j["data"][ids[r]] = data.row(r);
  }
}

template <typename T>
bool check_json(const nlohmann::json &j,
                const FluidDataSet<std::string, T, 1> &) {
  return fluid::check_json(j,
    {"data", "cols"},
    {JSONTypes::OBJECT, JSONTypes::NUMBER}
  );
}

template <typename T>
void from_json(const nlohmann::json &j, FluidDataSet<std::string, T, 1> &ds) {
  auto rows = j.at("data");
  index pointSize = j.at("cols").get<index>();
  ds.resize(pointSize);
  FluidTensor<T, 1> tmp(pointSize);
  for (auto r = rows.begin(); r != rows.end(); ++r) {
    r.value().get_to(tmp);
    ds.add(r.key(), tmp);
  }
}

namespace algorithm {
// KDTree
void to_json(nlohmann::json &j, const KDTree &tree) {
  KDTree::FlatData treeData = tree.toFlat();
  j["tree"] = FluidTensorView<index, 2>(treeData.tree);
  j["rows"] = treeData.data.rows();
  j["cols"] = treeData.data.cols();
  j["data"] = FluidTensorView<double, 2>(treeData.data);
  j["ids"] = FluidTensorView<std::string, 1>(treeData.ids);
}

bool check_json(const nlohmann::json &j, const KDTree &) {
  return fluid::check_json(j,
    {"rows", "cols", "data", "tree", "ids"},
    {JSONTypes::NUMBER, JSONTypes::NUMBER,
      JSONTypes::ARRAY, JSONTypes::ARRAY, JSONTypes::ARRAY
    }
  );
}

void from_json(const nlohmann::json &j, KDTree &tree) {
  index rows = j.at("rows").get<index>();
  index cols = j.at("cols").get<index>();
  KDTree::FlatData treeData(rows, cols);
  j.at("tree").get_to(treeData.tree);
  j.at("data").get_to(treeData.data);
  j.at("ids").get_to(treeData.ids);
  tree.fromFlat(treeData);
}

// KMeans
void to_json(nlohmann::json &j, const KMeans &kmeans) {
  RealMatrix means(kmeans.getK(), kmeans.dims());
  kmeans.getMeans(means);
  j["means"] = RealMatrixView(means);
  j["rows"] = means.rows();
  j["cols"] = means.cols();
}

bool check_json(const nlohmann::json &j, const KMeans &) {
  return fluid::check_json(j,
    {"rows", "cols", "means"},
    {JSONTypes::NUMBER, JSONTypes::NUMBER,JSONTypes::ARRAY}
  );
}

void from_json(const nlohmann::json &j, KMeans &kmeans) {
  index rows = j.at("rows").get<index>();
  index cols = j.at("cols").get<index>();
  RealMatrix means(rows, cols);
  j.at("means").get_to(means);
  kmeans.setMeans(means);
}

// SKMeans
void to_json(nlohmann::json &j, const SKMeans &skmeans) {
  RealMatrix means(skmeans.getK(), skmeans.dims());
  skmeans.getMeans(means);
  j["means"] = RealMatrixView(means);
  j["rows"] = means.rows();
  j["cols"] = means.cols();
}

bool check_json(const nlohmann::json &j, const SKMeans &) {
  return fluid::check_json(j,
    {"rows", "cols", "means"},
    {JSONTypes::NUMBER, JSONTypes::NUMBER,JSONTypes::ARRAY}
  );
}

void from_json(const nlohmann::json &j, SKMeans &skmeans) {
  index rows = j.at("rows").get<index>();
  index cols = j.at("cols").get<index>();
  RealMatrix means(rows, cols);
  j.at("means").get_to(means);
  skmeans.setMeans(means);
}


// Normalize
void to_json(nlohmann::json &j, const Normalization &normalization) {
  RealVector dataMin(normalization.dims());
  RealVector dataMax(normalization.dims());
  normalization.getDataMin(dataMin);
  normalization.getDataMax(dataMax);
  j["data_min"] = RealVectorView(dataMin);
  j["data_max"] = RealVectorView(dataMax);
  j["min"] = normalization.getMin();
  j["max"] = normalization.getMax();
  j["cols"] = normalization.dims();
}

bool check_json(const nlohmann::json &j, const Normalization &) {
  return fluid::check_json(j,
    {"cols", "data_min", "data_max", "min", "max"},
    {JSONTypes::NUMBER, JSONTypes::ARRAY, JSONTypes::ARRAY,
      JSONTypes::NUMBER, JSONTypes::NUMBER
    }
  );
}

void from_json(const nlohmann::json &j, Normalization &normalization) {
  index cols = j.at("cols").get<index>();
  RealVector dataMin(cols);
  RealVector dataMax(cols);
  j.at("data_min").get_to(dataMin);
  j.at("data_max").get_to(dataMax);
  double min = j.at("min").get<double>();
  double max = j.at("max").get<double>();
  normalization.init(min, max, dataMin, dataMax);
}

// RobustScale
void to_json(nlohmann::json &j, const RobustScaling &robustScaling) {
  RealVector median(robustScaling.dims());
  RealVector range(robustScaling.dims());
  RealVector dataLow(robustScaling.dims());
  RealVector dataHigh(robustScaling.dims());
  robustScaling.getMedian(median);
  robustScaling.getRange(range);
  robustScaling.getDataLow(dataLow);
  robustScaling.getDataHigh(dataHigh);
  j["data_low"] = RealVectorView(dataLow);
  j["data_high"] = RealVectorView(dataHigh);
  j["median"] = RealVectorView(median);
  j["range"] = RealVectorView(range);
  j["low"] = robustScaling.getLow();
  j["high"] = robustScaling.getHigh();
  j["cols"] = robustScaling.dims();
}

bool check_json(const nlohmann::json &j, const RobustScaling &) {
  return fluid::check_json(j,
    {"cols", "median", "range", "data_low", "data_high","low", "high"},
    {JSONTypes::NUMBER, JSONTypes::ARRAY, JSONTypes::ARRAY,
      JSONTypes::ARRAY, JSONTypes::ARRAY,
      JSONTypes::NUMBER, JSONTypes::NUMBER
    }
  );
}

void from_json(const nlohmann::json &j, RobustScaling &robustScaling) {
  index cols = j.at("cols").get<index>();
  RealVector median(cols);
  RealVector range(cols);
  RealVector dataLow(cols);
  RealVector dataHigh(cols);
  j.at("median").get_to(median);
  j.at("range").get_to(range);
  j.at("data_low").get_to(dataLow);
  j.at("data_high").get_to(dataHigh);
  double low = j.at("low").get<double>();
  double high = j.at("high").get<double>();
  robustScaling.init(low, high, dataLow, dataHigh, median, range);
}

// Standardize
void to_json(nlohmann::json &j, const Standardization &standardization) {
  RealVector mean(standardization.dims());
  RealVector std(standardization.dims());
  standardization.getMean(mean);
  standardization.getStd(std);
  j["mean"] = RealVectorView(mean);
  j["std"] = RealVectorView(std);
  j["cols"] = standardization.dims();
}

bool check_json(const nlohmann::json &j, const Standardization &) {
  return fluid::check_json(j,
    {"cols", "mean", "std"},
    {JSONTypes::NUMBER, JSONTypes::ARRAY, JSONTypes::ARRAY}
  );
}

void from_json(const nlohmann::json &j, Standardization &standardization) {
  index cols = j.at("cols").get<index>();
  RealVector mean(cols);
  RealVector std(cols);
  j.at("mean").get_to(mean);
  j.at("std").get_to(std);
  standardization.init(mean, std);
}

// PCA
void to_json(nlohmann::json &j, const PCA &pca) {
  index rows = pca.dims();
  index cols = pca.size();
  RealMatrix bases(rows, cols);
  RealVector values(cols);
  RealVector mean(rows);
  index numPoints = pca.getNumDataPoints();
  pca.getBases(bases);
  pca.getValues(values);
  pca.getMean(mean);
  j["bases"] = RealMatrixView(bases);
  j["values"] = RealVectorView(values);
  j["mean"] = RealVectorView(mean);
  j["numpoints"] = numPoints;
  j["rows"] = rows;
  j["cols"] = cols;
}

bool check_json(const nlohmann::json &j, const PCA &) {
  return fluid::check_json(j,
    {"rows","cols", "bases", "values", "mean"},
    {JSONTypes::NUMBER, JSONTypes::NUMBER, JSONTypes::ARRAY, JSONTypes::ARRAY, JSONTypes::ARRAY}
  );
}

void from_json(const nlohmann::json &j, PCA &pca) {

  index rows = j.at("rows").get<index>();
  index cols = j.at("cols").get<index>();
  index numPoints = 2;// default for backwards compatibility

  RealMatrix bases(rows, cols);
  RealVector mean(rows);
  RealVector values(cols);
  j.at("mean").get_to(mean);
  j.at("values").get_to(values);
  j.at("bases").get_to(bases);
  if (j.contains("numpoints")){
    j.at("numpoints").get_to(numPoints);
  }
  pca.init(bases, values, mean, numPoints);
}


// LabelSetEncoder
void to_json(nlohmann::json& j, const LabelSetEncoder& lse) {
  FluidTensor<std::string, 1> labels(lse.numLabels());
  lse.getLabels(labels);
  j["labels"] = FluidTensorView<std::string, 1>(labels);
  j["rows"] = labels.size();
}

bool check_json(const nlohmann::json &j, const LabelSetEncoder &) {
  return fluid::check_json(j,
    {"rows", "labels", "bases", "mean"},
    {JSONTypes::NUMBER, JSONTypes::ARRAY}
  );
}

void from_json(const nlohmann::json &j, LabelSetEncoder &lse) {
  index rows = j["rows"].get<index>();
  FluidTensor<std::string, 1> labels(rows);
  j.at("labels").get_to(labels);
  lse.init(labels);
}

// MLP
void to_json(nlohmann::json &j, const MLP &mlp) {
  using namespace std;
  for (index i = 0; i < mlp.size(); i++){
    nlohmann::json layer;
    index rows = mlp.inputSize(i);
    index cols = mlp.outputSize(i + 1);
    RealMatrix W(rows, cols);
    RealVector b(cols);
    index a;
    mlp.getParameters(i, W, b, a);
    layer["weights"] = RealMatrixView(W);
    layer["biases"] = RealVectorView(b);
    layer["activation"] = a;
    layer["rows"] =  rows;
    layer["cols"] = cols;
    j["layers"].push_back(layer);
  }
}

//TODO
bool check_json(const nlohmann::json &j, const MLP &) {
  return fluid::check_json(j,
    {"layers"},
    {JSONTypes::ARRAY}
  );
}

void from_json(const nlohmann::json &j, MLP &mlp) {
  using namespace std;
  index nLayers = asSigned(j["layers"].size());
  if(nLayers <= 0) return;
  index inputSize = j["layers"][0]["rows"].get<index>();
  index outputSize = j["layers"][asUnsigned(nLayers - 1)]["cols"].get<index>();
  index activation = j["layers"][0]["activation"].get<index>();
  index finalActivation = j["layers"][asUnsigned(nLayers - 1)]["activation"].get<index>();
  FluidTensor<index, 1> hiddenSizes(asSigned(j["layers"].size()) - 1);
  if(nLayers > 1){
    for (index i = 0; i < nLayers - 1; i++){
      hiddenSizes(i) =  j["layers"][asUnsigned(i)]["cols"].get<index>();
    }
  }
  mlp.init(inputSize,outputSize, hiddenSizes, activation, finalActivation);
  for (index i = 0; i < nLayers; i++){
    auto l = j["layers"][asUnsigned(i)];
    index rows = l["rows"].get<index>();
    index cols = l["cols"].get<index>();
    RealMatrix W(rows, cols);
    l.at("weights").get_to(W);
    RealVector b(cols);
    l.at("biases").get_to(b);
    index a = l.at("activation").get<index>();
    mlp.setParameters(i, W, b, a);
  }
  mlp.setTrained(true);
}

// UMAP
void to_json(nlohmann::json &j, const UMAP &umap) {
  RealMatrix embedding(umap.size(), umap.dims());
  umap.getEmbedding(embedding);
  j["embedding"] = RealMatrixView(embedding);
  j["rows"] = embedding.rows();
  j["cols"] = embedding.cols();
  j["tree"] = umap.getTree();
  j["a"] = umap.getA();
  j["b"] = umap.getB();
  j["k"] = umap.getK();
}

bool check_json(const nlohmann::json &j, const UMAP &) {
  return fluid::check_json(j,
    {"rows", "cols", "embedding", "a", "b", "k"},
    {JSONTypes::NUMBER, JSONTypes::NUMBER,JSONTypes::ARRAY,
      JSONTypes::NUMBER, JSONTypes::NUMBER, JSONTypes::NUMBER}
  );
}

void from_json(const nlohmann::json &j, UMAP &umap) {
  index rows = j.at("rows").get<index>();
  index cols = j.at("cols").get<index>();
  RealMatrix embedding(rows, cols);
  j.at("embedding").get_to(embedding);
  KDTree tree = j.at("tree").get<algorithm::KDTree>();
  double a = j.at("a").get<index>();
  double b = j.at("b").get<index>();
  index k = j.at("k").get<index>();
  umap.init(embedding, tree, k, a, b);
}

} // namespace algorithm

class JSONFile {
public:
  using json = nlohmann::json;
  using string = std::string;
  using fstream = std::fstream;

  JSONFile(string fileName, string rw) : mFileName(fileName), mRW(rw) {
    assert(rw == "r" || rw == "w");
    if (fileName.empty()) {
      mError = "Filename not specified";
      return;
    }
    if (mRW == "r") {
      openRead();
    } else if (mRW == "w") {
      openWrite();
    } else {
      mError = "Invalid read/write specifier";
    }
  }

  void openRead() {
    mFile.open(mFileName, fstream::in);
    if (mFile.fail())
      mError = "File not found";
  }

  void openWrite() {
    mFile.open(mFileName, fstream::out);
    if (mFile.fail())
      mError = "Could not open file for writing";
  }

  string error() { return mError; }

  bool ok() { return mError.empty(); }

  bool write(json data) {
    if (ok()) {
      mFile << data.dump(2) << std::endl;
      return mFile.good();
    }
    return false;
  }

  json read() {
    json result;
    if (ok()) {
      result = json::parse(mFile, nullptr, false);
      if (result.is_discarded())
        mError = "Error parsing JSON";
    }
    return result;
  }

private:
  fstream mFile;
  json mData;
  string mFileName;
  string mRW;
  string mError;
};

} // namespace fluid
