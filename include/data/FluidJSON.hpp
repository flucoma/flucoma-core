#pragma once

#include <algorithms/KDTree.hpp>
#include <algorithms/KMeans.hpp>
#include <algorithms/Normalization.hpp>
#include <algorithms/PCA.hpp>
#include <algorithms/MLP.hpp>
#include <algorithms/Standardization.hpp>
#include <algorithms/LabelSetEncoder.hpp>
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
template <typename T>
void from_json(const nlohmann::json &j, FluidTensor<T, 1> &t) {
  if(!j.is_array()) return;
  for(auto&& el:j)if(!el.is_number()) return;
  std::vector<T> row = j;
  t = FluidTensorView<T, 1>(row.data(), 0, row.size());
}

void from_json(const nlohmann::json &j, FluidTensor<std::string, 1> &t) {
  if(!j.is_array()) return;
  for(auto&& el:j)if(!el.is_string()) return;
  std::vector<std::string> row = j;
  t = FluidTensorView<std::string, 1>(row.data(), 0, row.size());
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
    auto result = FluidTensor<T, 2>(j.size(), j[0].size());
    FluidTensor<T, 1> tmp(j[0].size());
    for (size_t i = 0; i < j.size(); i++) {
      j.at(i).get_to(tmp);
      result.row(asSigned(i)) = tmp;
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
  index pointSize = j.at("cols");
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
  index rows = j.at("rows");
  index cols = j.at("cols");
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
  index rows = j.at("rows");
  index cols = j.at("cols");
  RealMatrix means(rows, cols);
  j.at("means").get_to(means);
  kmeans.init(rows, cols);
  kmeans.setMeans(means);

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
  index cols = j.at("cols");
  RealVector dataMin(cols);
  RealVector dataMax(cols);
  j.at("data_min").get_to(dataMin);
  j.at("data_max").get_to(dataMax);
  double min = j.at("min");
  double max = j.at("max");
  normalization.init(min, max, dataMin, dataMax);
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
  index cols = j.at("cols");
  RealVector mean(cols);
  RealVector std(cols);
  j.at("mean").get_to(mean);
  j.at("std").get_to(std);
  standardization.init(mean, std);
}

// PCA
void to_json(nlohmann::json &j, const PCA &pca) {
  index dims = pca.dims();
  index k = pca.size();
  RealMatrix bases(dims, k);
  RealVector mean(dims);
  pca.getBases(bases);
  pca.getMean(mean);
  j["bases"] = RealMatrixView(bases);
  j["mean"] = RealVectorView(mean);
  j["cols"] = k;
  j["rows"] = dims;
}

bool check_json(const nlohmann::json &j, const PCA &) {
  return fluid::check_json(j,
    {"rows", "cols", "bases", "mean"},
    {JSONTypes::NUMBER, JSONTypes::NUMBER,
      JSONTypes::ARRAY, JSONTypes::ARRAY}
  );
}

void from_json(const nlohmann::json &j, PCA &pca) {
  index k = j.at("cols");
  index dims = j.at("rows");
  RealMatrix bases(dims, k);
  RealVector mean(dims);
  j.at("mean").get_to(mean);
  j.at("bases").get_to(bases);
  pca.init(bases, mean);
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
  index rows = j.at("rows");
  FluidTensor<std::string, 1> labels;
  j.at("labels").get_to(labels);
  lse.init(labels);
}

// MLP
void to_json(nlohmann::json &j, const MLP &mlp) {
  using namespace std;
  for (index i = 0; i < mlp.size(); i++){
    nlohmann::json layer;
    index rows = mlp.inputSize(i);
    index cols = mlp.outputSize(i);
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
  index nLayers = j["layers"].size();
  if(nLayers <= 0) return;
  index inputSize = j["layers"][0]["rows"];
  index outputSize = j["layers"][nLayers - 1]["cols"];
  index activation = j["layers"][0]["activation"];
  index finalActivation = j["layers"][nLayers - 1]["activation"];
  FluidTensor<index, 1> hiddenSizes(j["layers"].size() - 1);
  if(nLayers > 1){
    for (index i = 0; i < nLayers - 1; i++){
      hiddenSizes(i) =  j["layers"][i]["cols"];
    }
  }
  mlp.init(inputSize,outputSize, hiddenSizes, activation, finalActivation);
  for (index i = 0; i < nLayers; i++){
    auto l = j["layers"][i];
    index rows = l["rows"];
    index cols = l["cols"];
    RealMatrix W(rows, cols);
    l.at("weights").get_to(W);
    RealVector b(cols);
    l.at("biases").get_to(b);
    index a = l.at("activation");
    mlp.setParameters(i, W, b, a);
  }
  mlp.setTrained(true);
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
  bool mValid{false};
  string mError;
};

} // namespace fluid
