#pragma once

#include <data/FluidTensor.hpp>
#include <data/FluidIndex.hpp>
#include <data/TensorTypes.hpp>
#include <data/FluidDataSet.hpp>
#include <algorithms/KDTree.hpp>
#include <fstream>
#include <nlohmann/json.hpp>

namespace fluid {

// FluidTensor
template <typename T>
void from_json(const nlohmann::json& j, FluidTensor<T, 1>& t) {
  std::vector<T> row = j;
  t = FluidTensorView<T, 1>(row.data(), 0, row.size());
}

template <typename T>
void to_json(nlohmann::json& j, const FluidTensorView<T, 1>& t) {
  j = std::vector<T>(t.begin(), t.end());
}

template <typename T>
void to_json(nlohmann::json& j, const FluidTensorView<const T, 1>& t) {
  j = std::vector<T>(t.begin(), t.end());
}

template <typename T>
void from_json(const nlohmann::json& j, FluidTensor<T, 2>& t) {
  //j = nlohmann::json::array();
  if(j.size() > 0){
    auto result = FluidTensor<T, 2>(j.size(), j[0].size());
    FluidTensor<T, 1> tmp(j[0].size());
    for(index i = 0; i < j.size(); i++){
      j[i].get_to(tmp);
      result.row(i) = tmp;
    }
    t = result;
  }
}

template <typename T>
void to_json(nlohmann::json& j, const FluidTensorView<T, 2>& t) {
  for(index i = 0; i < t.rows(); i++){
    j.push_back(t.row(i));
  }
}


// FluidDataSet
void to_json(nlohmann::json& j, const FluidDataSet<std::string, double, 1>& ds) {
  try {
    auto ids = ds.getIds();
    auto data = ds.getData();
    j["cols"] = ds.pointSize();
    for(index r = 0; r < ds.size();r++){
      j["data"][ids[r]] = data.row(r);
    }
  } catch (std::exception& e) {}
}

void from_json(const nlohmann::json& j, FluidDataSet<std::string, double, 1>& ds){
   using namespace nlohmann;
   try {
     auto rows = j.at("data");
     index pointSize = j["cols"];
     ds.resize(pointSize);
     FluidTensor<double, 1> tmp(pointSize);
     for (json::iterator r = rows.begin(); r != rows.end(); ++r) {
         r.value().get_to(tmp);
         ds.add(r.key(), tmp);
      }
    } catch (std::exception& e) {}
 }


namespace algorithm {
  // KDTree
 void to_json(nlohmann::json& j, const algorithm::KDTree tree) {
   try {
     KDTree::FlatData treeData = tree.toFlat();
     j["tree"] = FluidTensorView<index, 2>(treeData.tree);
     j["rows"] = treeData.data.rows();
     j["cols"] = treeData.data.cols();
     j["data"] = FluidTensorView<double, 2>(treeData.data);
     j["ids"] = FluidTensorView<std::string, 1>(treeData.ids);
   } catch (std::exception& e) {}
 }

 void from_json(const nlohmann::json& j, algorithm::KDTree& tree){
   index rows = j["rows"];
   index cols = j["cols"];
   KDTree::FlatData treeData(rows, cols);
   j["tree"].get_to(treeData.tree);
   j["data"].get_to(treeData.data);
   j["ids"].get_to(treeData.ids);
   tree.fromFlat(treeData);
}

}

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
     if (mRW == "r")
     {
       openRead();
     }
     else if (mRW == "w")
     {
       openWrite();
     }
     else {
       mError = "Invalid read/write specifier";
     }
   }

   void openRead() {
     mFile.open(mFileName, fstream::in);
     if (mFile.fail()) mError = "File not found";
   }

   void openWrite() {
     mFile.open(mFileName, fstream::out);
     if (mFile.fail()) mError = "Could not open file for writing";
   }
   string error() { return mError; }

   bool ok() {return mError.empty();}

   bool write(json data) {
     if (ok()) {
       try {
         mFile << data.dump(2) << std::endl;
       } catch (std::exception& e) {
         mError = "Invalid JSON";
       }
       return mFile.good();
     }
     return false;
   }

   json read() {
     json result;
     if (ok()) {
       try {
         mFile >> result;
       } catch (std::exception& e) {
         mError = "Error parsing JSON";
       }
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
