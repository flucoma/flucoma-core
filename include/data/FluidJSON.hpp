#pragma once

#include <data/FluidTensor.hpp>
#include <data/FluidIndex.hpp>
#include <fstream>
#include <nlohmann/json.hpp>

namespace fluid {

void to_json(nlohmann::json& j, const FluidDataSet<std::string, double, 1>& ds) {
  using namespace std;
  using namespace nlohmann;
  try {
  auto rowArray = json::array();
  auto ids = ds.getIds();
  auto data = ds.getData();
  for(index r = 0; r < ds.size();r++){
    auto row = data.row(r);
    auto rowV = vector<double>(row.begin(), row.end());
    auto rowObj = json::object({{"id",ids(r)},{"data",rowV}});
    rowArray.push_back(rowObj);
  }
  j["rows"] = ds.size();
  j["cols"] = ds.pointSize();
  j["data"] = rowArray;
  } catch (std::exception& e) {}
}

 void from_json(const nlohmann::json& j, FluidDataSet<std::string, double, 1>& ds){
   using namespace std;
   using namespace nlohmann;
   try {
     auto rowArray = j.at("data");
     index pointSize = j["cols"];
     ds.resize(pointSize);
     for(auto&& r:rowArray){
       vector<double> row = r.at("data");
       auto ftv = FluidTensorView<double, 1>(row.data(), 0, row.size());
       ds.add(r.at("id"), ftv);
     }
     } catch (std::exception& e) {}
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
