#pragma once

#include <data/FluidTensor.hpp>
#include <data/FluidIndex.hpp>
#include <fstream>
#include <nlohmann/json.hpp>

namespace fluid {

void to_json(nlohmann::json& j, const FluidTensorView<double, 1>& t) {
  j = std::vector<double>(t.begin(), t.end());
}

void from_json(const nlohmann::json& j, FluidTensor<double, 1>& t) {
  std::vector<double> row = j;
  t = FluidTensorView<double, 1>(row.data(), 0, row.size());
}

void to_json(nlohmann::json& j, const FluidDataSet<std::string, double, 1>& ds) {
  try {
    auto ids = ds.getIds();
    auto data = ds.getData();
    j["rows"] = ds.size();
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
