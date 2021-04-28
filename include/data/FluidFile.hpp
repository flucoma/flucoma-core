#pragma once

#include <data/FluidTensor.hpp>
#include <data/FluidIndex.hpp>
#include <fstream>
#include <nlohmann/json.hpp>

namespace fluid {

class FluidFile {

public:
  using json = nlohmann::json;
  using string = std::string;
  using fstream = std::fstream;

  FluidFile(string fileName, string rw) : mFileName(fileName), mRW(rw) {
    assert(rw == "r" || rw == "w");
    if (fileName.empty()) {
      mValid = false;
      mError = "Filename not specified";
    }
    if (mRW == "r")
      openRead();
    else if (mRW == "w")
      openWrite();
    else {
      mValid = false;
      mError = "Invalid read/write specifier";
    }
  }

  void openRead() {
    mFile.open(mFileName, fstream::in);
    if (mFile.fail()) {
      mValid = false;
      mError = "File not found";
    } else
      mValid = true;
  }

  void openWrite() {
    mFile.open(mFileName, fstream::out);
    if (mFile.fail()) {
      mValid = false;
      mError = "Could not open file for writing";
    } else
      mValid = true;
  }

  bool checkKeys(std::vector<string> keys) {
    for (auto &k : keys) {
      auto pos = mData.find(k);
      if (pos == mData.end()) {
        mError = "Invalid file format";
        return false;
      }
    }
    return true;
  }

  bool valid() { return mValid; }

  string error() { return mError; }

  bool write() {
    if (mValid) {
      mFile << mData.dump(2) << std::endl;
      if (mFile.good())
        return true;
    }
    return false;
  }

  bool read() {
    if (mValid) {
      mFile >> mData;
      if (mFile.good())
        return true;
    }
    return false;
  }

  void add(string key, index value) { mData[key] = value; }

  void add(string key, double value) { mData[key] = value; }

  void add(string key, string value) { mData[key] = value; }

  void add(string key, FluidTensorView<string, 1> value) {
    mData[key] = std::vector<string>(value.begin(), value.end());
  }

  void add(string key, FluidTensorView<double, 2> value) {
    mData[key] = std::vector<double>(value.begin(), value.end());
  }

  void add(string key, FluidTensorView<double, 1> value) {
    mData[key] = std::vector<double>(value.begin(), value.end());
  }

  void add(string key, FluidTensorView<index, 2> value) {
    mData[key] = std::vector<int>(value.begin(), value.end());
  }

  void add(string key, FluidTensorView<string, 2> value) {
    mData[key] = std::vector<string>(value.begin(), value.end());
  }

  void get(string key, index &value) { value = mData[key]; }

  void get(string key, double &value) { value = mData[key]; }

  void get(string key, string &value) { value = mData[key]; }

  void get(string key, FluidTensorView<string, 1> value, index rows) {
    std::vector<string> tmp = mData[key];
    value = FluidTensorView<string, 1>{tmp.data(), 0, rows};
  }

  void get(string key, FluidTensorView<double, 2> value, index rows,
           index cols) {
    std::vector<double> tmp = mData[key];
    value = FluidTensorView<double, 2>{tmp.data(), 0, rows, cols};
  }

  void get(string key, FluidTensorView<double, 1> value, index rows) {
    std::vector<double> tmp = mData[key];
    value = FluidTensorView<double, 1>{tmp.data(), 0, rows};
  }


  void get(string key, FluidTensorView<index, 2> value, index rows,
           index cols) {
    std::vector<int> tmp = mData[key];
    value = FluidTensorView<int, 2>{tmp.data(), 0, rows, cols};
  }

  void get(string key, FluidTensorView<string, 2> value, index rows,
           index cols) {
    std::vector<string> tmp = mData[key];
    value = FluidTensorView<string, 2>{tmp.data(), 0, rows, cols};
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
