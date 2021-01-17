#pragma once
#include "../common/SharedClientUtils.hpp"
#include "NRTClient.hpp"
#include "data/FluidDataSet.hpp"
#include "data/FluidJSON.hpp"
#include <nlohmann/json.hpp>
#include <sstream>
#include <string>

namespace fluid {
namespace client {

template <typename T>
class DataClient{

public:
  using string = std::string;

  MessageResult<index> size() {
    return mAlgorithm.size();
  }

  MessageResult<index> dims() {
    return mAlgorithm.dims();
  }

  MessageResult<void> clear() {
    mAlgorithm.clear();
    return OK();
  }


  MessageResult<void> write(string fileName) {
    auto file = JSONFile(fileName, "w");
    file.write(mAlgorithm);
    return file.ok() ? OK() : Error(file.error());
  }

  MessageResult<void> read(string fileName) {
    auto file = JSONFile(fileName, "r");
    nlohmann::json j = file.read();
    if (!file.ok()) {
      return Error(file.error());
    } else {
      if(!check_json(j, mAlgorithm)) return Error("Invalid JSON format");
      mAlgorithm = j.get<T>();
    }
    return OK();
  }

  MessageResult<string> dump() {
    using namespace nlohmann;
    if (!mAlgorithm.initialized()) return string();
    nlohmann::json j = mAlgorithm;
    return j.dump();
  }

  MessageResult<void> load(string s) {
    using namespace std;
    using namespace nlohmann;
    json j = json::parse(s, nullptr, false);
    if (j.is_discarded()) {
      return Error("Parse error");
    } else {
      if(!check_json(j, mAlgorithm)) return Error("Invalid JSON format");
      mAlgorithm = j.get<T>();
      return OK();
    }
  }

protected:
  T mAlgorithm;
};

} // namespace client
} // namespace fluid
