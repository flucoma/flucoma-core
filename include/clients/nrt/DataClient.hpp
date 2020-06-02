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

  DataClient(T& object):mObject(object){}

  MessageResult<index> size() {
    return mObject.size();
  }

  MessageResult<index> dims() {
    return mObject.dims();
  }

  MessageResult<void> write(string fileName) {
    auto file = JSONFile(fileName, "w");
    file.write(mObject);
    return file.ok() ? OK() : Error(file.error());
  }

  MessageResult<void> read(string fileName) {
    auto file = JSONFile(fileName, "r");
    nlohmann::json j = file.read();
    if (!file.ok()) {
      return Error(file.error());
    } else {
      if(!check_json(j, mObject)) return Error("Invalid JSON format");
      mObject = j.get<T>();
    }
    return OK();
  }

  MessageResult<string> dump() {
    using namespace nlohmann;
    nlohmann::json j = mObject;
    return j.dump();
  }

  MessageResult<void> load(string s) {
    using namespace std;
    using namespace nlohmann;
    json j = json::parse(s, nullptr, false);
    if (j.is_discarded()) {
      return Error("Parse error");
    } else {
      if(!check_json(j, mObject)) return Error("Invalid JSON format");
      mObject = j.get<T>();
      return OK();
    }
  }

private:
  T& mObject;
};

} // namespace client
} // namespace fluid
