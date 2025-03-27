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
#include "NRTClient.hpp"
#include "../common/SharedClientUtils.hpp"
#include "../../data/FluidDataSet.hpp"
#include "../../data/FluidJSON.hpp"
#include <nlohmann/json.hpp>
#include <sstream>
#include <string>

namespace fluid {
namespace client {

template <typename T>
class DataClient
{

public:
  using string = std::string;

  MessageResult<index> size() const { return mAlgorithm.size(); }

  MessageResult<index> dims() const { return mAlgorithm.dims(); }

  MessageResult<void> clear()
  {
    mAlgorithm.clear();
    return OK();
  }


  MessageResult<void> write(string fileName)
  {
    auto file = JSONFile(fileName, "w");
    file.write(mAlgorithm);
    return file.ok() ? OK() : Error(file.error());
  }

  MessageResult<void> read(string fileName)
  {
    auto           file = JSONFile(fileName, "r");
    nlohmann::json j = file.read();
    if (!file.ok()) { return Error(file.error()); }
    else
    {
      if (!check_json(j, mAlgorithm)) return Error("Invalid JSON format");
      mAlgorithm = j.get<T>();
    }
    return OK();
  }

  MessageResult<string> dump()
  {
    using namespace nlohmann;
    if (!mAlgorithm.initialized()) return string();
    nlohmann::json j = mAlgorithm;
    return j.dump();
  }

  MessageResult<void> load(string s)
  {
    using namespace std;
    using namespace nlohmann;
    json j = json::parse(s, nullptr, false);
    if (j.is_discarded()) { return Error("Parse error"); }
    else
    {
      if (!check_json(j, mAlgorithm)) return Error("Invalid JSON format");
      mAlgorithm = j.get<T>();
      return OK();
    }
  }
  
  bool initialized() const { return mAlgorithm.initialized(); }
  T const& algorithm() const { return mAlgorithm; }
protected:
  T mAlgorithm;
};

} // namespace client
} // namespace fluid
