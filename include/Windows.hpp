#pragma once

#include <cmath> 
#include <vector>
#include <iostream>
#include <map>

namespace fluid {
namespace windows {

using std::vector;
using std::map;
using std::function;


enum class WindowType {Hann, Hamming};
using WindowFuncMap = map<WindowType, function<vector<double>(int)>>;

static WindowFuncMap windowFuncs = {
  {
    WindowType::Hann,  [] ( int size){
      vector<double> result(size);
      for(int i = 0;i< size; i++){
        result[i] = 0.5 - 0.5 * std::cos((M_PI * 2 * i) / size );
      }
      return result;
    }
  },
  {
    WindowType::Hamming,  [] (int size){
      vector<double> result(size);
      for(int i = 0;i< size; i++){
        result[i] = 0.54  - 0.46 * std::cos((M_PI * 2 * i) / size );
      }
      return result;
    }
  }
};
}
}
