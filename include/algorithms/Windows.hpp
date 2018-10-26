#pragma once

#include <cmath>
#include <iostream>
#include <map>
#include <vector>

namespace fluid {
namespace windows {

using std::function;
using std::map;
using std::vector;

enum class WindowType { kHann, kHamming, kBlackmanHarris, kGaussian };
using WindowFuncMap = map<WindowType, function<vector<double>(int)>>;

static WindowFuncMap windowFuncs = {
    {WindowType::kHann,
     [](int size) {
       vector<double> result(size);
       for (int i = 0; i < size; i++) {
         result[i] = 0.5 - 0.5 * std::cos((M_PI * 2 * i) / size);
       }
       return result;
     }},
    {WindowType::kHamming,
     [](int size) {
       vector<double> result(size);
       for (int i = 0; i < size; i++) {
         result[i] = 0.54 - 0.46 * std::cos((M_PI * 2 * i) / size);
       }
       return result;
     }},
    {WindowType::kBlackmanHarris,
     [](int size) {
       using std::cos;
       vector<double> result(size);
       for (int i = 0; i < size; i++) {
         result[i] = 0.35875 - 0.48829 * cos((M_PI * 2 * i) / size) +
                     0.14128 * cos((M_PI * 2 * i) / size) +
                     0.01168 * cos((M_PI * 2 * i) / size);
       }
       return result;
     }},
    {WindowType::kGaussian, [](int size) {
       using std::exp;
       double sigma = size / 3; // TODO: should be argument
       assert(size % 2);
       int h = (size - 1) / 2;
       vector<double> result(size);
       for (int i = -h; i <= h; i++) {
         result[i + h] = exp(-i * i / (2 * sigma * sigma));
       }
       return result;
     }}};
} // namespace windows
} // namespace fluid
