#pragma once

#include <cassert>
#include <cmath>
#include <functional>
#include <iostream>
#include <map>
#include <vector>

namespace fluid {
namespace algorithm {

enum class WindowType { kHann, kHamming, kBlackmanHarris, kGaussian };
using WindowFuncMap = std::map<WindowType, std::function<std::vector<double>(int)>>;

static WindowFuncMap windowFuncs = {
    {WindowType::kHann,
     [](int size) {
       std::vector<double> result(size);
       for (int i = 0; i < size; i++) {
         result[i] = 0.5 - 0.5 * std::cos((M_PI * 2 * i) / size);
       }
       return result;
     }},
    {WindowType::kHamming,
     [](int size) {
       std::vector<double> result(size);
       for (int i = 0; i < size; i++) {
         result[i] = 0.54 - 0.46 * std::cos((M_PI * 2 * i) / size);
       }
       return result;
     }},
    {WindowType::kBlackmanHarris,
     [](int size) {
       using std::cos;
       std::vector<double> result(size);
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
       std::vector<double> result(size);
       for (int i = -h; i <= h; i++) {
         result[i + h] = exp(-i * i / (2 * sigma * sigma));
       }
       return result;
     }}};
} // namespace algorithm
} // namespace fluid
