#pragma once

#include <cassert>
#include <cmath>
#include <iostream>
#include <map>
#include <Eigen/Core>

namespace fluid {
namespace algorithm {


enum class WindowTypes { kHann, kHamming, kBlackmanHarris, kGaussian };
using WindowFuncsMap = std::map<WindowTypes, std::function<void(int,  Eigen::Ref<Eigen::ArrayXd>)>>;

static WindowFuncsMap windows = {
    {WindowTypes::kHann,
     [](int size, Eigen::Ref<Eigen::ArrayXd> out) {
       for (int i = 0; i < size; i++) {
         out(i) = 0.5 - 0.5 * std::cos((M_PI * 2 * i) / size);
       }
     }},
    {WindowTypes::kHamming,
     [](int size, Eigen::Ref<Eigen::ArrayXd> out) {
       for (int i = 0; i < size; i++) {
         out(i) = 0.54 - 0.46 * std::cos((M_PI * 2 * i) / size);
       }
     }},
    {WindowTypes::kBlackmanHarris,
     [](int size, Eigen::Ref<Eigen::ArrayXd> out) {
       using std::cos;
       for (int i = 0; i < size; i++) {
         out(i) = 0.35875 - 0.48829 * cos((M_PI * 2 * i) / size) +
                     0.14128 * cos((M_PI * 2 * i) / size) +
                     0.01168 * cos((M_PI * 2 * i) / size);
       }
     }},
    {WindowTypes::kGaussian,
      [](int size, Eigen::Ref<Eigen::ArrayXd> out) {
       using std::exp;
       double sigma = size / 3; // TODO: should be argument
       assert(size % 2);
       int h = (size - 1) / 2;
       for (int i = -h; i <= h; i++) {
         out(i + h) = exp(-i * i / (2 * sigma * sigma));
       }
     }}};
} // namespace algorithm
} // namespace fluid
