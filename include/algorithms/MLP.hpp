#pragma once

#include "NNFuncs.hpp"
#include "NNLayer.hpp"
#include "algorithms/util/FluidEigenMappings.hpp"
#include "data/FluidDataSet.hpp"
#include "data/FluidIndex.hpp"
#include "data/FluidTensor.hpp"
#include "data/TensorTypes.hpp"
#include <Eigen/Core>
#include <random>

namespace fluid {
namespace algorithm {

class MLP {
  using ArrayXd = Eigen::ArrayXd;
  using ArrayXXd = Eigen::ArrayXXd;

public:
  explicit MLP() = default;
  ~MLP() = default;

  void init(index inputSize, index outputSize,
            FluidTensor<index, 1> hiddenSizes, index activation) {
    mLayers.clear();
    std::vector<index> sizes = {inputSize};
    for (auto &&s : hiddenSizes)
      sizes.push_back(s);
    sizes.push_back(outputSize);
    for (index i = 0; i < sizes.size() - 1; i++) {
      mLayers.push_back(NNLayer(sizes[i], sizes[i + 1], activation));
    }
    for (auto &&l : mLayers)
      l.init();
    mInitialized = true;
    mTrained = false;
  }

  void reset(){
    for (auto &&l : mLayers)
      l.init();
    mInitialized = true;
    mTrained = false;
  }

  double loss(ArrayXXd pred, ArrayXXd out) {
    assert(pred.rows() == out.rows());
    return (pred - out).square().sum() / out.rows();
  }

  void process(RealMatrixView in, RealMatrixView out, index layerOutput) {
    using namespace _impl;
    using namespace Eigen;
    ArrayXXd input = asEigen<Eigen::Array>(in);
    ArrayXXd output = ArrayXXd::Zero(out.rows(), out.cols());
    forward(input, output, layerOutput);
    out = asFluid(output);
  }

  void processFrame(RealVectorView in, RealVectorView out, index layerOutput) {
    using namespace _impl;
    using namespace Eigen;
    ArrayXd tmpIn = asEigen<Eigen::Array>(in);
    ArrayXXd input(1, tmpIn.size());
    input.row(0) = tmpIn;
    ArrayXXd output = ArrayXXd::Zero(1, out.size());
    forward(input, output, layerOutput);
    ArrayXd tmpOut = output.row(0);
    out = asFluid(tmpOut);
  }

  void forward(Eigen::Ref<ArrayXXd> in, Eigen::Ref<ArrayXXd> out, index layer) {
    ArrayXXd input = in;
    ArrayXXd output;
    index nRows = input.rows();
    layer %= mLayers.size();
    for(index i = 0; i <= layer; i++){
      auto&& l = mLayers[i];
      output = ArrayXXd::Zero(input.rows(), l.outputSize());
      l.forward(input, output);
      input = output;
    }
    out = output;
  }

  void backward(Eigen::Ref<ArrayXXd> out) {
    index nRows = out.rows();
    ArrayXXd chain =
        ArrayXXd::Zero(nRows, mLayers[mLayers.size() - 1].inputSize());
    mLayers[mLayers.size() - 1].backward(out, chain);
    for (index i = mLayers.size() - 2; i >= 0; i--) {
      ArrayXXd tmp = ArrayXXd::Zero(nRows, mLayers[i].inputSize());
      mLayers[i].backward(chain, tmp);
      chain = tmp;
    }
  }

  void update(double learningRate, double momentum) {
    for (auto &&l : mLayers)
      l.update(learningRate, momentum);
  }

  index size() const { return mLayers.size(); }
  bool trained() const { return mTrained; }
  void setTrained(bool val) { mTrained = val;}
  index initialized() const { return mInitialized; }
  index outputSize(index layer = 0) const {
    if(layer >= mLayers.size()) return 0;
    else return mLayers[layer].outputSize();
  }

  index dims() const {
    return mLayers.size() == 0 ? 0 : mLayers[0].inputSize();
  }

  std::vector<NNLayer> mLayers;
  bool mInitialized{false};
  bool mTrained{false};
};
} // namespace algorithm
} // namespace fluid
