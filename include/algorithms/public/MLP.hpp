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

#include "../util/FluidEigenMappings.hpp"
#include "../util/NNFuncs.hpp"
#include "../util/NNLayer.hpp"
#include "../../data/FluidDataSet.hpp"
#include "../../data/FluidIndex.hpp"
#include "../../data/FluidTensor.hpp"
#include "../../data/TensorTypes.hpp"
#include "../../data/FluidMemory.hpp"
#include <Eigen/Core>
#include <random>

namespace fluid {
namespace algorithm {

class MLP
{
  using ArrayXd = Eigen::ArrayXd;
  using ArrayXXd = Eigen::ArrayXXd;

public:
  explicit MLP() = default;
  ~MLP() = default;

  void init(index inputSize, index outputSize,
            FluidTensor<index, 1> hiddenSizes, index hiddenAct, index outputAct)
  {
    mLayers.clear();
    std::vector<index> sizes = {inputSize};
    std::vector<index> activations = {};
    for (auto&& s : hiddenSizes)
    {
      sizes.push_back(s);
      activations.push_back(hiddenAct);
    }
    sizes.push_back(outputSize);
    mMaxLayerSize = *std::max_element(sizes.begin(), sizes.end());
    activations.push_back(outputAct);
    for (index i = 0; i < asSigned(sizes.size() - 1); i++)
    {
      mLayers.push_back(NNLayer(sizes[asUnsigned(i)], sizes[asUnsigned(i + 1)],
                                activations[asUnsigned(i)]));
    }
    for (auto&& l : mLayers) l.init();
    mInitialized = true;
    mTrained = false;
  }

  void getParameters(index layer, RealMatrixView W, RealVectorView b,
                     index& layerType) const
  {
    using namespace _impl;
    W <<= asFluid(mLayers[asUnsigned(layer)].getWeights());
    b <<= asFluid(mLayers[asUnsigned(layer)].getBiases());
    layerType = mLayers[asUnsigned(layer)].getActType();
  }

  void setParameters(index layer, RealMatrixView W, RealVectorView b,
                     index layerType)
  {
    using namespace Eigen;
    using namespace std;
    using namespace _impl;
    MatrixXd weights = asEigen<Matrix>(W);
    VectorXd biases = asEigen<Matrix>(b);
    mLayers[asUnsigned(layer)].init(weights, biases, layerType);
  }

  void clear()
  {
    for (auto&& l : mLayers) l.init();
    mInitialized = false;
    mTrained = false;
  }

  double loss(ArrayXXd pred, ArrayXXd out)
  {
    assert(pred.rows() == out.rows());
    return (pred - out).square().sum() / out.rows();
  }

  void process(RealMatrixView in, RealMatrixView out, index startLayer,
               index endLayer)
  {
    using namespace _impl;
    using namespace Eigen;
    ArrayXXd input = asEigen<Eigen::Array>(in);
    ArrayXXd output = ArrayXXd::Zero(out.rows(), out.cols());
    forward(input, output, startLayer, endLayer);
    out <<= asFluid(output);
  }

  void processFrame(RealVectorView in, RealVectorView out, index startLayer,
                    index      endLayer,
                    Allocator& alloc = FluidDefaultAllocator()) const
  {
    using namespace _impl;
    using namespace Eigen;
    ScopedEigenMap<ArrayXd> input(in.size(), alloc);
    input = asEigen<Eigen::Array>(in);
    ScopedEigenMap<ArrayXd> output(out.size(), alloc);
    forwardFrame(input, output, startLayer, endLayer, alloc);
    asEigen<Array>(out) = output;
  }

  void forward(Eigen::Ref<ArrayXXd> in, Eigen::Ref<ArrayXXd> out) const
  {
    forward(in, out, 0, asSigned(mLayers.size()));
  }

  void forward(Eigen::Ref<ArrayXXd> in, Eigen::Ref<ArrayXXd> out,
               index startLayer, index endLayer) const
  {
    if (startLayer >= asSigned(mLayers.size()) ||
        endLayer > asSigned(mLayers.size()))
      return;
    if (startLayer < 0 || endLayer <= 0) return;
    ArrayXXd input = in;
    ArrayXXd output;
    for (index i = startLayer; i < endLayer; i++)
    {
      auto&& l = mLayers[asUnsigned(i)];
      output = ArrayXXd::Zero(input.rows(), l.outputSize());
      l.forward(input, output);
      input = output;
    }
    out = output;
  }

  void forwardFrame(Eigen::Ref<ArrayXd> in, Eigen::Ref<ArrayXd> out,
                    index startLayer, index endLayer,
                    Allocator& alloc = FluidDefaultAllocator()) const
  {
    if (startLayer >= asSigned(mLayers.size()) ||
        endLayer > asSigned(mLayers.size()))
      return;
    if (startLayer < 0 || endLayer <= 0) return;
    ScopedEigenMap<ArrayXd> input(mMaxLayerSize, alloc);
    input.head(in.size()) = in;
    ScopedEigenMap<ArrayXd> output(mMaxLayerSize, alloc);
    index                   inSize = in.size();
    for (index i = startLayer; i < endLayer; i++)
    {
      auto& l = mLayers[asUnsigned(i)];
      auto  inputBlock = input.head(inSize);
      auto  outputBlock = output.head(l.outputSize());
      outputBlock.setZero();
      l.forwardFrame(inputBlock, outputBlock, alloc);
      input.head(l.outputSize()) = outputBlock;
      inSize = l.outputSize();
    }
    out = output.head(out.size());
  }

  void backward(Eigen::Ref<ArrayXXd> out) 
  {
    index    nRows = out.rows();
    ArrayXXd chain =
        ArrayXXd::Zero(nRows, mLayers[mLayers.size() - 1].inputSize());
    mLayers[mLayers.size() - 1].backward(out, chain);
    for (index i = asSigned(mLayers.size() - 2); i >= 0; i--)
    {
      ArrayXXd tmp = ArrayXXd::Zero(nRows, mLayers[asUnsigned(i)].inputSize());
      mLayers[asUnsigned(i)].backward(chain, tmp);
      chain = tmp;
    }
  }

  void update(double learningRate, double momentum)
  {
    for (auto&& l : mLayers) l.update(learningRate, momentum);
  }

  index size() const { return asSigned(mLayers.size()); }
  bool  trained() const { return mTrained; }
  void  setTrained(bool val) { mTrained = val; }
  index initialized() const { return mInitialized; }

  // 0 = size of the input, 1 = output size of first hidden
  index outputSize(index layer) const
  {
    if (layer == 0) return mLayers[0].inputSize();
    if (layer < 0 || layer > asSigned(mLayers.size())) return 0;
    return mLayers[asUnsigned(layer - 1)].outputSize();
  }

  index inputSize(index layer) const
  {
    return (layer >= asSigned(mLayers.size()) || layer < 0)
               ? 0
               : mLayers[asUnsigned(layer)].inputSize();
  }

  index hiddenActivation() const{
    return mLayers.size() == 0 ? 0 : mLayers[0].getActType();
  }

  index outputActivation() const{
    return mLayers.size() == 0 ? 0 : mLayers[mLayers.size() - 1].getActType();
  }
  
  index dims() const
  {
    return mLayers.size() == 0 ? 0 : mLayers[0].inputSize();
  }

  std::vector<NNLayer> mLayers;
  bool                 mInitialized{false};
  bool                 mTrained{false};
  index mMaxLayerSize;
};
} // namespace algorithm
} // namespace fluid
