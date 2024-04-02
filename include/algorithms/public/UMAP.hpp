#pragma once
#include "KDTree.hpp"
#include "../util/DistanceFuncs.hpp"
#include "../util/FluidEigenMappings.hpp"
#include "../util/SpectralEmbedding.hpp"
#include "../../data/TensorTypes.hpp"
#include "../../data/FluidMemory.hpp"
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <cassert>
#include <cmath>
#include <random>
#include <unsupported/Eigen/NonLinearOptimization>
#include <unsupported/Eigen/NumericalDiff>


namespace fluid {
namespace algorithm {

namespace impl {
template <typename RefeferenceArray>
void optimizeLayout(Eigen::ArrayXXd& embedding, RefeferenceArray& reference,
                    Eigen::ArrayXi const&  embIndices,
                    Eigen::ArrayXi const&  refIndices,
                    Eigen::ArrayXd const&  epochsPerSample,
                    Eigen::VectorXd const& AB, bool updateReference,
                    double learningRate, index maxIter, double gamma = 1.0)
{
  using namespace std;
  using namespace Eigen;
  double alpha = learningRate;
  double negativeSampleRate = 5.0;
  auto   distance = DistanceFuncs::map()[DistanceFuncs::Distance::kSqEuclidean];
  double a = AB(0);
  double b = AB(1);
  random_device                   rd;
  mt19937                         mt(rd());
  uniform_int_distribution<index> randomInt(0, reference.rows() - 1);
  ArrayXd epochsPerNegativeSample = epochsPerSample / negativeSampleRate;
  ArrayXd nextEpoch = epochsPerSample;
  ArrayXd nextNegEpoch = epochsPerNegativeSample;
  ArrayXd bound = VectorXd::Constant(embedding.cols(),
                                     4); // based on umap python implementation
  for (index i = 0; i < maxIter; i++)
  {
    for (index j = 0; j < epochsPerSample.size(); j++)
    {
      if (nextEpoch(j) > i) continue;
      ArrayXd current = embedding.row(embIndices(j));
      ArrayXd other = reference.row(refIndices(j));
      double  dist = distance(current, other); // todo: try to have dist member
      double  gradCoef = 0;
      ArrayXd grad;
      if (dist > 0)
      {
        gradCoef = -2.0 * a * b * pow(dist, b - 1.0);
        gradCoef /= a * pow(dist, b) + 1.0;
      }
      grad = (gradCoef * (current - other)).cwiseMin(bound).cwiseMax(-bound);
      current += grad * alpha;
      if constexpr (!std::is_const_v<RefeferenceArray>)
        if (updateReference) other += -grad * alpha;
      nextEpoch(j) += epochsPerSample(j);
      index numNegative = static_cast<index>((i - nextNegEpoch(j)) /
                                             epochsPerNegativeSample(j));
      for (index k = 0; k < numNegative; k++)
      {
        index negativeIndex = randomInt(mt);
        if (negativeIndex == embIndices(j)) continue;
        ArrayXd negative = reference.row(negativeIndex);
        dist = distance(current, negative);
        gradCoef = 0;
        grad = VectorXd::Constant(reference.cols(), 4.0);
        if (dist > 0)
        {
          gradCoef = 2.0 * gamma * b;
          gradCoef /= (0.001 + dist) * (a * pow(dist, b) + 1);
          grad = (gradCoef * (current - negative))
                     .cwiseMin(bound)
                     .cwiseMax(-bound);
        }
        current += grad * alpha;
      }
      nextNegEpoch(j) += numNegative * epochsPerNegativeSample(j);
      embedding.row(embIndices(j)) = current;
      if constexpr (!std::is_const_v<RefeferenceArray>)
        if (updateReference) reference.row(refIndices(j)) = other;
    }
    alpha = learningRate * (1.0 - (i / double(maxIter)));
  }
}
} // namespace impl

struct UMAPEmbeddingParamsFunctor
{
  typedef double Scalar;
  enum {
    InputsAtCompileTime = 2,
    ValuesAtCompileTime = 300 // from UMAP python implementation
  };
  typedef Eigen::VectorXd InputType;
  typedef Eigen::VectorXd ValueType;
  typedef Eigen::MatrixXd JacobianType;

  UMAPEmbeddingParamsFunctor(double minDist, double spread = 1.0)
  {
    mX = Eigen::ArrayXd::LinSpaced(values(), 0, 3 * spread);
    mY = (mX <= minDist).select(1, ((-mX + minDist) / spread).exp());
  }

  int operator()(const Eigen::VectorXd& x, Eigen::VectorXd& fvec) const
  {
    fvec = mY - (1 / (1 + x(0) * mX.pow(2 * x(1))));
    return 0;
  }

  int values() const { return ValuesAtCompileTime; }
  int inputs() const { return InputsAtCompileTime; }

  Eigen::ArrayXd mX;
  Eigen::ArrayXd mY;
};

class UMAP
{
public:
  using ArrayXXd = Eigen::ArrayXXd;
  using ArrayXd = Eigen::ArrayXd;
  using ArrayXi = Eigen::ArrayXi;
  using VectorXd = Eigen::VectorXd;
  using SparseMatrixXd = Eigen::SparseMatrix<double>;
  using DataSet = FluidDataSet<std::string, double, 1>;
  template <typename T>
  using Ref = Eigen::Ref<T>;

  void init(RealMatrixView embedding, KDTree tree, index k, double a, double b)
  {
    mEmbedding = _impl::asEigen<Eigen::Array>(embedding);
    mTree = tree;
    mK = k;
    mAB = VectorXd(2);
    mAB << a, b;
    mInitialized = true;
  }

  void getEmbedding(RealMatrixView out) const
  {
    if (mInitialized) out <<= _impl::asFluid(mEmbedding);
  }

  double getA() const { return mInitialized ? mAB(0) : 0; }

  double getB() const { return mInitialized ? mAB(1) : 0; }

  index getK() const { return mInitialized ? mK : 0; }

  KDTree getTree() const { return mTree; }

  void clear()
  {
    mEmbedding.setZero();
    mTree.clear();
    mInitialized = false;
  }

  index dims() const { return mInitialized ? mEmbedding.cols() : 0; }

  index inputDims() const { return mInitialized ? mTree.dims() : 0; }

  index size() const { return mInitialized ? mEmbedding.rows() : 0; }

  bool initialized() const { return mInitialized; }

  DataSet train(DataSet& in, index k = 15, index dims = 2, double minDist = 0.1,
                index maxIter = 200, double learningRate = 1.0)
  {
    using namespace Eigen;
    using namespace _impl;
    using namespace std;
    SpectralEmbedding      spectralEmbedding;
    index                  n = in.size();
    FluidTensor<string, 1> ids{in.getIds()};
    FluidTensor<string, 1> newIds(n);
    for (index i = 0; i < n; i++) newIds(i) = to_string(i);
    mTree = KDTree(DataSet(newIds, in.getData()));
    SparseMatrixXd knnGraph = SparseMatrixXd(in.size(), in.size());
    ArrayXXd       dists = ArrayXXd::Zero(in.size(), k);
    mK = k;
    makeGraph(in, mK, knnGraph, dists, true);
    ArrayXd sigma = findSigma(k, dists);
    computeHighDimProb(dists, sigma, knnGraph);
    SparseMatrixXd knnGraphT = knnGraph.transpose();
    knnGraph = (knnGraph + knnGraphT) - knnGraph.cwiseProduct(knnGraphT);
    mAB = findAB(minDist);
    mEmbedding = spectralEmbedding.train(knnGraph, dims);
    mEmbedding = normalizeEmbedding(mEmbedding);
    knnGraph.makeCompressed();
    ArrayXi rowIndices(knnGraph.nonZeros());
    ArrayXi colIndices(knnGraph.nonZeros());
    ArrayXd epochsPerSample(knnGraph.nonZeros());
    getGraphIndices(knnGraph, rowIndices, colIndices);
    computeEpochsPerSample(knnGraph, epochsPerSample);
    epochsPerSample = (epochsPerSample == 0).select(-1, epochsPerSample);
    optimizeLayoutAndUpdate(mEmbedding, mEmbedding, rowIndices, colIndices,
                   epochsPerSample, learningRate, maxIter);
    DataSet out(ids, _impl::asFluid(mEmbedding));
    mInitialized = true;
    return out;
  }

  DataSet transform(DataSet& in, index maxIter = 200, double learningRate = 1.0) const
  {
    if (!mInitialized) return DataSet();
    SparseMatrixXd knnGraph(in.size(), mEmbedding.rows());
    ArrayXXd       dists = ArrayXXd::Zero(in.size(), mK);
    makeGraph(in, mK, knnGraph, dists, false);
    knnGraph.makeCompressed();
    ArrayXd sigma = findSigma(mK, dists);
    computeHighDimProb(dists, sigma, knnGraph);
    normalizeRows(knnGraph);
    ArrayXXd embedding =
        initTransformEmbedding(knnGraph, mEmbedding, in.size());
    ArrayXi rowIndices(knnGraph.nonZeros());
    ArrayXi colIndices(knnGraph.nonZeros());
    ArrayXd epochsPerSample(knnGraph.nonZeros());
    getGraphIndices(knnGraph, rowIndices, colIndices);
    computeEpochsPerSample(knnGraph, epochsPerSample);
    epochsPerSample = (epochsPerSample == 0).select(-1, epochsPerSample);
    optimizeLayout(embedding, mEmbedding, rowIndices, colIndices,
                   epochsPerSample, learningRate, maxIter);
    DataSet out(in.getIds(), _impl::asFluid(embedding));
    return out;
  }


  void transformPoint(RealVectorView in, RealVectorView out,
                      Allocator& alloc = FluidDefaultAllocator()) const
  {
    if (!mInitialized) return;

    auto [distances, nearestIds] = mTree.kNearest(in, mK, 0, alloc);
    using SparseMap = Eigen::Map<Eigen::SparseMatrix<double, Eigen::RowMajor>>;

    rt::vector<double> data(alloc);
    rt::vector<int>    inner(alloc);
    data.reserve(asUnsigned(mK));
    inner.reserve(asUnsigned(mK));

    rt::vector<int> outer(2, 0, alloc);
    outer[1] = static_cast<int>(mK);

    ScopedEigenMap<ArrayXXd> dists(1, mK, alloc);
    for (size_t j = 0; j < asUnsigned(mK); j++)
    {
      int neighborIndex = stoi(*nearestIds[j]);
      dists(0, asSigned(j)) = distances[j];
      data.push_back(distances[j]);
      inner.push_back(neighborIndex);
    }
    int       maxIndex = *std::max_element(inner.cbegin(), inner.cend());
    SparseMap knnGraph(1, maxIndex, mK, outer.data(), inner.data(),
                       data.data());
    ScopedEigenMap<ArrayXd> sigma = findSigma(mK, dists, 64, 1e-5, alloc);
    computeHighDimProb(dists, sigma, knnGraph);
    normalizeRows(knnGraph, alloc);
    ScopedEigenMap<ArrayXXd> embedding =
        initTransformEmbedding(knnGraph, mEmbedding, 1, alloc);
    _impl::asEigen<Eigen::Array>(out) = embedding.row(0).transpose();
  }


private:
  template <typename F, typename Derived>
  void traverseGraph(Eigen::SparseCompressedBase<Derived>& graph, F func) const
  {
    for (index i = 0; i < graph.outerSize(); i++)
    {
      for (typename Eigen::SparseCompressedBase<Derived>::InnerIterator it(
               graph, i);
           it; ++it)
      {
        func(it);
      }
    }
  }

  double loss(Ref<ArrayXXd> P, Ref<ArrayXXd> Y, double a, double b)
  {
    ArrayXXd D = DistanceMatrix(Y, 2);
    ArrayXXd Q = 1 / (1 + a * D.pow(b));
    Q = Q + epsilon;
    ArrayXXd CE =
        -P * (Q + 0.01).log() - (1 - P) * (1e-6 + (1 - Q + 0.01)).log();
    return CE.sum();
  }

  ScopedEigenMap<ArrayXd>
  findSigma(index k, Ref<ArrayXXd> dists, index maxIter = 64,
            double     tolerance = 1e-5,
            Allocator& alloc = FluidDefaultAllocator()) const
  {
    using namespace std;
    double                  target = log2(k);
    ScopedEigenMap<ArrayXd> result(dists.rows(), alloc);
    result.setZero();
    for (index i = 0; i < dists.rows(); i++)
    {
      index  iter = maxIter;
      double lo = 0;
      double hi = infinity;
      double mid = 1.0;
      double rho = dists(i, 0);
      while (iter-- > 0)
      {
        double pSum = 0;
        for (index j = 1; j < dists.cols(); j++)
        {
          double d = dists(i, j) - rho;
          pSum += (d <= 0 ? 1.0 : exp(-(d / mid)));
        }
        if (abs(pSum - target) < tolerance) break;
        if (pSum > target)
        {
          hi = mid;
          mid = (lo + hi) / 2.0;
        }
        else
        {
          lo = mid;
          mid = (hi == infinity ? mid * 2 : (lo + hi) / 2.0);
        }
      }
      result(i) = mid;
    }
    return result;
  }

  template <typename Derived>
  void computeHighDimProb(const Ref<ArrayXXd>& dists, const Ref<ArrayXd>& sigma,
                          Eigen::SparseCompressedBase<Derived>& graph) const
  {
    traverseGraph(graph, [&](auto it) {
      it.valueRef() =
          std::exp(-(it.value() - dists(it.row(), 0)) / sigma(it.row()));
    });
  }

  VectorXd findAB(double minDist)
  {
    using namespace Eigen;
    VectorXd ab(2);
    ab << 1.0, 1.0;
    UMAPEmbeddingParamsFunctor                functor(minDist);
    NumericalDiff<UMAPEmbeddingParamsFunctor> numDiff(functor);
    LevenbergMarquardt<NumericalDiff<UMAPEmbeddingParamsFunctor>> lm(numDiff);
    lm.minimize(ab);
    return ab;
  }

  void makeGraph(const DataSet& in, index k, SparseMatrixXd& graph,
                 Ref<ArrayXXd> dists, bool discardFirst) const
  {
    graph.reserve(in.size() * k);
    auto data = in.getData();
    for (index i = 0; i < in.size(); i++)
    {
      auto [distances, nearestIds] =
          mTree.kNearest(data.row(i), discardFirst ? k + 1 : k);

      for (size_t j = 0; j < asUnsigned(k); j++)
      {
        size_t pos = discardFirst ? j + 1 : j;
        index  neighborIndex = stoi(*nearestIds[pos]);
        dists(i, asSigned(j)) = distances[pos];
        graph.insert(i, neighborIndex) = distances[pos];
      }
    }
  }

  ArrayXXd normalizeEmbedding(const Ref<ArrayXXd>& embedding)
  {
    // based on umap python implementation
    double   expansion = 10.0 / embedding.abs().maxCoeff();
    ArrayXXd noise =
        1e-4 * ArrayXXd::Random(embedding.rows(), embedding.cols()); // uniform
    ArrayXXd result = (embedding * expansion) + noise;
    ArrayXd  min = result.colwise().minCoeff();
    ArrayXd  max = result.colwise().maxCoeff();
    ArrayXd  range = (max - min).max(epsilon);
    result = (result.rowwise() - min.transpose());
    result = result.rowwise() / range.transpose();
    return 10.0 * result;
  }

  template <typename Derived>
  void getGraphIndices(Eigen::SparseCompressedBase<Derived>& graph,
                       Ref<ArrayXi> rowIndices, Ref<ArrayXi> colIndices) const
  {
    index p = 0;
    traverseGraph(graph, [&](auto it) {
      rowIndices(p) = static_cast<int>(it.row());
      colIndices(p) = static_cast<int>(it.col());
      p++;
    });
  }

  template <typename Derived>
  void computeEpochsPerSample(Eigen::SparseCompressedBase<Derived>& graph,
                              Ref<ArrayXd> epochsPerSample) const
  {
    index  p = 0;
    double maxVal = graph.coeffs().maxCoeff();
    traverseGraph(graph, [&](auto it) {
      epochsPerSample(p++) = 1.0 / (it.value() / maxVal);
    });
  }

  void optimizeLayoutAndUpdate(ArrayXXd& embedding, ArrayXXd& reference,
                               ArrayXi const& embIndices,
                               ArrayXi const& refIndices,
                               ArrayXd const& epochsPerSample,
                               double learningRate, index maxIter,
                               double gamma = 1.0)
  {
    impl::optimizeLayout(embedding, reference, embIndices, refIndices,
                         epochsPerSample, mAB, true, learningRate, maxIter,
                         gamma);
  }

  void optimizeLayout(ArrayXXd& embedding, ArrayXXd const& reference,
                      ArrayXi const& embIndices, ArrayXi const& refIndices,
                      ArrayXd const& epochsPerSample, double learningRate,
                      index maxIter, double gamma = 1.0) const
  {
    impl::optimizeLayout(embedding, reference, embIndices, refIndices,
                         epochsPerSample, mAB, false, learningRate, maxIter,
                         gamma);
  }

  template <typename Derived>
  ScopedEigenMap<ArrayXXd>
  initTransformEmbedding(Eigen::SparseCompressedBase<Derived>& graph,
                         Ref<const ArrayXXd> reference, index N,
                         Allocator& alloc = FluidDefaultAllocator()) const
  {
    ScopedEigenMap<ArrayXXd> embedding(N, reference.cols(), alloc);
    embedding.setZero(); // todo: sort out 2D expression constructor?
    traverseGraph(graph, [&](auto it) {
      embedding.row(it.row()) += (reference.row(it.col()) * it.value());
    });
    return embedding;
  }

  template <typename Derived>
  void normalizeRows(Eigen::SparseCompressedBase<Derived>& graph,
                     Allocator& alloc = FluidDefaultAllocator()) const
  {
    ScopedEigenMap<ArrayXd> sums(ArrayXd::Zero(graph.innerSize()), alloc);
    traverseGraph(graph, [&](auto it) { sums(it.row()) += it.value(); });
    traverseGraph(
        graph, [&](auto it) { it.valueRef() = it.value() / sums(it.row()); });
  }

private:
  KDTree   mTree;
  index    mK;
  VectorXd mAB;
  ArrayXXd mEmbedding;
  bool     mInitialized{false};
};
}// namespace algorithm
}// namespace fluid
