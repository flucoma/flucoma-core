#pragma once
#include "KDTree.hpp"
#include "../util/DistanceFuncs.hpp"
#include "../util/FluidEigenMappings.hpp"
#include "../util/SpectralEmbedding.hpp"
#include "../../data/TensorTypes.hpp"
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <cassert>
#include <cmath>
#include <random>
#include <unsupported/Eigen/NonLinearOptimization>
#include <unsupported/Eigen/NumericalDiff>


namespace fluid {
namespace algorithm {

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
    optimizeLayout(mEmbedding, mEmbedding, rowIndices, colIndices,
                   epochsPerSample, true, learningRate, maxIter);
    DataSet out(ids, _impl::asFluid(mEmbedding));
    mInitialized = true;
    return out;
  }

  DataSet transform(DataSet& in, index maxIter = 200, double learningRate = 1.0)
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
                   epochsPerSample, false, learningRate, maxIter);
    DataSet out(in.getIds(), _impl::asFluid(embedding));
    return out;
  }


  void transformPoint(RealVectorView in, RealVectorView out)
  {
    if (!mInitialized) return;
    SparseMatrixXd knnGraph(1, mEmbedding.rows());
    ArrayXXd       dists = ArrayXXd::Zero(1, mK);
    knnGraph.reserve(mK);
    auto nearest = mTree.kNearest(in, mK);
    auto nearestIds = nearest.getIds();
    auto distances = nearest.getData().col(0);
    for (index j = 0; j < mK; j++)
    {
      index neighborIndex = stoi(nearestIds(j));
      dists(0, j) = distances(j);
      knnGraph.insert(0, neighborIndex) = distances(j);
    }
    knnGraph.makeCompressed();
    ArrayXd sigma = findSigma(mK, dists);
    computeHighDimProb(dists, sigma, knnGraph);
    normalizeRows(knnGraph);
    ArrayXXd embedding = initTransformEmbedding(knnGraph, mEmbedding, 1);
    ArrayXd  result = embedding.row(0);
    out <<= _impl::asFluid(result);
  }


private:
  template <typename F>
  void traverseGraph(const SparseMatrixXd& graph, F func)
  {
    for (index i = 0; i < graph.outerSize(); i++)
    {
      for (SparseMatrixXd::InnerIterator it(graph, i); it; ++it) { func(it); }
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

  ArrayXd findSigma(index k, Ref<ArrayXXd> dists, index maxIter = 64,
                    double tolerance = 1e-5)
  {
    using namespace std;
    double  target = log2(k);
    ArrayXd result = ArrayXd::Zero(dists.rows());
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

  void computeHighDimProb(const Ref<ArrayXXd>& dists, const Ref<ArrayXd>& sigma,
                          SparseMatrixXd& graph)
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
                 Ref<ArrayXXd> dists, bool discardFirst)
  {
    graph.reserve(in.size() * k);
    auto data = in.getData();
    for (index i = 0; i < in.size(); i++)
    {
      auto nearest = mTree.kNearest(data.row(i), discardFirst ? k + 1 : k);
      auto nearestIds = nearest.getIds();
      auto distances = nearest.getData().col(0);
      for (index j = 0; j < k; j++)
      {
        index pos = discardFirst ? j + 1 : j;
        index neighborIndex = stoi(nearestIds(pos));
        dists(i, j) = distances(pos);
        graph.insert(i, neighborIndex) = distances(pos);
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

  void getGraphIndices(const SparseMatrixXd& graph, Ref<ArrayXi> rowIndices,
                       Ref<ArrayXi> colIndices)
  {
    index p = 0;
    traverseGraph(graph, [&](auto it) {
      rowIndices(p) = static_cast<int>(it.row());
      colIndices(p) = static_cast<int>(it.col());
      p++;
    });
  }

  void computeEpochsPerSample(const SparseMatrixXd& graph,
                              Ref<ArrayXd>          epochsPerSample)
  {
    index  p = 0;
    double maxVal = graph.coeffs().maxCoeff();
    traverseGraph(graph, [&](auto it) {
      epochsPerSample(p++) = 1.0 / (it.value() / maxVal);
    });
  }

  void optimizeLayout(Ref<ArrayXXd> embedding, Ref<ArrayXXd> reference,
                      Ref<ArrayXi> embIndices, Ref<ArrayXi> refIndices,
                      Ref<ArrayXd> epochsPerSample, bool updateReference,
                      double learningRate, index maxIter, double gamma = 1.0)
  {
    using namespace std;
    double alpha = learningRate;
    double negativeSampleRate = 5.0;
    auto distance = DistanceFuncs::map()[DistanceFuncs::Distance::kSqEuclidean];
    double                          a = mAB(0);
    double                          b = mAB(1);
    random_device                   rd;
    mt19937                         mt(rd());
    uniform_int_distribution<index> randomInt(0, reference.rows() - 1);
    ArrayXd epochsPerNegativeSample = epochsPerSample / negativeSampleRate;
    ArrayXd nextEpoch = epochsPerSample;
    ArrayXd nextNegEpoch = epochsPerNegativeSample;
    ArrayXd bound = VectorXd::Constant(
        embedding.cols(), 4); // based on umap python implementation
    for (index i = 0; i < maxIter; i++)
    {
      for (index j = 0; j < epochsPerSample.size(); j++)
      {
        if (nextEpoch(j) > i) continue;
        ArrayXd current = embedding.row(embIndices(j));
        ArrayXd other = reference.row(refIndices(j));
        double dist = distance(current, other); // todo: try to have dist member
        double gradCoef = 0;
        ArrayXd grad;
        if (dist > 0)
        {
          gradCoef = -2.0 * a * b * pow(dist, b - 1.0);
          gradCoef /= a * pow(dist, b) + 1.0;
        }
        grad = (gradCoef * (current - other)).cwiseMin(bound).cwiseMax(-bound);
        current += grad * alpha;
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
        if (updateReference) reference.row(refIndices(j)) = other;
      }
      alpha = learningRate * (1.0 - (i / double(maxIter)));
    }
  }

  ArrayXXd initTransformEmbedding(const SparseMatrixXd& graph,
                                  Ref<ArrayXXd> reference, index N)
  {
    ArrayXXd embedding = ArrayXXd::Zero(N, reference.cols());
    traverseGraph(graph, [&](auto it) {
      embedding.row(it.row()) += (reference.row(it.col()) * it.value());
    });
    return embedding;
  }

  void normalizeRows(const SparseMatrixXd& graph)
  {
    ArrayXd sums = ArrayXd::Zero(graph.innerSize());
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
