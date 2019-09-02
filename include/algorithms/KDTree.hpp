#pragma once

#include "algorithms/util/FluidEigenMappings.hpp"
#include "data/FluidDataset.hpp"
#include "data/FluidTensor.hpp"
#include "data/TensorTypes.hpp"
#include <Eigen/Core>
#include <queue>
#include <string>

namespace fluid {
namespace algorithm {

template <typename T>
class KDTree {

public:
  using string = std::string;
  struct Node;
  using NodePtr = std::shared_ptr<Node>;
  using Dataset = FluidDataset<string, double, T, 1>;

  struct Node {
    const T target;
    const RealVectorView data;
    NodePtr left{nullptr}, right{nullptr};
  };
  using knnCandidate = std::pair<double, NodePtr>;
  using knnQueue = std::priority_queue<knnCandidate, std::vector<knnCandidate>,
                                       std::less<knnCandidate>>;
  using iterator = const std::vector<int>::iterator;

  KDTree(int nDims) : mDims(nDims) {}

  KDTree(const Dataset &dataset) {
    using namespace std;
    mNPoints = dataset.size();
    mDims = dataset.pointSize();
    vector<int> indices(dataset.size());
    iota(indices.begin(), indices.end(), 0);
    mRoot = buildTree(indices, indices.begin(), indices.end(), dataset, 0);
  }

  void addNode(const T target, const RealVectorView data) {
    mRoot = addNode(mRoot, target, data, 0);
    mNPoints++;
  }

  //FluidTensor<T, 1> kNearest(const RealVectorView data, int k = 1) {
  FluidDataset<int, double, T, 1>  kNearest(const RealVectorView data, int k = 1) {
    assert(data.size() == mDims);
    knnQueue queue;
    //auto result = FluidTensor<string, 1>(k);
    auto result = FluidDataset<int, double, T, 1>(1);
    kNearest(mRoot, data, queue, k, 0);
    for (int i = k; i > 0; i--) {
      auto kNearest =   queue.top();
      auto dist = FluidTensor<double, 1>{kNearest.first};
      auto val = kNearest.second->target;
      result.add(i, dist, val);
      //result(i - 1) = queue.top().second->target;
      queue.pop();
    }
    return result;
  }

  void print() { print(mRoot, 0); }
  int nPoints(){return mNPoints;}


private:
  NodePtr buildTree(std::vector<int> indices, iterator from, iterator to,
                    const Dataset &dataset,
                    const int depth) {
    using namespace std;
    if (from == to)
      return nullptr;
    else if (std::distance(from, to) == 1) {
      return makeNode(dataset.getTargets()(*from), dataset.getData().row(*from));
    }
    const int d = depth % mDims;
    sort(from, to, [&](int a, int b) {
      return dataset.getData().row(a)(d) < dataset.getData().row(b)(d);
    });
    const int range = std::distance(from, to);
    const int median = range / 2;
    NodePtr current = makeNode(dataset.getTargets().row(*(from + median)),
                               dataset.getData().row(*(from + median)));
    if (median > 0)
      current->left =
          buildTree(indices, from, from + median, dataset, depth + 1);
    if (range - median > 1)
      current->right =
          buildTree(indices, from + median + 1, to, dataset, depth + 1);
    return current;
  }

  NodePtr makeNode(const T target, const RealVectorView data) {
    return std::make_shared<Node>(Node{target, data, nullptr, nullptr});
  }

  NodePtr addNode(NodePtr current, const T target,
                  const RealVectorView data, const int depth) {
    if (current == nullptr) {
      return makeNode(target, data);
    }

    const int d = depth % mDims;
    if (data(d) < current->data(d)) {
      current->left = addNode(current->left, target, data, depth + 1);
    } else {
      current->right = addNode(current->right, target, data, depth + 1);
    }
    return current;
  }

  double distance(const RealVectorView p1, const RealVectorView p2) {
    using namespace Eigen;
    ArrayXd v1 = _impl::asEigen<Array>(p1);
    ArrayXd v2 = _impl::asEigen<Array>(p2);
    return (v1 - v2).matrix().norm();
  }

  void print(NodePtr current, int depth) {
    for (int i = 0; i < depth; ++i)
      std::cout << "  ";
    if (current == nullptr) {
      std::cout << " null" << std::endl;
      return;
    }
    std::cout << " " << current->target << std::endl;
    for (int i = 0; i < depth; ++i)
      std::cout << "  ";
    std::cout << " left" << std::endl;
    print(current->left, depth + 1);
    for (int i = 0; i < depth; ++i)
      std::cout << "  ";
    std::cout << " right" << std::endl;
    print(current->right, depth + 1);
  }
  void kNearest(NodePtr current, const RealVectorView data, knnQueue &knn,
                const int k, const int depth) {
    if (current == nullptr)
      return;
    const double currentDist = distance(current->data, data);
    if (knn.size() < k)
      knn.push(make_pair(currentDist, current));
    else if (currentDist < knn.top().first) {
      knn.pop();
      knn.push(make_pair(currentDist, current));
    }
    const int d = depth % mDims;
    const double dimDif = current->data(d) - data(d);
    NodePtr firstBranch = current->left;
    NodePtr secondBranch = current->right;
    if (dimDif <= 0) {
      firstBranch = current->right;
      secondBranch = current->left;
    }
    kNearest(firstBranch, data, knn, k, depth + 1);
    if (dimDif < knn.top().first) // ball centered at query with diametre
                                  // kthDist intersects with current partition
    {
      kNearest(secondBranch, data, knn, k, depth + 1);
    }
  }

  NodePtr mRoot{nullptr};
  int mDims;
  int mNPoints;
};
} // namespace algorithm
} // namespace fluid
