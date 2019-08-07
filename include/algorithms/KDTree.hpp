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

class KDTree {

public:
  using string = std::string;
  struct Node;
  using NodePtr = std::shared_ptr<Node>;

  struct Node {
    const string label;
    const RealVectorView data;
    NodePtr left{nullptr}, right{nullptr};
  };
  using knnCandidate = std::pair<double, NodePtr>;
  using knnQueue = std::priority_queue<knnCandidate, std::vector<knnCandidate>,
                                       std::less<knnCandidate>>;

  KDTree(int nDims) : mDims(nDims) {}

  // TODO: find dimension with greatest variance, and median of that to start
  // the tree
  KDTree(FluidDataset<double, std::string, 2> dataset)
  {
    mDims = dataset.pointSize();
  }

  void addNode(const string label, const RealVectorView data)
  {
    mRoot = addNode(mRoot, label, data, 0);
  }

  FluidTensor<string, 1> kNearest(const RealVectorView data, int k = 1)
  {
    knnQueue queue;
    auto result = FluidTensor<string, 1>(k);
    kNearest(mRoot, data, queue, k, 0);
    for (int i = k; i > 0; i--) {
      result(i - 1) = queue.top().second->label;
      queue.pop();
    }
    return result;
  }

private:
  NodePtr makeNode(const string label, const RealVectorView data)
  {
    return std::make_shared<Node>(Node{label, data, nullptr, nullptr});
  }

  NodePtr addNode(NodePtr current, const string label,
                  const RealVectorView data, const int depth)
  {
    if (current == nullptr) {
      return makeNode(label, data);
    }

    const int d = depth % mDims;
    if (data(d) < current->data(d)) {
      current->left = addNode(current->left, label, data, depth + 1);
    } else {
      current->right = addNode(current->right, label, data, depth + 1);
    }
    return current;
  }

  double distance(const RealVectorView p1, const RealVectorView p2)
  {
    using namespace Eigen;
    ArrayXd v1 = _impl::asEigen<Array>(p1);
    ArrayXd v2 = _impl::asEigen<Array>(p2);
    return (v1 - v2).matrix().norm();
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
};
} // namespace algorithm
} // namespace fluid
