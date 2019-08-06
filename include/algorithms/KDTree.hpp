#pragma once

#include "data/FluidDataset.hpp"
#include "data/FluidTensor.hpp"
#include "data/TensorTypes.hpp"
#include "algorithms/util/FluidEigenMappings.hpp"
#include <string>
#include <Eigen/Core>

namespace fluid {
namespace algorithm {

class KDTree {

public:
  using string = std::string;
  struct Node;
  using NodePtr = std::shared_ptr<Node>;

  struct Node
  {
    const string label;
    const RealVectorView data;
    NodePtr left{nullptr}, right{nullptr};
  };

  KDTree(int nDims) : mDims(nDims) {}

  // TODO: find dimension with greatest variance, and median of that to start
  // the tree
  KDTree(FluidDataset<double, std::string, 2> dataset)
  {
    mDims = dataset.pointSize();
  }

  void addNode(const string label,
               const RealVectorView data)
  {
      mRoot = addNode(mRoot, label, data, 0);
  }

  string nearest(const RealVectorView data)
  {
    NodePtr n = nearest(mRoot, data, nullptr,
                        std::numeric_limits<double>::infinity(), 0);
    return n->label;
  }

private:


  NodePtr makeNode(const string label, const RealVectorView data)
  {
    return std::make_shared<Node>(Node{label, data, nullptr, nullptr});
  }

  NodePtr addNode(NodePtr current, const string label,
                  const RealVectorView data, const int depth)
  {
    if (current == nullptr){
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

  NodePtr nearest(NodePtr current, const RealVectorView data, NodePtr best,
                  double bestDist, const int depth)
  {
    if (current == nullptr)
      return best;
    const double currentDist = distance(current->data, data);
    if (currentDist < bestDist) {
      best = current;
      bestDist = currentDist;
    }
    const int d = depth % mDims;
    const double dimDif = current->data(d) - data(d);
    NodePtr firstBranch;
    NodePtr secondBranch;
    if (dimDif > 0) {
      firstBranch = current->left;
      secondBranch = current->right;
    } else {
      firstBranch = current->right;
      secondBranch = current->left;
    }
    NodePtr newBest = nearest(firstBranch, data, best, bestDist, depth + 1);
    if (newBest != best) {
      best = newBest;
      bestDist = distance(best->data, data); // TODO: computing twice
    }
    if (dimDif < bestDist) // ball centered at query with diametre bestDist
                           // intersects with current partition
    {
      best = nearest(secondBranch, data, best, bestDist, depth + 1);
    }
    return best;
  }

  NodePtr mRoot{nullptr};
  int mDims;
};
} // namespace algorithm
} // namespace fluid
