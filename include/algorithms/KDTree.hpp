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

template <typename T> class KDTree {

public:
  using string = std::string;
  struct Node;
  using NodePtr = std::shared_ptr<Node>;
  using Dataset = FluidDataset<string, double, T, 1>;

  struct Node {
    const string id;
    const T target;
    const RealVector data;
    NodePtr left{nullptr}, right{nullptr};
  };

  struct FlatData {
    FluidTensor<int, 2> tree;
    FluidTensor<T, 1> ids;
    FluidTensor<T, 1> targets;
    FluidTensor<double, 2> data;
    FlatData(int n, int m) : tree(n, 2), targets(n), data(n, m) {}
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

  void addNode(const string id, const T target, const RealVectorView data) {
    mRoot = addNode(mRoot, target, data, 0);
    mNPoints++;
  }

  FluidDataset<string, double, T, 1> kNearest(const RealVectorView data,
                                           int k = 1) {
    assert(data.size() == mDims);
    knnQueue queue;
    auto result = FluidDataset<string, double, T, 1>(1);
    std::vector<knnCandidate> sorted(k);
    kNearest(mRoot, data, queue, k, 0);
    for (int i = k - 1; i >= 0; i--) {
      sorted[i] = queue.top();
      queue.pop();
    }
    for (int i = 0; i < k; i++) {
      auto dist = FluidTensor<double, 1>{sorted[i].first};
      auto val = sorted[i].second->target;
      auto id = sorted[i].second->id;
      result.add(id, dist, val);
    }
    return result;
  }

  void print() const { print(mRoot, 0); }
  int nPoints() const { return mNPoints; }
  int nDims() const { return mDims; }

  FlatData toFlat() {
    FlatData store(mNPoints, mDims);
    flatten(0, mRoot, store);
    return store;
  }

  void fromFlat(FlatData vectors) {
     mRoot = unflatten(vectors, 0);
     mNPoints = vectors.data.rows();
     mDims =  vectors.data.cols();
   }

private:
  NodePtr buildTree(std::vector<int> indices, iterator from, iterator to,
                    const Dataset &dataset, const int depth) {
    using namespace std;
    if (from == to) return nullptr;
    else if (std::distance(from, to) == 1) {
      return makeNode(dataset.getIds()(*from),
                      dataset.getTargets()(*from),
                      dataset.getData().row(*from));
    }
    const int d = depth % mDims;
    sort(from, to, [&](int a, int b) {
      return dataset.getData().row(a)(d) < dataset.getData().row(b)(d);
    });
    const int range = std::distance(from, to);
    const int median = range / 2;
    NodePtr current = makeNode(dataset.getIds().row(*(from + median)),
                               dataset.getTargets().row(*(from + median)),
                               dataset.getData().row(*(from + median)));
    if (median > 0)
      current->left =
          buildTree(indices, from, from + median, dataset, depth + 1);
    if (range - median > 1)
      current->right =
          buildTree(indices, from + median + 1, to, dataset, depth + 1);
    return current;
  }

  NodePtr makeNode(const string id, const T target, const RealVectorView data) {
    const RealVector point{data};
    return std::make_shared<Node>(Node{id, target, point, nullptr, nullptr});
  }

  NodePtr addNode(NodePtr current, const string id, const T target, const RealVectorView data,
                  const int depth) {
    if (current == nullptr) {
      return makeNode(id, target, data);
    }

    const int d = depth % mDims;
    if (data(d) < current->data(d)) {
      current->left = addNode(current->left, target, data, depth + 1);
    } else {
      current->right = addNode(current->right, target, data, depth + 1);
    }
    return current;
  }

  double distance(RealVector p1, RealVector p2) {
    using namespace Eigen;
    ArrayXd v1 = _impl::asEigen<Array>(p1);
    ArrayXd v2 = _impl::asEigen<Array>(p2);
    return (v1 - v2).matrix().norm();
  }

  void print(NodePtr current, int depth) const {
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
    const RealVector point{data};
    const double currentDist = distance(current->data, point);
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

  int flatten(int nodeId, NodePtr current, FlatData &store) {
    if (current == nullptr) {
      return nodeId;
    }
    store.targets(nodeId) = current->target;
    store.ids(nodeId) = current->id;
    store.data.row(nodeId) = current->data;

    int nextNodeId = nodeId + 1;
    if(current->left == nullptr){
        store.tree(nodeId, 0) = -1;
    }
    else{
      store.tree(nodeId, 0) = nextNodeId;
      nextNodeId = flatten(nextNodeId, current->left, store);
    }
    if(current->right == nullptr){
        store.tree(nodeId, 1) = -1;
    }
    else {
      store.tree(nodeId, 1) = nextNodeId;
      nextNodeId = flatten(nextNodeId, current->right, store);
    }
    return nextNodeId;
  }

  NodePtr unflatten(FlatData &store, int index) {
    if(index == -1) return nullptr;
    NodePtr current = makeNode(store.ids[index], store.targets[index], store.data[index]);
    current->left = unflatten(store, store.tree(index, 0));
    current->right = unflatten(store, store.tree(index, 1));
    return current;
  }

  NodePtr mRoot{nullptr};
  int mDims;
  int mNPoints{0};
};
} // namespace algorithm
} // namespace fluid
