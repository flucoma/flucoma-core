#pragma once

#include "algorithms/util/FluidEigenMappings.hpp"
#include "data/FluidDataSet.hpp"
#include "data/FluidTensor.hpp"
#include "data/TensorTypes.hpp"
#include "data/FluidIndex.hpp"
#include <Eigen/Core>
#include <queue>
#include <string>

namespace fluid {
namespace algorithm {

class KDTree {

public:
  using string = std::string;
  struct Node;
  using NodePtr = Node*;
  using DataSet = FluidDataSet<string, double, 1>;
  using ConstRealVectorView = FluidTensorView<const double, 1>; 

  struct Node {
    const string id;
    const RealVector data;
    std::shared_ptr<Node> left{nullptr}, right{nullptr};
  };

  struct FlatData {
    FluidTensor<index, 2> tree;
    FluidTensor<string, 1> ids;
    FluidTensor<double, 2> data;
    FlatData(index n, index m) : tree(n, 2), ids(n), data(n, m) {}
  };

  using knnCandidate = std::pair<double, NodePtr>;
  using knnQueue = std::priority_queue<knnCandidate, std::vector<knnCandidate>,
                                       std::less<knnCandidate>>;
  using iterator = const std::vector<index>::iterator;

  explicit KDTree() = default;
  ~KDTree() = default;

  KDTree(const DataSet &dataset) {
    using namespace std;
    mNPoints = dataset.size();
    mDims = dataset.pointSize();
    if(mDims > 0 && mNPoints > 0){
      vector<index> indices(asUnsigned(dataset.size()));
      iota(indices.begin(), indices.end(), 0);
      mRoot.reset(buildTree(indices, indices.begin(), indices.end(), dataset, 0));
    }
    mInitialized = true;
  }

  void addNode(string id, ConstRealVectorView data) {
    mRoot.reset(addNode(mRoot.get(), id, data, 0));
    mNPoints++;
  }

  DataSet kNearest(ConstRealVectorView data, index k = 1, double radius = 0) const {
    assert(data.size() == mDims);
    knnQueue queue;
    auto result = DataSet(1);
    kNearest(mRoot.get(), data, queue, k, radius, 0);
    index numFound = queue.size();
    std::vector<knnCandidate> sorted(numFound);
    for (index i = numFound - 1; i >= 0; i--) {
      sorted[asUnsigned(i)] = queue.top();
      queue.pop();
    }
    for (index i = 0; i < numFound; i++) {
      auto dist = FluidTensor<double, 1>{sorted[asUnsigned(i)].first};
      auto id = sorted[asUnsigned(i)].second->id;
      result.add(id, dist);
    }
    return result;
  }

  void print() const { print(mRoot.get(), 0); }
  index dims() const { return mDims; }
  index size() const { return mNPoints; }
  bool initialized() const { return mInitialized; }

  void clear(){
    mRoot = nullptr;
    mInitialized = false;
  }

  FlatData toFlat() const{
    FlatData store(mNPoints, mDims);
    flatten(0, mRoot.get(), store);
    return store;
  }

  void fromFlat(FlatData vectors) {
     mRoot.reset(unflatten(vectors, 0));
     mNPoints = vectors.data.rows();
     mDims =  vectors.data.cols();
     mInitialized = true;
   }

private:
  NodePtr buildTree(std::vector<index>& indices, iterator from, iterator to,
                    const DataSet &dataset, index depth) {
    using namespace std;
    if (from == to) return nullptr;
    else if (std::distance(from, to) == 1) {
      return makeNode(dataset.getIds()(*from),
                      dataset.getData().row(*from));
    }
    const index d = depth % mDims;
    sort(from, to, [&](index a, index b) {
      return dataset.getData().row(a)(d) < dataset.getData().row(b)(d);
    });
    const index range = std::distance(from, to);
    const index median = range / 2;
    NodePtr current = makeNode(dataset.getIds().row(*(from + median)),
                               dataset.getData().row(*(from + median)));
    if (median > 0)
      current->left.reset(buildTree(indices, from, from + median, dataset, depth + 1));
    if (range - median > 1)
      current->right.reset(buildTree(indices, from + median + 1, to, dataset, depth + 1));
    return current;
  }

  NodePtr makeNode(string id, ConstRealVectorView data) const{
    return new Node{id, RealVector{data}, nullptr, nullptr};
  }

  NodePtr addNode(NodePtr current, string id, ConstRealVectorView data,
                  const index depth) {
    if (current == nullptr) {
      return makeNode(id, data);
    }

    const index d = depth % mDims;
    if (data(d) < current -> data(d)) {
      current->left.reset(addNode(current->left.get(), id, data, depth + 1));
    } else {
      current->right.reset(addNode(current->right.get(), id, data, depth + 1));
    }
    return current;
  }

  double distance(ConstRealVectorView p1, ConstRealVectorView p2) const{
    using namespace Eigen;
    auto v1 = _impl::asEigen<Array>(p1);
    auto v2 = _impl::asEigen<Array>(p2);
    return (v1 - v2).matrix().norm();
  }

  void print(NodePtr current, index depth) const {
    for (index i = 0; i < depth; ++i)
      std::cout << "  ";
    if (current == nullptr) {
      std::cout << " null" << std::endl;
      return;
    }
    std::cout << " " << current->id << std::endl;
    for (index i = 0; i < depth; ++i)
      std::cout << "  ";
    std::cout << " left" << std::endl;
    print(current->left.get(), depth + 1);
    for (index i = 0; i < depth; ++i)
      std::cout << "  ";
    std::cout << " right" << std::endl;
    print(current->right.get(), depth + 1);
  }

  void kNearest(NodePtr current, ConstRealVectorView data, knnQueue &knn,
                 index k, double radius,  index depth) const{
    if (current == nullptr)
      return;
    const double currentDist = distance(current->data, data);
    bool withinRadius = radius > 0? currentDist < radius:true;
    if (withinRadius && (knn.size() < asUnsigned(k)|| k == 0)){
      knn.push(std::make_pair(currentDist, current));
    }
    else if (withinRadius && currentDist < knn.top().first) {
      knn.pop();
      knn.push(std::make_pair(currentDist, current));
    }
    const index d = depth % mDims;
    const double dimDif = current->data(d) - data(d);
    NodePtr firstBranch = current->left.get();
    NodePtr secondBranch = current->right.get();
    if (dimDif <= 0) {
      firstBranch = current->right.get();
      secondBranch = current->left.get();
    }
    kNearest(firstBranch, data, knn, k, radius, depth + 1);
    if (k == 0 || knn.size() < asUnsigned(k) || dimDif < knn.top().first) // ball centered at query with diametre
                                  // kthDist intersects with current partition
                                  // (or need to get more neighbors)
    {
      kNearest(secondBranch, data, knn, k, radius, depth + 1);
    }
  }

  index flatten(index nodeId, NodePtr current, FlatData &store) const{
    if (current == nullptr) {
      return nodeId;
    }
    store.ids(nodeId) = current->id;
    store.data.row(nodeId) = current->data;

    index nextNodeId = nodeId + 1;
    if(current->left == nullptr){
        store.tree(nodeId, 0) = -1;
    }
    else{
      store.tree(nodeId, 0) = nextNodeId;
      nextNodeId = flatten(nextNodeId, current->left.get(), store);
    }
    if(current->right == nullptr){
        store.tree(nodeId, 1) = -1;
    }
    else {
      store.tree(nodeId, 1) = nextNodeId;
      nextNodeId = flatten(nextNodeId, current->right.get(), store);
    }
    return nextNodeId;
  }

  NodePtr unflatten(FlatData &store, index index) const{
    if(index == -1) return nullptr;
    NodePtr current = makeNode(store.ids[index], store.data[index]);
    current->left.reset(unflatten(store, store.tree(index, 0)));
    current->right.reset(unflatten(store, store.tree(index, 1)));
    return current;
  }

  std::shared_ptr<Node> mRoot{nullptr};
  index mDims;
  index mNPoints{0};
  bool mInitialized{false};
};
} // namespace algorithm
} // namespace fluid
