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
#include "../../data/FluidDataSet.hpp"
#include "../../data/FluidIndex.hpp"
#include "../../data/FluidTensor.hpp"
#include "../../data/TensorTypes.hpp"
#include "../../data/FluidMemory.hpp"
#include <Eigen/Core>
#include <queue>
#include <memory>
#include <string>

namespace fluid {
namespace algorithm {

class KDTree
{

public:
  using string = std::string;

  using DataSet = FluidDataSet<string, double, 1>;
  using ConstRealVectorView = FluidTensorView<const double, 1>;
  struct Node;
  using NodePtr = std::shared_ptr<Node>;
  using knnCandidate = std::pair<double, const Node*>;
  using knnQueue = rt::vector<knnCandidate>;
  using KNNResult =
      std::pair<rt::vector<double>, rt::vector<const std::string*>>;
  using iterator = const std::vector<index>::iterator;

  struct Node
  {
    const string     id;
    const RealVector data;
    NodePtr          left{nullptr}, right{nullptr};
  };

  struct FlatData
  {
    FluidTensor<index, 2>  tree;
    FluidTensor<string, 1> ids;
    FluidTensor<double, 2> data;
    FlatData(index n, index m) : tree(n, 2), ids(n), data(n, m) {}
  };

  explicit KDTree() = default;
  ~KDTree() = default;

  KDTree(const DataSet& dataset)
  {
    using namespace std;
    mNPoints = dataset.size();
    mDims = dataset.pointSize();
    if (mDims > 0 && mNPoints > 0)
    {
      vector<index> indices(asUnsigned(dataset.size()));
      iota(indices.begin(), indices.end(), 0);
      mRoot = buildTree(indices, indices.begin(), indices.end(), dataset, 0);
    }
    mInitialized = true;
  }

  void addNode(string id, ConstRealVectorView data)
  {
    mRoot = addNode(mRoot.get(), id, data, 0);
    mNPoints++;
  }

  KNNResult kNearest(ConstRealVectorView data, index k = 1, double radius = 0,
                     Allocator& alloc = FluidDefaultAllocator()) const
  {
    assert(data.size() == mDims);
    rt::vector<knnCandidate> queue(alloc);
    if (k > 0) queue.reserve(asUnsigned(k));

    kNearest(mRoot.get(), data, queue, k, radius, 0);
    std::sort_heap(queue.begin(), queue.end());

    KNNResult result =
        std::make_pair(rt::vector<double>(queue.size(), alloc),
                       rt::vector<const std::string*>(queue.size(), alloc));

    std::for_each(queue.begin(), queue.end(),
                  [&result, i = 0u](knnCandidate const& x) mutable {
                    result.first[i] = x.first;
                    result.second[i++] = &(x.second->id);
                  });
    return result;
  }

  void  print() const { print(mRoot.get(), 0); }
  index dims() const { return mDims; }
  index size() const { return mNPoints; }
  bool  initialized() const { return mInitialized; }

  void clear()
  {
    mRoot = nullptr;
    mNPoints = 0; 
    mDims = 0; 
    mInitialized = false;
  }

  FlatData toFlat() const
  {
    FlatData store(mNPoints, mDims);
    flatten(0, mRoot.get(), store);
    return store;
  }

  void fromFlat(FlatData vectors)
  {
    mRoot = unflatten(vectors, 0);
    mNPoints = vectors.data.rows();
    mDims = vectors.data.cols();
    mInitialized = true;
  }

private:
  NodePtr buildTree(std::vector<index>& indices, iterator from, iterator to,
                    const DataSet& dataset, index depth) const
  {
    using namespace std;
    if (from == to)
      return nullptr;
    else if (std::distance(from, to) == 1)
    {
      return makeNode(dataset.getIds()(*from), dataset.getData().row(*from));
    }
    const index d = depth % mDims;
    sort(from, to, [&](index a, index b) {
      return dataset.getData().row(a)(d) < dataset.getData().row(b)(d);
    });
    const index range = std::distance(from, to);
    const index median = range / 2;
    NodePtr     current = makeNode(dataset.getIds().row(*(from + median)),
                               dataset.getData().row(*(from + median)));
    if (median > 0)
      current->left =
          buildTree(indices, from, from + median, dataset, depth + 1);
    if (range - median > 1)
      current->right =
          buildTree(indices, from + median + 1, to, dataset, depth + 1);
    return current;
  }

  NodePtr makeNode(string id, ConstRealVectorView data) const
  {
    return std::make_shared<Node>(Node{id, RealVector{data}, nullptr, nullptr});
  }

  NodePtr addNode(Node* current, string id, ConstRealVectorView data,
                  const index depth) const
  {
    if (current == nullptr) { return makeNode(id, data); }

    const index d = depth % mDims;
    if (data(d) < current->data(d))
    { current->left = addNode(current->left.get(), id, data, depth + 1); }
    else
    {
      current->right = addNode(current->right.get(), id, data, depth + 1);
    }
    return NodePtr(current);
  }

  double distance(ConstRealVectorView p1, ConstRealVectorView p2) const
  {
    using namespace Eigen;
    auto v1 = _impl::asEigen<Array>(p1);
    auto v2 = _impl::asEigen<Array>(p2);
    return (v1 - v2).matrix().norm();
  }

  void print(Node* current, index depth) const
  {
    for (index i = 0; i < depth; ++i) std::cout << "  ";
    if (current == nullptr)
    {
      std::cout << " null" << std::endl;
      return;
    }
    std::cout << " " << current->id << std::endl;
    for (index i = 0; i < depth; ++i) std::cout << "  ";
    std::cout << " left" << std::endl;
    print(current->left.get(), depth + 1);
    for (index i = 0; i < depth; ++i) std::cout << "  ";
    std::cout << " right" << std::endl;
    print(current->right.get(), depth + 1);
  }

  void kNearest(const Node* current, ConstRealVectorView data, knnQueue& knn,
                index k, double radius, index depth) const
  {
    if (current == nullptr) return;
    const double currentDist = distance(current->data, data);
    bool         withinRadius = radius > 0 ? currentDist < radius : true;
    if (withinRadius && (knn.size() < asUnsigned(k) || k == 0))
    {
      knn.push_back(std::make_pair(currentDist, current));
      std::push_heap(knn.begin(), knn.end());
    }
    else if (withinRadius && currentDist < knn.front().first)
    {
      std::pop_heap(knn.begin(), knn.end());
      knn.back() = std::make_pair(currentDist, current);
      std::push_heap(knn.begin(), knn.end());
    }
    const index  d = depth % mDims;
    const double dimDif = current->data(d) - data(d);
    Node*        firstBranch = current->left.get();
    Node*        secondBranch = current->right.get();
    if (dimDif <= 0)
    {
      firstBranch = current->right.get();
      secondBranch = current->left.get();
    }
    kNearest(firstBranch, data, knn, k, radius, depth + 1);
    if (k == 0 || knn.size() < asUnsigned(k) ||
        dimDif < knn.front().first) // ball centered at query with diametre
                                    // kthDist intersects with current partition
                                    // (or need to get more neighbors)
    {
      kNearest(secondBranch, data, knn, k, radius, depth + 1);
    }
  }

  index flatten(index nodeId, const Node* current, FlatData& store) const
  {
    if (current == nullptr) { return nodeId; }
    store.ids(nodeId) = current->id;
    store.data.row(nodeId) <<= current->data;

    index nextNodeId = nodeId + 1;
    if (current->left == nullptr) { store.tree(nodeId, 0) = -1; }
    else
    {
      store.tree(nodeId, 0) = nextNodeId;
      nextNodeId = flatten(nextNodeId, current->left.get(), store);
    }
    if (current->right == nullptr) { store.tree(nodeId, 1) = -1; }
    else
    {
      store.tree(nodeId, 1) = nextNodeId;
      nextNodeId = flatten(nextNodeId, current->right.get(), store);
    }
    return nextNodeId;
  }

  NodePtr unflatten(const FlatData& store, index index) const
  {
    if (index == -1) return nullptr;
    NodePtr current = makeNode(store.ids[index], store.data[index]);
    current->left = unflatten(store, store.tree(index, 0));
    current->right = unflatten(store, store.tree(index, 1));
    return current;
  }

  NodePtr mRoot{nullptr};
  index   mDims{0};
  index   mNPoints{0};
  bool    mInitialized{false};
};
} // namespace algorithm
} // namespace fluid
