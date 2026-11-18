#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>
#include <flucoma/clients/common/FluidContext.hpp>
#include <flucoma/clients/nrt/DataSetClient.hpp>
#include <flucoma/clients/nrt/FluidSharedInstanceAdaptor.hpp>
#include <flucoma/clients/nrt/KDTreeClient.hpp>
#include <flucoma/data/FluidMemory.hpp>
#include <string>
#include <vector>

using fluid::FluidDefaultAllocator;
using fluid::client::FluidContext;
using fluid::client::kdtree::KDTreeClient;
using fluid::client::NRTSharedInstanceAdaptor;
using fluid::client::dataset::DataSetClient;

using fluid::client::MessageResult;
using fluid::client::NRTThreadedDataSetClient;
using fluid::client::NRTThreadedKDTreeClient;

using DataSet = NRTSharedInstanceAdaptor<DataSetClient>;
using Params = DataSet::ParamSetType;

// What hosts hold: the threading adaptor around the sharing adaptor
using Threaded = NRTThreadedDataSetClient;
constexpr size_t kRefer = DataSet::NumClientMessages;
constexpr size_t kSetName = kRefer + 1;
constexpr size_t kGetName = kRefer + 2;

namespace {
Params makeParams()
{
  return Params(DataSet::getParameterDescriptors(), FluidDefaultAllocator());
}

std::string printed(DataSet::ClientPointer const& c)
{
  return static_cast<std::string>(c->print());
}

fluid::index sizeOf(DataSet::ClientPointer const& c)
{
  return static_cast<fluid::index>(c->size());
}

const char* const oneEntry = R"({"cols":1,"data":{"a":[1.0]}})";

template <size_t N, typename M>
struct CollectName
{
  void operator()(M const& m, std::vector<std::string>& names)
  {
    names.emplace_back(m.name);
  }
};

template <typename Set>
std::vector<std::string> messageNames(Set const& set)
{
  std::vector<std::string> names;
  set.template iterate<CollectName>(names);
  return names;
}

// Send "size" to a host-side object through the normal message machinery,
// to see which client it really talks to
template <size_t N, typename M>
struct SendSize
{
  void operator()(M const& m, Threaded& c, fluid::index& out)
  {
    if constexpr (std::is_same_v<typename M::ReturnType,
                                 MessageResult<fluid::index>> &&
                  std::tuple_size_v<typename M::ArgumentTypes> == 0)
      if (std::string(m.name) == "size") out = c.template invoke<N>(c);
  }
};

fluid::index sizeVia(Threaded& c)
{
  fluid::index out = -1;
  Threaded::getMessageDescriptors().template iterate<SendSize>(c, out);
  return out;
}
} // namespace

TEST_CASE("Shared instances are found by the name given to setName",
          "[SharedInstanceAdaptor]")
{
  CHECK_FALSE(DataSet::lookup("setname_a"));

  Params p1 = makeParams();
  CHECK(p1.setName("setname_a").ok());
  CHECK(p1.name() == "setname_a");
  DataSet d1(p1, FluidContext());

  auto found = DataSet::lookup("setname_a");
  REQUIRE(found);
  CHECK(found->name() == "setname_a");

  SECTION("a second object with the same name shares the same client")
  {
    Params p2 = makeParams();
    CHECK(p2.setName("setname_a").ok());
    DataSet d2(p2, FluidContext());
    REQUIRE(found->load(oneEntry).ok());
    CHECK(sizeOf(DataSet::lookup("setname_a")) == 1);
    CHECK(DataSet::lookup("setname_a").get() == found.get());
  }

  SECTION("an empty name warns and is not registered")
  {
    Params p2 = makeParams();
    CHECK_FALSE(p2.setName("").ok());
    CHECK(p2.name() == "");
    CHECK_FALSE(p2.lookup("").ok());
  }
}

TEST_CASE("Entries are removed when the last holder goes away",
          "[SharedInstanceAdaptor]")
{
  {
    Params p = makeParams();
    p.setName("lifetime_a");
    DataSet d(p, FluidContext());
    CHECK(DataSet::lookup("lifetime_a"));
    CHECK(p.lookup("lifetime_a").ok());
    CHECK(p.rename("lifetime_b").ok());
    CHECK(DataSet::lookup("lifetime_b"));
  }
  CHECK_FALSE(DataSet::lookup("lifetime_a"));
  CHECK_FALSE(DataSet::lookup("lifetime_b"));
  Params probe = makeParams();
  CHECK_FALSE(probe.lookup("lifetime_a").ok());
  CHECK_FALSE(probe.lookup("lifetime_b").ok());
}

TEST_CASE("rename re-keys the entry without touching the shared data",
          "[SharedInstanceAdaptor]")
{
  Params p1 = makeParams();
  p1.setName("rename_old");
  DataSet d1(p1, FluidContext());

  Params p2 = makeParams();
  p2.setName("rename_old");
  DataSet d2(p2, FluidContext());

  auto before = DataSet::lookup("rename_old");
  REQUIRE(before);
  REQUIRE(before->load(oneEntry).ok());
  CHECK(printed(before).rfind("DataSet rename_old:", 0) == 0);

  SECTION("to a free name")
  {
    CHECK(p1.rename("rename_new").ok());

    CHECK_FALSE(DataSet::lookup("rename_old"));
    CHECK_FALSE(p1.lookup("rename_old").ok());
    CHECK(p1.lookup("rename_new").ok());

    auto after = DataSet::lookup("rename_new");
    REQUIRE(after);
    CHECK(after.get() == before.get());
    CHECK(sizeOf(after) == 1);
    CHECK(printed(after).rfind("DataSet rename_new:", 0) == 0);

    // the other object sharing the entry sees the new name too
    CHECK(p1.name() == "rename_new");
    CHECK(p2.name() == "rename_new");
  }

  SECTION("to the same name is a no-op")
  {
    CHECK(p1.rename("rename_old").ok());
    CHECK(DataSet::lookup("rename_old").get() == before.get());
  }

  SECTION("to a name already in use fails and changes nothing")
  {
    Params other = makeParams();
    other.setName("rename_taken");
    DataSet d3(other, FluidContext());

    CHECK_FALSE(p1.rename("rename_taken").ok());
    CHECK(p1.name() == "rename_old");
    CHECK(DataSet::lookup("rename_old").get() == before.get());
    CHECK(DataSet::lookup("rename_taken").get() != before.get());
    CHECK(sizeOf(DataSet::lookup("rename_taken")) == 0);
  }

  SECTION("to an empty name fails")
  {
    CHECK_FALSE(p1.rename("").ok());
    CHECK(p1.name() == "rename_old");
  }
}

TEST_CASE("The name is no longer a parameter", "[SharedInstanceAdaptor]")
{
  using Tree = NRTSharedInstanceAdaptor<KDTreeClient>;
  CHECK(DataSet::getParameterDescriptors().size() == 0);
  CHECK(Tree::getParameterDescriptors().size() == 2);

  Tree::ParamSetType p(Tree::getParameterDescriptors(),
                       FluidDefaultAllocator());
  p.setName("tree_a");
  Tree t(p, FluidContext());
  CHECK(p.rename("tree_b").ok());
  CHECK(Tree::lookup("tree_b"));
  CHECK_FALSE(Tree::lookup("tree_a"));
}

TEST_CASE("Shared clients declare refer, setname and getname as messages",
          "[SharedInstanceAdaptor]")
{
  auto ds = messageNames(Threaded::getMessageDescriptors());
  REQUIRE(ds.size() == DataSet::NumClientMessages + 3);
  CHECK(ds[kRefer] == "refer");
  CHECK(ds[kSetName] == "setname");
  CHECK(ds[kGetName] == "getname");
  // the client's own messages are untouched, and come first
  CHECK(ds[0] == messageNames(DataSetClient::getMessageDescriptors())[0]);

  auto tree = messageNames(NRTThreadedKDTreeClient::getMessageDescriptors());
  CHECK(tree.back() == "getname");
}

TEST_CASE("getname and setname work through the message machinery",
          "[SharedInstanceAdaptor]")
{
  Params p = makeParams();
  p.setName("msg_name_a");
  Threaded c(p, FluidContext());

  CHECK(static_cast<std::string>(c.template invoke<kGetName>(c)) ==
        "msg_name_a");

  auto r = c.template invoke<kSetName>(c, std::string("msg_name_b"));
  CHECK(r.ok());
  CHECK(p.name() == "msg_name_b"); // the host's own copy sees it
  CHECK(DataSet::lookup("msg_name_b"));
  CHECK_FALSE(DataSet::lookup("msg_name_a"));
  CHECK(static_cast<std::string>(c.template invoke<kGetName>(c)) ==
        "msg_name_b");

  Params other = makeParams();
  other.setName("msg_name_taken");
  Threaded o(other, FluidContext());
  CHECK_FALSE(c.template invoke<kSetName>(c, std::string("msg_name_taken")).ok());
  CHECK(p.name() == "msg_name_b");
}

TEST_CASE("refer works through the message machinery",
          "[SharedInstanceAdaptor]")
{
  Params pa = makeParams();
  pa.setName("msg_refer_a");
  Threaded a(pa, FluidContext());
  REQUIRE(DataSet::lookup("msg_refer_a")->load(oneEntry).ok());

  Params pb = makeParams();
  pb.setName("msg_refer_b");
  Threaded b(pb, FluidContext());
  CHECK(sizeVia(b) == 0);

  SECTION("to an existing name")
  {
    CHECK(b.template invoke<kRefer>(b, std::string("msg_refer_a")).ok());
    // the host's copy of the params followed, without the host doing anything
    CHECK(pb.name() == "msg_refer_a");
    // and messages now reach the referred-to client
    CHECK(sizeVia(b) == 1);
    // nobody uses msg_refer_b any more, so it is gone
    CHECK_FALSE(DataSet::lookup("msg_refer_b"));
    CHECK_FALSE(pb.lookup("msg_refer_b").ok());
  }

  SECTION("to an unknown name fails and changes nothing")
  {
    auto r = b.template invoke<kRefer>(b, std::string("msg_nope"));
    CHECK(r.status() == fluid::client::Result::Status::kError);
    CHECK(pb.name() == "msg_refer_b");
    CHECK(sizeVia(b) == 0);
  }

  SECTION("to its own name is a no-op")
  {
    CHECK(b.template invoke<kRefer>(b, std::string("msg_refer_b")).ok());
    CHECK(DataSet::lookup("msg_refer_b"));
  }
}

TEST_CASE("refer moves the host's parameter listeners along",
          "[SharedInstanceAdaptor]")
{
  using TreeParams = NRTThreadedKDTreeClient::ParamSetType;
  constexpr size_t treeRefer =
      NRTSharedInstanceAdaptor<KDTreeClient>::NumClientMessages;
  auto make = [] {
    return TreeParams(NRTThreadedKDTreeClient::getParameterDescriptors(),
                      FluidDefaultAllocator());
  };

  TreeParams pa = make();
  pa.setName("listen_a");
  NRTThreadedKDTreeClient a(pa, FluidContext());

  TreeParams pb = make();
  pb.setName("listen_b");
  NRTThreadedKDTreeClient b(pb, FluidContext());

  int calls = 0;
  int key = 0;
  pb.template addListener<0>([&calls] { ++calls; }, &key);

  CHECK(b.template invoke<treeRefer>(b, std::string("listen_a")).ok());
  CHECK(calls == 1); // fired once so the host can refresh what it displays

  // a change made through the other object now reaches b's listener
  pa.template set<0>(7, nullptr);
  CHECK(calls == 2);
  CHECK(pb.template get<0>() == 7);

  pb.template removeListener<0>(&key);
  pa.template set<0>(3, nullptr);
  CHECK(calls == 2);
}
