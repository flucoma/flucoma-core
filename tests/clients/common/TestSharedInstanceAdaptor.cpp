#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>
#include <flucoma/clients/common/FluidContext.hpp>
#include <flucoma/clients/nrt/DataSetClient.hpp>
#include <flucoma/clients/nrt/FluidSharedInstanceAdaptor.hpp>
#include <flucoma/clients/nrt/KDTreeClient.hpp>
#include <flucoma/data/FluidMemory.hpp>
#include <string>

using fluid::FluidDefaultAllocator;
using fluid::client::FluidContext;
using fluid::client::kdtree::KDTreeClient;
using fluid::client::NRTSharedInstanceAdaptor;
using fluid::client::dataset::DataSetClient;

using DataSet = NRTSharedInstanceAdaptor<DataSetClient>;
using Params = DataSet::ParamSetType;

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

TEST_CASE("refer unregisters the data it was the last user of",
          "[SharedInstanceAdaptor]")
{
  Params a = makeParams();
  a.setName("refer_last_a");
  {
    Params b = makeParams();
    b.setName("refer_last_b");
    CHECK(b.refer("refer_last_a").ok()); // b was the only user of its data
  }
  Params probe = makeParams();
  CHECK_FALSE(probe.lookup("refer_last_b").ok());
  CHECK(probe.setName("refer_last_b").ok()); // used to crash
  CHECK(probe.name() == "refer_last_b");
}

TEST_CASE("A client that outlives its objects doesn't break the tables",
          "[SharedInstanceAdaptor]")
{
  // something other than an object (e.g. a query) can hold a client while
  // the last object using it goes away
  DataSet::ClientPointer held;
  {
    Params p = makeParams();
    p.setName("outlive_a");
    DataSet d(p, FluidContext());
    held = DataSet::lookup("outlive_a");
  }

  Params q = makeParams();
  q.setName("outlive_b");
  DataSet d(q, FluidContext());
  auto    client = DataSet::lookup("outlive_b");

  // while that client lives, its name stays taken
  CHECK_FALSE(q.rename("outlive_a").ok());
  CHECK(DataSet::lookup("outlive_a") == held);

  held.reset(); // so its entry is now dead

  CHECK_FALSE(DataSet::lookup("outlive_a")); // used to crash

  SECTION("a new object can take the name")
  {
    Params r = makeParams();
    r.setName("outlive_a");
    DataSet dr(r, FluidContext()); // used to crash
    CHECK(DataSet::lookup("outlive_a"));
  }

  SECTION("an object can be renamed onto it")
  {
    CHECK(q.rename("outlive_a").ok());
    CHECK(DataSet::lookup("outlive_a") == client); // used to be dropped
  }
}
