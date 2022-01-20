// #include <catch2/catch_test_macros.hpp>
// #include <catch2/matchers/catch_matchers_templated.hpp>
#include <catch2/catch.hpp> 

namespace fluid { 

template<typename Range>
struct EqualsRangeMatcher : public Catch::MatcherBase<Range> { //Catch::Matchers::MatcherGenericBase {
    EqualsRangeMatcher(Range&& range):
        range{ range }
    {}

    bool match(Range const& other) const override {
        return match<Range>(other); 
    }

    template<typename OtherRange>
    bool match(OtherRange const& other) const {
        using std::begin; using std::end;

        return std::equal(begin(range), end(range), begin(other), end(other));
    }

    std::string describe() const override {
        return "Equals: " + Catch::rangeToString(range);
    }

private:
    Range range;
};

template<typename Range>
auto EqualsRange(Range&& range) -> EqualsRangeMatcher<Range> {
    return EqualsRangeMatcher<Range>{std::forward<Range>(range)};
}

}