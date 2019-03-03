/**
@file TupleUtils.hpp
 
 Templates to help workin with std::tuples. Some of these
 are just easier to do in C++14 onwards, but hey
 
 **/
namespace fluid {
    namespace impl{
    /**
     Stand in for std::index_sequence<>
     
     This is for generating *compile_time* sequences of integers,
     using a crafty inheritence idiom that seems to be the common
     way
     e.g.
     generateSequence<3>
     -> generateSequence<2,2>
     -> generateSequence<1, 1, 2>
     -> generateSequence<0, 0, 1, 2> (specialisation on gen_seq starting with 0 kicks in)
     -> Sequence<0, 1, 2>
     Neat!
     **/
    template <int...Is> struct Sequence{};
    
    template <int N, int...Is>
    struct generateSequence: generateSequence<N - 1, N - 1, Is...> {};
    
    template <int...Is>
    struct generateSequence<0, Is...> : Sequence<Is...> {};
    
    
    /**
     Template for iterating over members of a tuple
     **/
    template<typename T, typename F, int...Is>
    void for_each(T&& t, F f, seq<Is...>)
    {
      //std::initialiser list trick for running a function over tuple items in turn
      auto l = {(f(std::get<Is>(t)),0)...};
    }
    
    template<typename ...Ts,typename F>
    void for_each_in_tuple(std::tuple<Ts...> const& t, F f)
    {
        for_each(t, f, generateSequence<sizeof...(Ts)>());
    }
    
    
}//namespace impl
}//namespace fluid

