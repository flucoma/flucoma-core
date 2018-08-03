/****
 Container class, based lovingly on Stroustrop's in C++PL 4th ed, but we don't need operations, as we delegate these out to Eigen wrappers in our algorithms.
 *****/
#pragma once

#include <vector>
#include <numeric>
#include <array>
#include <Eigen/Core>
#include <iostream>
#include <initializer_list>


using std::size_t;
namespace fluid {
    /******************************************************************************
     Forward declarations
     *****************************************************************************/

    /*****************************
     FluidTensor is the main container class.
     It wraps flat continguous storage and maps indicies in n-dim to points in this storage.
     ******************************/
    template <typename T, size_t N>
    class FluidTensor; //keep trendy

    /*****************************
     FluidTensorView gives you a view over some part of the container
     ******************************/
    template <typename T, size_t N>
    class FluidTensorView;//Rename to view?

    /*****************************
     FluidTensorSlice describes mappings of indices to points in our storage
     ******************************/
    template <size_t N>
    struct FluidTensorSlice;

    /*****************************
     FluidTensorBase is a base class for ...Tensor and ...View
     ******************************/
    template <typename T, size_t N>
    class FluidTensorBase;

    /*****************************
     slice
     Used for requesting slices from client code using operater() on FluidTensor and FluidTensorView.
     Implementation replicates Stroustrup's in C++PL4 (p841)
     Not sure I like the deliberate wrapping of the unsigned indices though
     The actual action happens in the FluidTensorSlice template, with some recursive
     variadic args goodness
     ********************************/
    struct slice
    {
        slice()
        :start(-1), length(-1), stride(1)
        {}

        explicit slice(size_t s)
        :start(s), length(-1), stride(1)
        {}

        slice(size_t s, size_t l, size_t n =1)
        :start(s), length(l), stride(n)
        {}

        size_t start;
        size_t length;
        size_t stride;
    };

    /******************************************************************************
     Traits and constraints.
     TODO: move to own header
     *****************************************************************************/

    /***
     enable_if_t replicates the same structure in C++14.
     Note that BS calls this Requires() in C++PL4.

     It is used to switch functions on and off depending on template arguments.

     It defines a type coditionally, depending on whether the boolean condition
     is satisfied. The second argument defines the type it would return.

     So, e.g., for a function that returned an int, that we wanted to enable based on some template argument N being > 0, we could do

     template<size_t N>
     enable_if_t< (N > 0), int> foo(){...

     This gets quite a bit of use below, to avoid specializing whole classes
     for different dimensioned containers.
     ***/
    template <bool B, typename T = void>
    using enable_if_t = typename std::enable_if<B,T>::type;


    /****
     All() and Some() are can be used with enable_if_t to check that
     variadic template arguments satisfy some condition
    These are names that stroustrup uses, however C++17 introduces
    'conjunction' & 'disjunction' that seem to do the same, but are defined
    slightly differently

    So, one example of use is to make sure that all variadic arguments can
     be converted to a given type. Let's say size_t, and again use our function
     foo that returns int

    enable_if_t<All(std::is_convertible<size_t, Args>::value...),int>
     foo(){...

     We use these for getting different versions of operator() for arguments
     of indices and of slice specifications.

     Both All() and Some() use the common trick with variadic template args of
     recursing through the list. You'll see in both cases a 'base case' declared
     as constexpr, which is the function without any args. Then, the template below
     calls itself, consuming one more arg from the list at a time.
    ****/
    //Base case
    constexpr bool All() {return true;}
    //Recurse
    template<typename...Args>
    constexpr bool All(bool b, Args...args)
    {
        return b && All(args...);
    }
    //Base case
    constexpr bool Some() {return false;}
    //Recurse
    template<typename...Args>
    constexpr bool Some(bool b, Args...args)
    {
        return b || Some(args...);
    }

    /****
     Convinience constraint for determining if a arg list can be treated as a set of
     slice specifications. If there is at least once slice instance, and possibly a
     mixture of indcies, it returns true.
     ****/
    template<typename ...Args>
    constexpr bool is_slice_sequence()
    {
        return All((std::is_convertible<Args, size_t>() || std::is_same<Args, fluid::slice>())...)
        && Some(std::is_same<Args,fluid::slice>()...);
    }

    /*****
     Constraint for a set of indicies (all can be converted to size_t)
     *****/
    template <typename ...Args>
    constexpr bool is_index_sequence()
    {
        return All(std::is_convertible<Args, size_t>()...) ;
    }

    /***
     Is this type complex? (Not currently used
    template <typename T> struct is_complex:std::false_type{};
    template <typename T> struct is_complex<std::complex<T>>:std::true_type {};

     template <typename T>
     using if_complex_get_value_type = typename std::conditional<is_complex<T>{},typename T::value_type,T>::type;
    ****/

    /****
     Does the iterator of this type fulfill the given itertator category?
     Used by FluidTensorSlice to ensure that we have at least a ForwardIterator
     in its constructor that takes a range
     ****/
    template <typename Iterator, typename IteratorTag>
    using IsIteratorType = std::is_base_of<IteratorTag,
    typename std::iterator_traits<Iterator>::iterator_category>;

namespace _impl{
/****************************************************
 * Helper templates for the container. Put into _impl namespace to make clear that these are internal
 ******************************************************/

    /********************
    Check bounds is used to ensure that a set of dimension extents
     will fit within the FluidTensorSlice being asked for them
     ********************/
    template<size_t N, typename... Dims>
    bool check_bounds(const fluid::FluidTensorSlice<N>& slice, Dims... dims)
    {
        size_t indexes[N] {size_t(dims)...};
        return std::equal(indexes, indexes+N, slice.extents.begin(), std::less<size_t> {});
    }

    /**
     fluidtensor_init is used in parsing nested initaliser lists for instantiating
     a tensor with values from a braced list

     my_tensor{{1,2},{3,4}} would be 2D.

     These are used to recurse through a nested initialzier_list type for
     constructing a container. Given the rank of the container, we can deduce the
     type by nesting ones of one-lower rank, until we hit rank one (whereupon the
     type is an initializer_list of T

     Ending up with rank 0 should cause an error
     **/
    //Recursion
    template <typename T, size_t N>
    struct fluidtensor_init
    {
        using type = std::initializer_list<typename fluidtensor_init<T,N-1>::type>;
    };
    //Terminating case (specialize on N=1):
    template <typename T>
    struct fluidtensor_init<T,1>
    {
        using type = std::initializer_list<T>;
    };

    template <typename T>
    struct fluidtensor_init<T,0>; //undefined on purpose: things should barf if this happens

    /**
     * add_extents & derive_extents are used to take a FluidTensor_Initializer
     * (which is a template for managing nested std::initializer lists)
     * and drill down through the nested lists to work out the dimensions ('extents')
     * of the resulting structure. i.e.
     * {3,2,1} should give you extents of (3,1) [or 1,3 can't remember]
     * {{4,5,6}, {7,8,9}} -> (3,2) [ or 2,3 etc]
     *
     * Note that this happens using a recursion, and some enable_if magic
     * Immediately below is the function call when there's only a single list
     * Below that this the function that does the recursing
     **/

    //Terminating case, constrained on N==1
    template <std::size_t N, typename I, typename List>
    enable_if_t<(N==1)>
    //TODO: add constraint for FluidTensor_Initializer<T> here?
    add_extents(I& first, const List& list) {
        *first++ = list.size(); //deepest nesting
    }

    //Recursion
    template <std::size_t N, typename I, typename List>
    enable_if_t<(N > 1)>
    //TODO: add constraint for FluidTensor_Initializer<T> here?
    add_extents(I& first, const List& list) {
        //    assert(check_non_jagged(list));
        *first = list.size();
        add_extents<N-1>(++first, *list.begin());
    }

    //This is the function we call from main code.
    //It takes a list of nested exents in, and
    //drills down, filling an array that finally
    //gets returned.
    template<size_t N, typename List>
    std::array<size_t,N> derive_extents(const List& list)
    {
        std::array<size_t,N> a;
        auto f = a.begin();
        add_extents<N>(f,list);
        return a;
    }

    /*********
     Makes a new FluidTensorSlice based on offsetting into an existing one
     *********/
    template<size_t D, size_t N>
    fluid::FluidTensorSlice<N-1> slice_dim(const fluid::FluidTensorSlice<N>& inp,size_t idx)
    {
        static_assert(D<=N, "Requested dimension too big");

        fluid::FluidTensorSlice<N-1> r;
        r.size = inp.size / inp.extents[D];
        r.start = inp.start + idx * inp.strides[D];
        auto i = std::copy_n(inp.extents.begin(),D,r.extents.begin());
        std::copy_n(inp.extents.begin() + D + 1, N-D-1,i);
        auto j = std::copy_n(inp.strides.begin(),D,r.strides.begin());
        std::copy_n(inp.strides.begin() + D + 1, N-D-1,j);
        return r;
    }

    /************************************************
     do_slice_dim does the hard work in making an arbitary
     new slice from an existing one, used by the operator()
     of FluidTensor and FluidRensorView. These are called by
     do_slice, immediately below
    ************************************************/
    template<size_t N, size_t M>
    size_t do_slice_dim(const fluid::FluidTensorSlice<M>& original_slice, fluid::FluidTensorSlice<M>& new_slice, size_t n)
    {
        return do_slice_dim<N>(new_slice, fluid::slice(n, 1, 1));
    }

    template<size_t N, size_t M, typename T>
    size_t do_slice_dim(const fluid::FluidTensorSlice<M>& original_slice, fluid::FluidTensorSlice<M>& new_slice, const T& s)
    {
        fluid::slice reformed(s);
        if(reformed.start >= original_slice.extents[N])
            reformed.start = 0;

        if(reformed.length > original_slice.extents[N] || reformed.start + reformed.length > original_slice.extents[N])
            reformed.length = original_slice.extents[N] - reformed.start;

        if(reformed.start + reformed.length * reformed.stride > original_slice.extents[N])
            reformed.length = ((original_slice.extents[N] - reformed.start) + reformed.stride -1) / reformed.stride;

        new_slice.extents[N] = reformed.length;
        new_slice.strides[N] = original_slice.strides[N] * reformed.stride;
        new_slice.size = new_slice.extents[0] * new_slice.strides[0];
        return reformed.start * original_slice.strides[N];
    }

    /************************************************
     do_slice recursively populates a new slice from
     an old one, based on some variable number of slicing
     args (that should match the number of dimensions of the
     container or ref at hand.
     ************************************************/
    //Terminating conidition
    template<size_t N>
    size_t do_slice(const fluid::FluidTensorSlice<N>& os, fluid::FluidTensorSlice<N>& ns)
    {
        return 0;
    }
    //Recursion. Works out the offset for a dimension,
    //and calls again, for the next dimension
    template<size_t N, typename T, typename ...Args>
    size_t do_slice(const fluid::FluidTensorSlice<N>& os, fluid::FluidTensorSlice<N>& ns, const T& s, const Args&... args)
    {
        size_t m = do_slice_dim<N - sizeof...(Args) - 1>(os, ns, s);
        size_t n = do_slice(os, ns, args...);
        return n+m;
    }


    /**
     * These templates are for _populating_ our container with the contents of
     * a nested initializer list structure. Whereas the stuff above was about
     * working out the dimensions
     *
     * insert_flat() is what gets called,
     * and add_list recurses down. This is done with overloads: add_list immediately below
     * handles the situation where an initializer_list<T> comes in; when it hits an
     * element, rather than a list (which could, in principle, be any old thing) the
     * more general template kicks in
     **/

    //Terminating condition, actually copy
    template<class T,class Vec>
    void add_list(const T* first, const T* last, Vec& vec)
    {
        vec.insert(vec.end(), first, last);
    }

    //Iterate over list of lists, and recurse
    template<class T,class Vec>
    void add_list(const std::initializer_list<T>* first, const std::initializer_list<T>* last, Vec& vec)
    {
        for(;first!=last;++first)
            add_list(first->begin(), first->end(),vec);
    }

    //Recurse
    template<class T,class Vec>
    void insert_flat(const std::initializer_list<T> list, Vec& vec) {
        add_list(list.begin(),list.end(), vec);
    }


    /********************************************************
     STL style forward iterator for iterating over a FluidTensorSlice,
     taking strides in to account. We need this for, e.g. copying from
     a FluidTensor / FluidTensorView into another FluidTensorView

     TODO: make bidretional? Random access?
     ********************************************************/
    template <typename T, size_t N>
    struct SliceIterator
    {
        //iterator boilerplate
        using value_type        = std::remove_const<T>;
        using reference         = T&;
        using pointer           = T*;
        using difference_type   = std::ptrdiff_t;
        using iterator_category = std::forward_iterator_tag;

        SliceIterator(const FluidTensorSlice<N>& s, pointer base,bool end = false):m_desc(s), m_base(base)
        {
            std::fill_n(m_indexes.begin(), N, 0);
            if(end)
            {
                m_indexes[0] = s.extents[0];
                m_ptr = base + s.start + (s.strides[0] * s.extents[0]);
            }else
                m_ptr = base + s.start;
        }

        const FluidTensorSlice<N>& descriptor() {return m_desc;}
        T& operator*() const {return *m_ptr;}
        T* operator->()const {return m_ptr;}

        //Forward iterator pre- and post-increment
        SliceIterator& operator++()
        {
            increment();
            return *this;
        }

        SliceIterator operator++(int)
        {
            SliceIterator tmp = *this;
            increment();
            return tmp;
        }

        bool operator==(const SliceIterator& rhs)
        {
            assert(m_desc == rhs.m_desc);
            return (m_base == rhs.m_base) && (m_ptr == rhs.m_ptr);
        }

        bool operator != (const SliceIterator& rhs)
        {
            return !(*this==rhs);
        }

    private:
        /*****
         Taken from the origin implementation. It's ugleee, but
         can't think of a better way
         *****/
        void increment()
        {
            std::size_t d = N - 1;
            while (true) {
                m_ptr += m_desc.strides[d];
                ++m_indexes[d];

                // If have not yet counted to the extent of the current dimension, then
                // we will continue to do so in the next iteration.
                if (m_indexes[d] != m_desc.extents[d])
                    break;

                // Otherwise, if we have not counted to the extent in the outermost
                // dimension, move to the next dimension and try again. If d is 0, then
                // we have counted through the entire slice.
                if (d != 0) {
                    m_ptr -= m_desc.strides[d] * m_desc.extents[d];
                    m_indexes[d] = 0;
                    --d;
                } else {
                    break;
                }
            }
        }
        const FluidTensorSlice<N>& m_desc;
        std::array<size_t,N> m_indexes;
        pointer m_ptr;
        const T* const m_base;
    };

    /********************************
     STL style iterator for iterating over the rows/cols
     of a container or ref. Implements the requirements of
     bidirectional_iterator but not random_access_iterator.

     TODO: template on dimension
     TODO: actually plumb in and test
     TODO: decide if we need random access

     Imagined use case: FluidTensor and FluidTensorView
     use an overload of rows()/cols() with no argument to return
     iterator, which dereferences to refs
     e.g for(auto r: container.rows())
     std::copy(r.begin(), r.end(), max_array);

     This will need me to make FluidTensor and ...View expose
     element-wise iterator behaviour as well (i.e. use begin()
     and end() to pass on the vector iterator for FluidTensor and
     the offset pointer for FluidTensorRe
     ********************************/
    // TODO: make more efficient with reference caching?
    template <typename T, size_t N>
    class FluidTensor_dim_iterator
    {
    public:
        /**
         Regulation STL style boiler plate
         **/
        using value_type = fluid::FluidTensorView<T,N>;
        using difference_type = std::ptrdiff_t;
        using pointer = fluid::FluidTensorView<T,N>*;
        using reference = fluid::FluidTensorView<T,N>&;
        using iterator_category = std::random_access_iterator_tag;
        using type = FluidTensor_dim_iterator<T,N>;

        FluidTensor_dim_iterator() = default;

        //copy
        FluidTensor_dim_iterator(FluidTensor_dim_iterator& other) = default;
        //move
        FluidTensor_dim_iterator(FluidTensor_dim_iterator&& other) = default;

        //Construct an instance from a Tensor/TensorView
        explicit FluidTensor_dim_iterator(const fluid::FluidTensorBase<T,N+1>& obj):container(obj){}

        //dereferenceable
        reference operator*() const
        {
            std::vector<double> s;
            return container.row(current);
        }

        //pre- and post-incrementable
        type& operator++()
        {
            ++current;
            return *this;
        }
        type operator++(int)
        {
            type tmp = *this;
            ++current;
            return tmp;
        }

        //pre- and post-decrementable
        type& operator--()
        {
            --current;
            return *this;
        }
        type operator--(int)
        {
            type tmp = *this;
            --current;
            return tmp;
        }

        //Todo, this isn't actually a sufficient condition,
        //should compare descriptors too
        bool operator==(const type& rhs)
        {
            return  container.row(current).data() == rhs.row(current).data();
        }
        bool operator!=(const type& rhs)
        {
            return !(*this==rhs);
        }
    private:
        const fluid::FluidTensorBase<T,N+1>& container;
        size_t current = 0;
    };//FluidDimIterator
} //namespace _impl

    /*****
     Templated alias of nested initializer lists of type and dimension
     of given container
     ****/
    template <typename T, size_t N>
    using FluidTensor_initializer = typename _impl::fluidtensor_init<T,N>::type;

    /********************
     FluidTensorSlice describes the shape of a container or subview of a container,
     and provides a mapping between points in the flat storage and indicies expressed
     in N dimensions. It comprises extents (the size of each dimension), strides
     (the distance to travel in each dimension) and a start point.

     The primary action is around operator()

     TODO: make adaptable for column major layout
     ********************/
    template <size_t N>
    struct FluidTensorSlice
    {
        //Standard constructors
        FluidTensorSlice() = default;
        //Copy
        FluidTensorSlice(FluidTensorSlice const& other) = default;
        //Move
        FluidTensorSlice(FluidTensorSlice&& other)
        :FluidTensorSlice(){ swap(*this,other);}

        /**
         Construct slice from incoming range. Note that this is constrained
         such that range must expose a STL forward_iterator

         This is used to construct a slice from the result of derive_extents
         (which returns a std:array<size_t,N>)

         We could, in principle, loosen this constraint and make the
         constructor also work with C arrays by using std::begin/end. But
         the need isn't clear.
         **/
        template <typename R,
        typename I = typename std::remove_reference<R>::type::iterator,
        typename = enable_if_t<
        IsIteratorType<I,std::forward_iterator_tag>::value>
        >
        FluidTensorSlice(size_t s,R&& range):start(s)
        {

            assert(range.size() == N && "Input list size doesn't match dimensions");
            std::copy(range.begin(), range.end(), extents.begin());
            init();
        }

        // Less flexible version of the above
        //        FluidTensor_Slice(size_t s, std::array<size_t,N> range)
        //        {
        //            std::copy(range.begin(), range.end(), extents.begin());
        //            init();
        //        }


        //Construct from a start point and an initializer_list of dimensions
        //e.g  FluidTensorSlice<2> my_slice<2>(s, {3,4}).
        //The number of dimenions given needs
        //to match the N that the slice is templated on.
        //Because std::initializer_list doesn't expose constexpr for its
        //size etc (fixed in C++14) we can't use a static_assert
        FluidTensorSlice(size_t s, std::initializer_list<size_t> exts)
        :start(s)
        {

            assert(exts.size()==N && "Wrong number of dimensions in extents");
            std::copy(exts.begin(), exts.end(), extents.begin());
            init();
        }

        //As above, but also taking a listt of strides
        FluidTensorSlice(size_t s, std::initializer_list<size_t> exts,std::initializer_list<size_t> str)
        :start(s)
        {
            assert(exts.size()==N && "Wrong number of dimensions in extents");
            assert(str.size()==N && "Wrong number of dimensions in strides");
            std::copy(exts.begin(), exts.end(), extents.begin());
            std::copy(str.begin(),str.end(),strides.begin());
            size = extents[0] * strides[0];
        }

        //Construct from a variable number of extents. Do we need this?
        //FluidTensorSlice<2> my_slice(3,4)
        template<typename... Dims,
        typename = enable_if_t<is_index_sequence<Dims...>()>>
        FluidTensorSlice(Dims...dims)
        {
            static_assert(sizeof...(Dims) == N, "Number of arguments must match matrix dimensions");
            extents={{size_t(dims)...}};
            init();
        }


        //Assign operator
        // Note param by valuehttps://stackoverflow.com/a/3279550
        FluidTensorSlice& operator=(FluidTensorSlice other)
        {
            swap(*this, other);
            return *this;
        }

        //Operator () is used for mapping indices back onto a flat data structure
        //For dimensions > 2 this is done using inner_product.
        //BS claims in CP++PL4 that this would need optimising.
        //How? Avoid copying args? Cache indices?
        template<typename... Dims>
        enable_if_t<(N > 2) && is_index_sequence<Dims...>(),size_t>
        operator()(Dims... dims) const
        {
            static_assert(sizeof...(Dims)==N,"");
            size_t args[N] {size_t(dims)...};
            return std::inner_product(args, args+N,strides.begin(),size_t(0));
        }

        //Rather than specialise the whole class again, slimmer versions of operator()
        //for N = 1 and N = 2 using enable_if idiom
        template<size_t DIM = N>
        enable_if_t<DIM == 1, size_t>
        operator()(size_t i) const
        {
            return i * strides[0];
        }

        //Specialise for N=2
        template<size_t DIM = N>
        enable_if_t<DIM == 2, size_t>
        operator()(size_t i,size_t j) const
        {
            return i * strides[0] + j;
        }


        friend void swap(FluidTensorSlice& first, FluidTensorSlice& second)
        {
            using std::swap;

            swap(first.extents, second.extents);
            swap(first.strides, second.strides);
            swap(first.size, second.size);
            swap(first.start, second.start);
        }

        //Slices are equal if they describe the same shape
        bool operator==(const FluidTensorSlice& rhs) const
        {
            return start == rhs.start && extents==rhs.extents && strides==rhs.strides;
        }
        bool operator!=(const FluidTensorSlice& rhs) const
        {
            return !(*this == rhs);
        }

        size_t size; //num of elements
        size_t start=0; //offset
        std::array<size_t,N> extents; //number of elements in each dimension
        std::array<size_t,N> strides; //offset between elements in each dimension

    private:
        //No point calling this before extents have been filled
        //This will generate strides in *row major* order, given a set of
        //extents
        void init()
        {
            strides[N-1] = 1;
            for(size_t i = N-1; i != 0; --i)
                strides[i - 1] = strides[i] * extents[i];
            size = extents[0] * strides[0];
        }
    };



    /********************************************************
     FluidTensorBase is a base class of whose worth I am not convinced.

     But why not? Well, both FluidTensor and FluidTensor ref have specializations
     for 0-dimensions that essentially present scalars. So, much as have pure
     virtual row(n) and col(n) here would make sense for n dims, not so much
     for these.
     *********************************************************/
    template<typename T,size_t N>
    class FluidTensorBase
    {
    public:
        //embed the order as a field
        static constexpr size_t order = N;

        //
        FluidTensorBase() = default;

        //Construct from reference
//        template<typename U, size_t O>
//        FluidTensorBase(FluidTensorView<U,O>)
//        {
//            static_assert(std::is_convertible<T,U>(),"Matrix constructor: imcompatible types.");
//        }


        //These make no sense for the N=0 specializations...
        //        virtual size_t extent(size_t n) const = 0;  //#elements in given dim
        //        virtual FluidTensor_View<T, N-1> row(size_t i) const = 0;
        //        virtual FluidTensor_View<T, N-1> col(size_t i) const = 0;
    };


    /********************************************************
     FluidTensor!

     A N-dimensional container that wraps STL vector.

     Templated on an element type T and its number of dimensions.

     Currently this is set up on the assumption of row major layout
     (following BS in C++PL4). To change this to column major we would need to
     – change FluidTensorSlice (and the things that build them)
     to hold its data the other way round (or address it the other
     way round.
     - Probably make that a template argument, like Eigen.

     Calls to row(n) etc return FluidTensorView instances which are
     *views* on the container, not copies.
     *********************************************************/
    template <typename T, size_t N>
    class FluidTensor: public FluidTensorBase<T,N>
    {
        //embed this so we can change our mind
        using container_type = std::vector<T>;
    public:
        //expose this so we can use as an iterator over elements
        using iterator = typename std::vector<T>::iterator;

        //Default constructor / destructor
        explicit FluidTensor()       = default;
        ~FluidTensor()               = default;

        //Move
        FluidTensor(FluidTensor&& mv) = default;
        FluidTensor& operator=(FluidTensor&& mv) = default;

        //Copy
        FluidTensor(FluidTensor const& cp) = default;
        FluidTensor& operator=(FluidTensor const& cp) = default;


        /************************************
        Conversion constructor, should we need to convert between containers
         holding different types (e.g. float and double).

         Will fail at compile time if the types aren't convertible
         ***********************************/
        //    template <typename U, size_t M>
        //    FluidTensor(const FluidTensor<U,M>& x){
        //        static_assert(std::is_convertible<U,T>(),"Cannot convert between container value types");
        //    }
        //
        /****
         Conversion assignment
         ****/
        //    template <typename M, typename = enable_if_t<FluidTensor<M>()>>
        //    FluidTensor& operator=(const M& x);


        template<typename... Dims,
        typename = enable_if_t<is_index_sequence<Dims...>()>>
        FluidTensor(Dims ...dims):m_desc(dims...)
        {
            static_assert(sizeof...(dims) == N, "Number of dimensions doesn't match");
            m_container.resize(m_desc.size,0);
        }

        /************************************************************
         Construct/assign from a possibly nested initializer_list of elements
         (/not/ extents).

         e.g. FluidTensor<double,2> my_tensor{{1,2},
                                              {3,4}}
         This calls dervie_extents() to work out the FluidTensorSlice required

         Again, because initializer_list doesn't give constexpr fields in
         C++11, we can't fail at compile time if dims don't match, but will
         fail at runtime instead.
         ************************************************************/
        FluidTensor(FluidTensor_initializer<T,N> init)
        :m_desc(0,_impl::derive_extents<N>(init))
        {
            m_container.reserve(this->m_desc.size);
            _impl::insert_flat(init,m_container);
            assert(m_container.size() == this->m_desc.size);
        }

        FluidTensor& operator=(FluidTensor_initializer<T,N> init)
        {
            FluidTensor f = FluidTensor(init);
            return f;
        }

        /*********************************************************
         Delete the standard initalizer_list constructors
         *********************************************************/
        template <typename U>
        FluidTensor(std::initializer_list<U>) = delete;
        template <typename U>
        FluidTensor& operator=(std::initializer_list<U>) = delete;

        FluidTensor(FluidTensorView<T,N> x)
        {
            m_desc = x.descriptor();
            m_container.resize(m_desc.size);
            std::copy(x.begin(),x.end(),m_container.begin());
        }


        /*********************************************************
         Specialized constructors for particular dimensions

         2D: construct from T** [TODO: enable different interpretations
         of layout for this between column major (e.g. interleaved buffers)
         and row major (array of buffers)

         1D: construct from T* and std::vector<T>

         Why all the enable_if hoo-ha? Well, otherwise we'd need to specialise
         the whole class template. Nu-huh.
         *********************************************************/

        /****
         T** constructor, copies the data by 'hand' because input won't be
         contiguous
         ****/
        template <typename U=T,size_t D = N,typename = enable_if_t<D==2>()>
        FluidTensor(T** input, size_t dim1, size_t dim2)
        :m_container(dim1*dim2,0), m_desc(0,{dim1,dim2})
        {
            for(int i = 0; i < dim1; ++i)
                std::copy(input[i],input[i] + dim2, m_container.data() + (i * dim2 ));
        }

        /****
         T* constructor only for 1D structure
         Copies using std::vec constructor
         ****/
        template <typename U=T,size_t D = N,typename = enable_if_t<D==1>()>
        FluidTensor(T* input, size_t dim)
        :m_container(input,input+dim),m_desc(0,{dim}){}


        /****
        vector<T> constructor only for 1D structure

         copies the vector using vector's copy constructor
         ****/
        template <typename U=T,size_t D = N, typename = enable_if_t<D==1>()>
        FluidTensor(std::vector<T> &input)
        :m_container(input), m_desc(0,{input.size()}){}


        /***************************************************************
         row(n) / col(n): return a FluidTensorView<T,N-1> (i.e. one dimension
         smaller) along the relevant dim. This feels like strange naming for
         N!=2 containers: like, is a face of a 3D really a 'row'? Hmm.

         Currently, this is row major, i.e. row(n) returns slices from
         dimension[0] and col from dimension[1]. These are made using
         slice_dim<>()
         ***************************************************************/
        FluidTensorView<T,N-1> row(size_t i) const
        {
            FluidTensorSlice<N-1> row = _impl::slice_dim<0>(m_desc, i);
            return {row,data()};
        }

        FluidTensorView<T,N-1> col(size_t i) const
        {
            FluidTensorSlice<N-1> col = _impl::slice_dim<1>(m_desc, i);
            return {col,data()};
        }

        /************
         TODO: overload rows() and cols() with no args
         to return slice iterators

         FluidTensor_dim_iterator<T,N-1> rows() const;
         FluidTensor_dim_iterator<T,N-1> cols() const;
         ************/

        /***************************************************************
         operator() can be used in two ways. In both cases, the number of
         arguments needs to match the number of dimensions, and be within bounds

         (1) With a list of indices, it returns the element at those indicies.
         (2) With a mixed list of slices (at least one) and size_t s, it returns
         slices as FluidTensorView<T,N>. size_t entries indicate the whole of a given dimension at some offset, so to grab two rows 5 elements long, from a 2D matrix:

         FluidTensorView<double,2> s = my_tensor(fluid::slice(0, 2),fluid::slice(0, 5));

         to grab 3 entire columns, offset by 3:

         FluidTensorView<double,2> s3 = my_tensor(0,fluid::slice(3, 3));

         TODO: Wouldn't better slicing syntax be nice? I think Eric Niebler did some work on this as part of the Ranges_V3 library that is set to become part of
         C++20.
         ***************************************************************/
        /****
         Element access operator(), enabled if args can
        be interpreted as indices (viz convertible to size_t)
         ****/

         template<typename... Args>
         enable_if_t<is_index_sequence<Args...>(), T&>
         operator()(Args... args)
         {
             assert(_impl::check_bounds(m_desc,args...)
                    && "Arguments out of bounds");
             return *(data() + m_desc(args...));
         }

        // const version
        template<typename... Args>
        enable_if_t<is_index_sequence<Args...>(),const T&>
        operator()(Args... args) const
        {
            assert(_impl::check_bounds(m_desc,args...)
                   && "Arguments out of bounds");
            return *(data() + m_desc(args...));
        }






        /****
         slice operator(), enabled only if args contain at least one
         fluid::slice struct and a mixture of integer types and fluid::slices
         ****/
        template<typename ...Args>
        enable_if_t<is_slice_sequence<Args...>(),FluidTensorView<T, N>>
        operator()(const Args&... args) const
        {
            static_assert(sizeof...(Args)==N,"Number of slices must match number of dimensions. Use an integral constant to represent the whole of a dimension,e.g. matrix(1,slice(0,10)).");
            FluidTensorSlice<N> d;
            d.start = _impl::do_slice(m_desc, d,args...);
            return {d,data()};
        }
        /************************************
         Expose begin() and end() from the container so that FluidTensor
         can be used in stl algorithms

         e.g. stl.copy(my_tensor.begin(), my_tensor.end(), my_pointer)

         TODO: const_iterators also?
         ************************************/
        iterator begin()
        {
            return m_container.begin();
        }

        iterator end()
        {
            return m_container.end();
        }

        /********************************************
         property accessors. I've got away with not having separate const
         and non-const versions by making m_container mutable. Ho hum.
         ********************************************/

        /************
         size of nth dimension
         ************/
        size_t extent(size_t n) const {return m_desc.extents[n];};
        /************
         size of 0th dimension
         ************/
        size_t rows() const { return extent(0);}
        /************
         size of 1st dimension
         ************/
        size_t cols() const { return extent(1);}
        /************
         Total number of elements
         ************/
        size_t size() const { return m_container.size(); }
        /***********
         Reference to internal slice description
         ***********/
        const FluidTensorSlice<N>& descriptor() const { return m_desc; }
        /***********
         Pointer to internal data
         ***********/
        T* data() const { return m_container.data();}

        template <typename F>
        FluidTensor& apply(F f)
        {
            for(auto i = begin(); i!=end(); ++i)
                f(*i);
            return *this;
        }

        //Passing by value here allows to pass r-values
        template <typename M, typename F>
        FluidTensor& apply(M m, F f)
        {
            //TODO: ensure same size? Ot take min?
            assert(m.descriptor().extents == m_desc.extents);

            auto i = begin();
            auto j = m.begin();
            for(; i!=end(); ++i, ++j)
                f(*i,*j);
            return *this;
        }

        /***************
         Operator << for printing to console. This recurses down through rows
         (i.e. it will call << for FluidTensorView and burrow down to N=0)
         ***************/
        friend std::ostream& operator<<( std::ostream& o, const FluidTensor& t ) {
            o << '[';
            for(int i = 0; i < t.rows(); ++i)
            {
                o  << t.row(i);
                if(i+1 != t.rows())
                    o << ',';
            }
            o << ']';
            return o;
        }
    private:
        //TODO: I'm not sure about this. The alternative to having this mutable
        //(i.e. returns non-const pointer to data()) is having separate non-const and
        //const accessor for evertything that touches the container.
        //Strictly, calling row(i) doesn't alter the state of the object, but can now
        //mess with the contained data all the time. 98
        mutable container_type m_container;
        FluidTensorSlice<N> m_desc;
    };

    /**
     A 0-dim container is just a scalar...

     TODO: does this expose all the methods it needs?
     **/
    template<typename T>
    class FluidTensor<T,0>
    {
    public:
        static constexpr size_t order = 0;
        using value_type = T;

        FluidTensor(const T& x):elem(x){}
        FluidTensor& operator=(const T& value){elem = value; return this;}

        T& operator()() const {return elem;}
        size_t size() const {return 1;}
    private:
        T elem;
    };

    /****************************************************************
     FluidTensorView

     View class for FluidTensor: just houses a pointer to
     the container's data. Because of this there is a (unavoidable) risk
     of dangling references, should  a FluidTensorView outlive its FluidTensor
     (as with all pointer things). It's the user's responsibility (as with all
     pointer things) to not do this.
     ****************************************************************/
    template<typename T,size_t N>
    class FluidTensorView: public FluidTensorBase<T,N> {
        static constexpr size_t order = N;
    public:
        /*****
         STL style shorthand
         *****/
        using pointer = T*;
        using iterator = _impl::SliceIterator<T,N>;

        /*****
         No default constructor, doesn't make sense
         ******/
        FluidTensorView() = delete;

        /*****
         Default destructor
         *****/
        ~FluidTensorView() = default;

        //Move
        FluidTensorView(FluidTensorView&& other)
        {
            swap(*this,other);
        }

        //Copy
        FluidTensorView(FluidTensorView const&) = default;

        //Assign.
        // Note param by value https://stackoverflow.com/a/3279550
        //Actually, is this a bad idea? We probably want
        //different move and copy behaviour
        FluidTensorView& operator=(FluidTensorView x)
        {
//            swap(*this, other);

            std::array<size_t,N> a;

            //Get the element-wise minimum of our extents and x's
            std::transform(m_desc.extents.begin(), m_desc.extents.end(), x.descriptor().extents.begin(), a.begin(), [](size_t a, size_t b){return std::min(a,b);});

            size_t count = std::accumulate(a.begin(), a.end(), 1, std::multiplies<size_t>());

            //Have to do this because haven't implemented += for slice iterator (yet),
            //so can't stop at arbitary offset from begin
            auto it = x.begin();
            auto ot = begin();
            for(int i = 0; i < count; ++i,++it,++ot)
                *ot = *it;

//            std::copy(x.begin(),stop,begin());

            return *this;
        }

        //Assign from FluidTensor = copy
        //Respect the existing extents, rather than the FluidTensor's
        FluidTensorView& operator=(FluidTensor<T,N>& x)
        {
            std::array<size_t,N> a;

            //Get the element-wise minimum of our extents and x's
            std::transform(m_desc.extents.begin(), m_desc.extents.end(), x.descriptor().begin(), a.begin(), std::less<size_t>());

            size_t count = std::accumulate(a.begin(), a.end(), 1, std::multiplies<size_t>());

            //Have to do this because haven't implemented += for slice iterator (yet),
            //so can't stop at arbitary offset from begin
            auto it = x.begin();
            auto ot = begin();
            for(int i = 0; i < count; ++i,++it,++ot)
                *ot = *it;

            //            std::copy(x.begin(),stop,begin());

            return *this;
        }



        /**********
         Construct from a slice and a pointer. This gets used by
         row() and col() of FluidTensor and FluidTensorView
         **********/
        FluidTensorView(const FluidTensorSlice<N>& s, T* p):m_desc(s), m_ref(p){}

        /***********
         Construct from a whole FluidTensor
         ***********/
        FluidTensorView(const FluidTensor<T,N>& x)
        :m_desc(x.descriptor()), m_ref(x.data())
        {}

        size_t size() const
        {
            return m_desc.size;
        }


        /**********
         Disable assigning a FluidTensorView from an r-value FluidTensor, as that's a gurranteed
         memory leak, i.e. you can't do
         FluidTensorView<double,1> r = FluidTensor(double,2);
        **********/
        FluidTensorView(FluidTensor<T,N>&& r) = delete;

        /****
         Element access operator(), enabled if args can
         be interpreted as indices (viz convertible to size_t)
         ****/
        template<typename... Args>
        enable_if_t<is_index_sequence<Args...>(),const T&>
        operator()(Args... args) const
        {
            assert(_impl::check_bounds(m_desc,args...)
                   && "Arguments out of bounds");
            return *(data() + m_desc(args...));
        }

        /****
         slice operator(), enabled only if args contain at least one
         fluid::slice struct and a mixture of integer types and fluid::slices
         ****/
        template<typename ...Args>
        enable_if_t<is_slice_sequence<Args...>(),FluidTensorView<T, N>>
        operator()(const Args&... args) const
        {
            static_assert(sizeof...(Args)==N,"Number of slices must match number of dimensions. Use an integral constant to represent the whole of a dimension,e.g. matrix(1,slice(0,10)).");
            FluidTensorSlice<N> d;
            d.start = _impl::do_slice(m_desc, d,args...);
            return {d,data()};
        }


//        template<typename... Args> //TODO type-checking voodoo a la Stroustrup
//        T& operator()(Args... args)
//        {
//            assert(_impl::check_bounds(m_desc, args...)
//                   && "Bounds out of range");
//            return *(data() + m_desc(args...));
//        }
//
//        template<typename... Args> //TODO type-checking voodoo a la Stroustrup
//        const T& operator()(Args... args) const
//        {
//            assert(_impl::check_bounds(m_desc, args...)
//                   && "Bounds out of range");
//            return *(data() + m_desc(args...));
//        }

        iterator begin()
        {
            return {m_desc,m_ref};
        }

        iterator end()
        {
            return {m_desc,m_ref,true};
        }

        size_t extent(size_t n) const
        {
            return m_desc.extents[n];
        }

        FluidTensorView<T,N-1> row(size_t i) const
        {
            FluidTensorSlice<N-1> row = _impl::slice_dim<0>(m_desc, i);
            return {row,m_ref};
        }

        FluidTensorView<T,N-1> col(size_t i) const
        {
            FluidTensorSlice<N-1> col = _impl::slice_dim<1>(m_desc, i);
            return {col,m_ref};
        }

        size_t rows() const
        {
            return m_desc.extents[0];
        }

        template <typename F>
        FluidTensorView& apply(F f)
        {
            for(auto i = begin(); i!=end(); ++i)
                f(*i);
            return *this;
        }

        //Passing by value here allows to pass r-values
        template <typename M, typename F>
        FluidTensorView& apply(M m, F f)
        {
            //TODO: ensure same size? Ot take min?
            assert(m.descriptor().extents == m_desc.extents);
            assert(!(begin() == end()));
            auto i = begin();
            auto j = m.begin();
            for(; i!=end(); ++i, ++j)
                f(*i,*j);
            return *this;
        }



        pointer data()     const           { return m_ref + m_desc.start; }
        FluidTensorSlice<N> descriptor() {return m_desc;}


        friend void swap(FluidTensorView& first, FluidTensorView& second)
        {
            using std::swap;
            swap(first.m_desc, second.m_desc);
            swap(first.m_ref, second.m_ref);
        }


        friend std::ostream& operator<<( std::ostream& o, const FluidTensorView& t ) {
            o << '[';
            //T* p = t.m_ref + t.m_desc.start;
            for(size_t i = 0; i < t.rows();++i)
            {
                //FluidTensor_View<T,N-1> row = t.row(i);
                o << t.row(i);
                if(i+1 != t.rows())
                    o << ',';
            }

            o << ']';
            return o;
        }
    private:
        FluidTensorSlice<N> m_desc;
        pointer m_ref;
    };

    template<typename T>
    class FluidTensorView<T,0>
    {
    public:
        using value_type = T;
        using pointer = T*;
        using reference = T&;

        FluidTensorView() = delete;

        FluidTensorView(const FluidTensorSlice<0>& s, pointer x):elem(x + s.start){}

        FluidTensorView& operator=(reference x)
        {
            *elem = x;
            return *this;
        }

        value_type operator()()const {return *elem;}
        operator T&() {return *elem;};

        friend std::ostream& operator<<(std::ostream& o, const FluidTensorView& t)
        {
            o<< t();
            return o;
        }
    private:
        pointer elem;
    };//View<T,0>

} //namespace fluid
