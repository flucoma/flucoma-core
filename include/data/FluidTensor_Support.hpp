/**
 Support Code for FluidTensor
 **/

#pragma once 
    /*****************************
     FluidTensorSlice describes mappings of indices to points in our storage
     ******************************/
    template <size_t N>
    struct FluidTensorSlice;
    
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

    //Alias integral_constant<size_t,N>
    template <std::size_t N>
        using size_constant = std::integral_constant<std::size_t, N>;

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
        
//        /*********
//         Makes a new FluidTensorSlice based on offsetting into an existing one
//         *********/
//        template<size_t D, size_t N>
//        fluid::FluidTensorSlice<N-1> slice_dim(const fluid::FluidTensorSlice<N>& inp,size_t idx)
//        {
//            static_assert(D<=N, "Requested dimension too big");
//
//            fluid::FluidTensorSlice<N-1> r;
//            r.size = inp.size / inp.extents[D];
//            r.start = inp.start + idx * inp.strides[D];
//            auto i = std::copy_n(inp.extents.begin(),D,r.extents.begin());
//            std::copy_n(inp.extents.begin() + D + 1, N-D-1,i);
//            auto j = std::copy_n(inp.strides.begin(),D,r.strides.begin());
//            std::copy_n(inp.strides.begin() + D + 1, N-D-1,j);
//            return r;
//        }
//
//        /************************************************
//         do_slice_dim does the hard work in making an arbitary
//         new slice from an existing one, used by the operator()
//         of FluidTensor and FluidRensorView. These are called by
//         do_slice, immediately below
//         ************************************************/
//        template<size_t N, size_t M>
//        size_t do_slice_dim(const fluid::FluidTensorSlice<M>& original_slice, fluid::FluidTensorSlice<M>& new_slice, size_t n)
//        {
//            return do_slice_dim<N>(new_slice, fluid::slice(n, 1, 1));
//        }
//
//        template<size_t N, size_t M, typename T>
//        size_t do_slice_dim(const fluid::FluidTensorSlice<M>& original_slice, fluid::FluidTensorSlice<M>& new_slice, const T& s)
//        {
//            fluid::slice reformed(s);
//            if(reformed.start >= original_slice.extents[N])
//                reformed.start = 0;
//
//            if(reformed.length > original_slice.extents[N] || reformed.start + reformed.length > original_slice.extents[N])
//                reformed.length = original_slice.extents[N] - reformed.start;
//
//            if(reformed.start + reformed.length * reformed.stride > original_slice.extents[N])
//                reformed.length = ((original_slice.extents[N] - reformed.start) + reformed.stride -1) / reformed.stride;
//
//            new_slice.extents[N] = reformed.length;
//            new_slice.strides[N] = original_slice.strides[N] * reformed.stride;
//            new_slice.size = new_slice.extents[0] * new_slice.strides[0];
//            return reformed.start * original_slice.strides[N];
//        }
//
//        /************************************************
//         do_slice recursively populates a new slice from
//         an old one, based on some variable number of slicing
//         args (that should match the number of dimensions of the
//         container or ref at hand.
//         ************************************************/
//        //Terminating conidition
//        template<size_t N>
//        size_t do_slice(const fluid::FluidTensorSlice<N>& os, fluid::FluidTensorSlice<N>& ns)
//        {
//            return 0;
//        }
//        //Recursion. Works out the offset for a dimension,
//        //and calls again, for the next dimension
//        template<size_t N, typename T, typename ...Args>
//        size_t do_slice(const fluid::FluidTensorSlice<N>& os, fluid::FluidTensorSlice<N>& ns, const T& s, const Args&... args)
//        {
//            //<N - sizeof...(Args) - 1>
//            size_t m = do_slice_dim<N - sizeof...(Args) - 1>(os, ns, s);
//            size_t n = do_slice(os, ns, args...);
//            return n+m;
//        }
        
        
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
            using value_type        = typename std::remove_const<T>::type;
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
                return /*(m_base == rhs.m_base) &&*/ (m_ptr == rhs.m_ptr);
            }
            
            bool operator != (const SliceIterator& rhs)
            {
                return !(*this==rhs);
            }
            
        private:
            /*****
             Taken from the origin implementation. It's ugleee, but
             can't think of a better way
             
             TODO: Is there a better way? It'd be nice,at least, to simplify for continguous ranges in the major dimension 
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
            FluidTensorSlice<N> m_desc;
            std::array<size_t,N> m_indexes;
            pointer m_ptr;
            pointer m_base;
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
//        template <typename T, size_t N>
//        class FluidTensor_dim_iterator
//        {
//        public:
//            /**
//             Regulation STL style boiler plate
//             **/
//            using value_type = fluid::FluidTensorView<T,N>;
//            using difference_type = std::ptrdiff_t;
//            using pointer = fluid::FluidTensorView<T,N>*;
//            using reference = fluid::FluidTensorView<T,N>&;
//            using iterator_category = std::random_access_iterator_tag;
//            using type = FluidTensor_dim_iterator<T,N>;
//            
//            FluidTensor_dim_iterator() = default;
//            
//            //copy
//            FluidTensor_dim_iterator(FluidTensor_dim_iterator& other) = default;
//            //move
//            FluidTensor_dim_iterator(FluidTensor_dim_iterator&& other) = default;
//            
//            //Construct an instance from a Tensor/TensorView
//            explicit FluidTensor_dim_iterator(const fluid::FluidTensorBase<T,N+1>& obj):container(obj){}
//            
//            //dereferenceable
//            reference operator*() const
//            {
//                std::vector<double> s;
//                return container.row(current);
//            }
//            
//            //pre- and post-incrementable
//            type& operator++()
//            {
//                ++current;
//                return *this;
//            }
//            type operator++(int)
//            {
//                type tmp = *this;
//                ++current;
//                return tmp;
//            }
//            
//            //pre- and post-decrementable
//            type& operator--()
//            {
//                --current;
//                return *this;
//            }
//            type operator--(int)
//            {
//                type tmp = *this;
//                --current;
//                return tmp;
//            }
//            
//            //Todo, this isn't actually a sufficient condition,
//            //should compare descriptors too
//            bool operator==(const type& rhs)
//            {
//                return  container.row(current).data() == rhs.row(current).data();
//            }
//            bool operator!=(const type& rhs)
//            {
//                return !(*this==rhs);
//            }
//        private:
//            const fluid::FluidTensorBase<T,N+1>& container;
//            size_t current = 0;
//        };//FluidDimIterator
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
        static constexpr std::size_t order = N;
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
        
        
        template <std::size_t M, typename T, std::size_t D>
        FluidTensorSlice(const FluidTensorSlice<M>& s,
                                      std::integral_constant<T, D>,
                                      std::size_t n)
        : size(s.size / s.extents[D]), start(s.start + n * s.strides[D])
        {
            static_assert(D <= N, "");
            static_assert(N < M, "");
            // Copy the extetns and strides, excluding the Dth dimension.
            std::copy_n(s.extents.begin() + D + 1, N - D, std::copy_n(s.extents.begin(), D, extents.begin()));
            std::copy_n(s.strides.begin() + D + 1, N - D, std::copy_n(s.strides.begin(), D, strides.begin()));
        }
        
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
       
        template<size_t M, typename... Args>
        FluidTensorSlice(FluidTensorSlice<M> s,const Args... args)
        {
            start = s.start + do_slice(s,args...);
            size = extents[0] * strides[0]; 
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
            
        FluidTensorSlice<N> transpose()
        {
          FluidTensorSlice<N> res(*this);
          std::reverse(res.extents.begin(), res.extents.end());
          std::reverse(res.strides.begin(), res.strides.end());
          return res;
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
        
        std::size_t size; //num of elements
        std::size_t start=0; //offset
        std::array<std::size_t,N> extents; //number of elements in each dimension
        std::array<std::size_t,N> strides; //offset between elements in each dimension
        
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
        
        /************************************************
         do_slice_dim does the hard work in making an arbitary
         new slice from an existing one, used by the operator()
         of FluidTensor and FluidRensorView. These are called by
         do_slice, immediately below
         ************************************************/
        template<size_t D, size_t M>
        size_t do_slice_dim(const fluid::FluidTensorSlice<M>& new_slice, size_t n)
        {
            return do_slice_dim<D>(new_slice, fluid::slice(n, 1, 1));
        }
        
        template<size_t D, size_t M>
        size_t do_slice_dim(const fluid::FluidTensorSlice<M>& ns, slice s)
        {
            
            if(s.start >= ns.extents[D])
                s.start = 0;
            
            if(s.length > ns.extents[D] || s.start + s.length > ns.extents[D])
                s.length = ns.extents[D] - s.start;
            
            if(s.start + s.length * s.stride > ns.extents[D])
                s.length = ((ns.extents[D] - s.start) + s.stride -1) / s.stride;
            
            extents[D] = s.length;
            strides[D] = ns.strides[D] * s.stride;
            size = extents[0] * strides[0];
            return s.start * ns.strides[D];
        }
        
        /************************************************
         do_slice recursively populates a new slice from
         an old one, based on some variable number of slicing
         args (that should match the number of dimensions of the
         container or ref at hand.
         ************************************************/
        //Terminating conidition
        template<size_t M>
        size_t do_slice(const fluid::FluidTensorSlice<M>& os)
        {
            return 0;
        }
        //Recursion. Works out the offset for a dimension,
        //and calls again, for the next dimension
        template<size_t M, typename T, typename ...Args>
        size_t do_slice(const fluid::FluidTensorSlice<M>& ns, const T& s, const Args&... args)
        {
            constexpr size_t D = N - sizeof...(Args) - 1;
            size_t m = do_slice_dim<D>(ns, s);
            size_t n = do_slice(ns, args...);
            return n+m;
        }
        
    };

template<std::size_t N>
    inline bool same_extents(const FluidTensorSlice<N>& a, const FluidTensorSlice<N>& b)
{
    return a.order == b.order &&
    std::equal(a.extents.begin(), a.extents.begin() + N, b.extents.begin());
}

template <typename M1, typename M2>
inline bool same_extents(const M1& a, const M2& b)
{
    return same_extents(a.descriptor(), b.descriptor()); 
}




