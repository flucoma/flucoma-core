/****
 Container class, based lovingly on Stroustrop's in C++PL 4th ed, but we don't need operations, as we delegate these out to Eigen wrappers in our algorithms.
 *****/
#pragma once

#include <vector>
#include <numeric>
#include <array>
#include <iostream>
#include <initializer_list>
#include <assert.h>

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
     slice
     Used for requesting slices from client code using operater() on FluidTensor and FluidTensorView.
     Implementation replicates Stroustrup's in C++PL4 (p841)
     Not sure I like the deliberate wrapping of the unsigned indices though
     The actual action happens in the FluidTensorSlice template, with some recursive
     variadic args goodness
     ********************************/
    struct Slice {
      //        /static constexpr slice all(0, std::size_t(-1),1);

      Slice() : start(-1), length(-1), stride(1) {}

      explicit Slice(size_t s) : start(s), length(-1), stride(1) {}

      Slice(size_t s, size_t l, size_t n = 1)
          : start(s), length(l), stride(n) {}

      size_t start;
      size_t length;
      size_t stride;
    };

//    slice slice::all(0, std::size_t(-1),1);


    #include "FluidTensor_Support.hpp"

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
    class FluidTensor//: public FluidTensorBase<T,N>
    {
        //embed this so we can change our mind
        using container_type = std::vector<T>;
    public:
        static constexpr size_t order = N;
        //expose this so we can use as an iterator over elements
        using iterator = typename std::vector<T>::iterator;
        using const_iterator = typename std::vector<T>::const_iterator;

//        FluidTensorView<T,N> global_view;


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
        template <typename U, size_t M>
        FluidTensor(const FluidTensor<U, M> &x)
            : mContainer(x.size()), mDesc(x.descriptor()) {
          static_assert(std::is_convertible<U, T>(),
                        "Cannot convert between container value types");

          std::copy(x.begin(), x.end(), mContainer.begin());

        }

        template <typename U, size_t M>
        explicit FluidTensor(const FluidTensorView<U, M> &x)
            : mContainer(x.size()), mDesc(0, x.descriptor().extents) {
          static_assert(std::is_convertible<U, T>(),
                        "Cannot convert between container value types");

          std::copy(x.begin(), x.end(), mContainer.begin());

        }




        //
        /****
         Conversion assignment
         ****/
        template <typename U, template <typename,size_t> class O,size_t M = N>
        enable_if_t< std::is_same<FluidTensor<U,N>, O<U,M>>() && (N>1),  FluidTensor&>
         operator=(const O<U,M>& x)
        {

            mDesc = x.descriptor();
            mContainer.assign(x.begin(), x.end());
        }

        template <typename... Dims,
                  typename = enable_if_t<isIndexSequence<Dims...>()>>
        FluidTensor(Dims... dims) : mDesc(dims...) {
          static_assert(sizeof...(dims) == N,
                        "Number of dimensions doesn't match");
          mContainer.resize(mDesc.size, 0);
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
        FluidTensor(FluidTensorInitializer<T, N> init)
            : mDesc(0, _impl::deriveExtents<N>(init)) {
          mContainer.reserve(this->mDesc.size);
          _impl::insertFlat(init, mContainer);
          assert(mContainer.size() == this->mDesc.size);

        }

        FluidTensor& operator=(FluidTensorInitializer<T,N> init)
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

        /**********************************************************
         Copy from a view
         *********************************************************/
        FluidTensor& operator=(const FluidTensorView<T,N> x)
        {
            mDesc = x.descriptor(); //we get the same size, extent and strides
            mDesc.start = 0; //but start at 0 now 
            mContainer.resize(mDesc.size);
            std::copy(x.begin(),x.end(),mContainer.begin());
            return *this;
        }

        template<typename U, size_t M>
        FluidTensor& operator=(const FluidTensorView<U,M> x)
        {
            static_assert(M <= N, "View has too many dimensions");
            static_assert(std::is_convertible<U, T>(), "Cannot convert between types");

            assert(sameExtents(mDesc, x.descriptor()));

            //Let's try this dirty, and just copy size values out of the incoming view, ignoring
            //whether dimensions match

            std::copy(x.begin(), x.end(), begin());
            return *this;
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

         This assumes row-major input, I think
         ****/
        template <typename U = T, size_t D = N,
                  typename = enable_if_t<D == 2>()>
        FluidTensor(T **input, size_t dim1, size_t dim2)
            : mContainer(dim1 * dim2, 0), mDesc(0, {dim1, dim2}) {
          for (int i = 0; i < dim1; ++i)
            std::copy(input[i], input[i] + dim2,
                      mContainer.data() + (i * dim2));
        }




        /****
         T* constructor only for 1D structure
         Allows for strided copying (e.g from interleaved audio)
         ****/
        template <typename U = T, size_t D = N,
                  typename = enable_if_t<D == 1>()>
        FluidTensor(T *input, size_t dim, size_t stride = 1)
            : mContainer(dim), mDesc(0, {dim}) {
          for (size_t i = 0, j = 0; i < dim; ++i, j += stride) {
            mContainer[i] = input[j];
          }
        }
        /***
         TODO: multidim version of the above
         input: T*, possibly interleaved
         - Return 2D dim * n_channels thing, appropriately
         strided
         – Will need varadic strides?
         ***/

        /****
        vector<T> constructor only for 1D structure

         copies the vector using vector's copy constructor
         ****/
        template <typename U = T, size_t D = N,
                  typename = enable_if_t<D == 1>()>
        FluidTensor(std::vector<T> &&input)
            : mContainer(input), mDesc(0, {input.size()}) {}

        template <typename U = T, size_t D = N,
                  typename = enable_if_t<D == 1>()>
        FluidTensor(std::vector<T> &input)
            : mContainer(input), mDesc(0, {input.size()}) {}
        /***************************************************************
         row(n) / col(n): return a FluidTensorView<T,N-1> (i.e. one dimension
         smaller) along the relevant dim. This feels like strange naming for
         N!=2 containers: like, is a face of a 3D really a 'row'? Hmm.

         Currently, this is row major, i.e. row(n) returns slices from
         dimension[0] and col from dimension[1]. These are made using
         slice_dim<>()
         ***************************************************************/


        const FluidTensorView<const T,N-1> row(const size_t i) const
        {
            assert(i < rows());
            FluidTensorSlice<N-1> row(mDesc,size_constant<0>(), i);
            return {row,mContainer.data()};
        }

        FluidTensorView<T,N-1> row(const size_t i)
        {
            assert(i < rows());
            FluidTensorSlice<N-1> row(mDesc,size_constant<0>(), i);
            return {row,mContainer.data()};
        }

        const FluidTensorView<const T,N-1> col(const size_t i) const
        {
            assert(i < cols());
            FluidTensorSlice<N-1> col(mDesc,size_constant<1>(), i);
            return {col,data()};
        }

        FluidTensorView<T,N-1> col(const size_t i)
        {
            assert(i < cols());
            FluidTensorSlice<N-1> col(mDesc,size_constant<1>(), i);
            return {col,data()};
        }

        /************
         TODO: overload rows() and cols() with no args
         to return slice iterators

         FluidTensor_dimIterator<T,N-1> rows() const;
         FluidTensor_dimIterator<T,N-1> cols() const;
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


        FluidTensorView<T, N-1> operator[](const size_t i)
        {
//            assert(i < mContainer.size());
            return row(i);
        }

        const FluidTensorView<T, N-1> operator[](const size_t i) const
        {
//            assert(i < mContainer.size());
            return row(i);
        }


        /****
         Multisubscript Element access operator(), enabled if args can
        be interpreted as indices (viz convertible to size_t)
         ****/

        template <typename... Args>
        enable_if_t<isIndexSequence<Args...>(), T &> operator()(Args... args) {
          assert(_impl::checkBounds(mDesc, args...) &&
                 "Arguments out of bounds");
          return *(data() + mDesc(args...));
         }

        // const version
         template <typename... Args>
         enable_if_t<isIndexSequence<Args...>(), const T &>
         operator()(Args... args) const {
           assert(_impl::checkBounds(mDesc, args...) &&
                  "Arguments out of bounds");
           return *(data() + mDesc(args...));
        }
//
//        /**
//         implicit cast to view
//         **/
        operator const FluidTensorView<T, N>() const
        {
            return {mDesc,data()};
        }

        operator FluidTensorView<T, N>()
        {
            return {mDesc,data()};
        }

//
//
//
//
//
//
//        /****
//         slice operator(), enabled only if args contain at least one
//         fluid::slice struct and a mixture of integer types and fluid::slices
//         ****/
        template <typename... Args>
        enable_if_t<isSliceSequence<Args...>(),
                    const FluidTensorView<const T, N>>
        operator()(const Args &... args) const {
          static_assert(sizeof...(Args) == N,
                        "Number of slices must match number of dimensions. Use "
                        "an integral constant to represent the whole of a "
                        "dimension,e.g. matrix(1,slice(0,10)).");
          //            FluidTensorSlice<N> d;
          //            d.start = _impl::do_slice(mDesc, d,args...);
          FluidTensorSlice<N> d{mDesc, args...};
          return {d, data()};
        }

        template <typename... Args>
        enable_if_t<isSliceSequence<Args...>(), FluidTensorView<T, N>>
        operator()(const Args &... args) {
          static_assert(sizeof...(Args) == N,
                        "Number of slices must match number of dimensions. Use "
                        "an integral constant to represent the whole of a "
                        "dimension,e.g. matrix(1,slice(0,10)).");
          //            FluidTensorSlice<N> d;
          //            d.start = _impl::do_slice(mDesc, d,args...);
          FluidTensorSlice<N> d{mDesc, args...};
          return {d, data()};
        }

        /************************************
         Expose begin() and end() from the container so that FluidTensor
         can be used in stl algorithms

         e.g. stl.copy(my_tensor.begin(), my_tensor.end(), my_pointer)

         TODO: const_iterators also?
         ************************************/
        iterator begin()
        {
            return mContainer.begin();
        }

        iterator end()
        {
            return mContainer.end();
        }

        const_iterator begin() const
        {
            return mContainer.cbegin();
        }

        const_iterator end() const
        {
            return mContainer.cend();
        }

        /********************************************
         property accessors. I've got away with not having separate const
         and non-const versions by making mContainer mutable. Ho hum.
         ********************************************/

        /************
         size of nth dimension
         ************/
        size_t extent(size_t n) const {return mDesc.extents[n];};
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
        size_t size() const { return mContainer.size(); }
        /***********
         Reference to internal slice description
         ***********/
        const FluidTensorSlice<N>& descriptor() const { return mDesc; }
        FluidTensorSlice<N>& descriptor(){ return mDesc; }
        /***********
         Pointer to internal data
         ***********/
        const T* data() const { return mContainer.data();}
        T* data() { return mContainer.data();}

        template<typename... Dims,
        typename = enable_if_t<isIndexSequence<Dims...>()>>
        void resize(Dims...dims)
        {
            static_assert(sizeof...(dims) == N, "Number of dimensions doesn't match");
            mDesc = FluidTensorSlice<N>(dims...);
            mContainer.resize(mDesc.size,0);
        }

        void fill(T v)
        {
            std::fill(mContainer.begin(), mContainer.end(), v); 
        }

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

            //assert(m.descriptor().extents == mDesc.extents);
            sameExtents(*this, m);

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
      container_type mContainer;
      FluidTensorSlice<N> mDesc;
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
        operator T&() {return elem;}
        operator const T&() const {

            return elem;}

        T& operator()() {return elem;}
        const T& operator()() const {return elem;}
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
    class FluidTensorView {//: public FluidTensorBase<T,N> {
        static constexpr size_t order = N;

//        using base_type = typename std::remove_cv<typename std::remove_reference<T>::type>::type;

    public:
        /*****
         STL style shorthand
         *****/
        using pointer = T*;
        using iterator = _impl::SliceIterator<T,N>;
        using const_iterator = _impl::SliceIterator<const T,N>;

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

        //Convert to a larger dimension by adding single sized
        //dimenion, a la numpy newaxis
        explicit FluidTensorView(FluidTensorView<T,N-1> x)
        {
            mDesc.start = x.descriptor().start;

            std::copy_n(x.descriptor().extents.begin(),N-1,mDesc.extents.begin()+1);
            std::copy_n(x.descriptor().strides.begin(),N-1,mDesc.strides.begin());
            mDesc.extents[0] = 1;
            mDesc.strides[N-1] = 1;
            mDesc.size = x.descriptor().size;
            mRef = x.data()-mDesc.start;
        }

        //Assign.
        // Note param by value https://stackoverflow.com/a/3279550
        //Actually, is this a bad idea? We probably want
        //different move and copy behaviour


//      //Move
//      FluidTensorView& operator=(FluidTensorView&& x)
//      {
//        if(this != &x){
//          auto m = x;
//          swap(*this,m);
//        }
//          return *this;
//      }


        FluidTensorView& operator=(const FluidTensorView& x)
        {

          assert(sameExtents(mDesc, x.descriptor()));

          std::array<size_t, N> a;

          // Get the element-wise minimum of our extents and x's
          std::transform(mDesc.extents.begin(), mDesc.extents.end(),
                         x.descriptor().extents.begin(), a.begin(),
                         [](size_t a, size_t b) { return std::min(a, b); });

          size_t count =
              std::accumulate(a.begin(), a.end(), 1, std::multiplies<size_t>());

          // Have to do this because haven't implemented += for slice iterator
          // (yet), so can't stop at arbitary offset from begin
          auto it = x.begin();
          auto ot = begin();
          for (int i = 0; i < count; ++i, ++it, ++ot) {
            *ot = *it;
            }

//            std::copy(x.begin(),stop,begin());

            return *this;
        }

        FluidTensorView& operator=(const FluidTensor<T,N>& x)
        {

          assert(sameExtents(mDesc, x.descriptor()));

          //            std::move(x.begin(), x.end(),begin());

          //            swap(*this, other);
          std::array<size_t, N> a;

          // Get the element-wise minimum of our extents and x's
          std::transform(mDesc.extents.begin(), mDesc.extents.end(),
                         x.descriptor().extents.begin(), a.begin(),
                         [](size_t a, size_t b) { return std::min(a, b); });

          size_t count =
              std::accumulate(a.begin(), a.end(), 1, std::multiplies<size_t>());

          // Have to do this because haven't implemented += for slice iterator
          // (yet), so can't stop at arbitary offset from begin
          auto it = x.begin();
          auto ot = begin();
          for (int i = 0; i < count; ++i, ++it, ++ot)
            *ot = *it;

          //            std::copy(x.begin(),stop,begin());

          return *this;
        }




        template <typename U>
        FluidTensorView& operator=(const FluidTensorView<U,N> x)
        {
            //            swap(*this, other);
            static_assert(std::is_convertible<T,U>(),"Can't convert between types");

            assert(sameExtents(mDesc, x.descriptor()));

            std::array<size_t,N> a;

            //Get the element-wise minimum of our extents and x's
            std::transform(mDesc.extents.begin(), mDesc.extents.end(), x.descriptor().extents.begin(), a.begin(), [](size_t a, size_t b){return std::min(a,b);});

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

//        template <typename U>
//        FluidTensorView& operator=(FluidTensor<U,N>& x)
//        {
//            static_assert(std::is_convertible<T,U>(),"Can't convert between types");
//            std::array<size_t,N> a;
//
//            //Get the element-wise minimum of our extents and x's
//            std::transform(mDesc.extents.begin(), mDesc.extents.end(), x.descriptor().extents.begin(), a.begin(), [](size_t a, size_t b){return std::min(a,b);});
//
//            size_t count = std::accumulate(a.begin(), a.end(), 1, std::multiplies<size_t>());
//
//            //Have to do this because haven't implemented += for slice iterator (yet),
//            //so can't stop at arbitary offset from begin
//            auto it = x.begin();
//            auto ot = begin();
//            for(int i = 0; i < count; ++i,++it,++ot)
//                *ot = *it;
//
//            //            std::copy(x.begin(),stop,begin());
//
//            return *this;
//        }



        /**********
         Construct from a slice and a pointer. This gets used by
         row() and col() of FluidTensor and FluidTensorView
         **********/

        FluidTensorView(const FluidTensorSlice<N> &s, T *p)
            : mDesc(s), mRef(p) {}

        /**
         Wrap around an arbitary pointer, with an offset and some dimensions
         **/
        template <typename... Dims,
                  typename = enable_if_t<isIndexSequence<Dims...>()>>
        FluidTensorView(T *p, size_t start, Dims... dims)
            : mDesc(start, {static_cast<size_t>(dims)...}), mRef(p) {}

        //        /***********
        //         Construct from a whole FluidTensor
        //         ***********/
        //        FluidTensorView(const FluidTensor<T,N>& x)
        //        :mDesc(x.descriptor()), mRef(x.data())
        //        {}




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
        enable_if_t<isIndexSequence<Args...>(),const T&>
        operator()(Args... args) const
        {
            assert(_impl::checkBounds(mDesc,args...)
                   && "Arguments out of bounds");
            return *(data() + mDesc(args...));
        }

        template<typename... Args>
        enable_if_t<isIndexSequence<Args...>(),T&>
        operator()(Args... args)
        {
            assert(_impl::checkBounds(mDesc,args...)
                   && "Arguments out of bounds");
            return *(data() + mDesc(args...));
        }


        /****
         slice operator(), enabled only if args contain at least one
         fluid::slice struct and a mixture of integer types and fluid::slices
         ****/
        template <typename... Args>
        enable_if_t<isSliceSequence<Args...>(), FluidTensorView<T, N>>
        operator()(const Args &... args) const {
          static_assert(sizeof...(Args) == N,
                        "Number of slices must match number of dimensions. Use "
                        "an integral constant to represent the whole of a "
                        "dimension,e.g. matrix(1,slice(0,10)).");
          FluidTensorSlice<N> d{mDesc, args...};
          // d.start = _impl::do_slice(mDesc, d,args...);
          return {d, mRef};
        }

        iterator begin()
        {
            return {mDesc,mRef};
        }

        const const_iterator begin() const
        {
            return {mDesc,mRef};
        }

        iterator end()
        {
            return {mDesc,mRef,true};
        }


        const const_iterator end() const
        {
            return {mDesc,mRef,true};
        }


        /**
         Return the size of the nth dimension (0 based)
         **/
        size_t extent(const size_t n) const
        {
            assert(n < mDesc.extents.size());
            return mDesc.extents[n];
        }

        /**
         [i] is equivalent to i. This overload allows C-style element access
         (because of the way that the <T,0> case collapses to a scalar),

         viz. for 2D you can do my_tensorview[i][j]
         **/
        FluidTensorView<T,N-1> operator[](const size_t i)
        {

            return row(i);
        }

        const FluidTensorView<T,N-1> operator[](const size_t i) const
        {

            return row(i);
        }

        /**
         Slices across the first dimension of the view
         **/
        const FluidTensorView<T,N-1> row(const size_t i) const
        {
            //FluidTensorSlice<N-1> row = _impl::slice_dim<0>(mDesc, i);
            assert(i < extent(0));
            FluidTensorSlice<N-1> row(mDesc, size_constant<0>(), i);
            return {row,mRef};
        }

        FluidTensorView<T,N-1> row(const size_t i)
        {
            //FluidTensorSlice<N-1> row = _impl::slice_dim<0>(mDesc, i);
            assert(i < extent(0));
            FluidTensorSlice<N-1> row(mDesc, size_constant<0>(), i);
            return {row,mRef};
        }

        /**
          Slices across the second dimension of the view
         **/
        const FluidTensorView<T,N-1> col(const size_t i) const
        {
            assert(i < extent(1));
            FluidTensorSlice<N-1> col(mDesc, size_constant<1>(), i);
            return {col,mRef};
        }

        FluidTensorView<T,N-1> col(const size_t i)
        {
            assert(i < extent(1));
            FluidTensorSlice<N-1> col(mDesc, size_constant<1>(), i);
            return {col,mRef};
        }

        //The extent of the first dimension
        size_t rows() const
        {
            return mDesc.extents[0];
        }

        //For order > 1, the extent of the second dimension
        size_t cols() const
        {
            return order > 1? mDesc.extents[1]: 0; 
        }

        //The total number of elements encompassed by this view
        size_t size() const
        {
            return mDesc.size;
        }

        //Fill this view with a value
        void fill(const T x)
        {
            std::fill(begin(),end(),x);
        }

        /**
         Apply some function to each element of the view.

         If using a lambda, the general form might be
         apply([](T& x){ x = ...

         viz. remember to pass a reference to x, and you don't need to return

        **/
        template <typename F>
        FluidTensorView& apply(F f)
        {
            for(auto i = begin(); i!=end(); ++i)
                f(*i);
            return *this;
        }

        //Passing by value here allows to pass r-values
        //this tacilty assumes at the moment that M is
        // a FluidTensor or FluidTensorView. Maybe this should be more explicit
        template <typename M, typename F>
        FluidTensorView& apply(M m, F f)
        {
            //TODO: ensure same size? Ot take min?
            assert(m.descriptor().extents == mDesc.extents);
            assert(!(begin() == end()));
            auto i = begin();
            auto j = m.begin();
            for(; i!=end(); ++i, ++j)
                f(*i,*j);
            return *this;
        }

        /**
         Retreive pointer to underlying data.
         **/
        const T* data()  const { return mRef + mDesc.start; }
        pointer data() { return mRef + mDesc.start; }

        /**
         Retreive description of View's shape
         **/
        const FluidTensorSlice<N> descriptor() const {return mDesc;}
        FluidTensorSlice<N> descriptor() {return mDesc;}

        friend void swap(FluidTensorView& first, FluidTensorView& second)
        {
            using std::swap;
            swap(first.mDesc, second.mDesc);
            swap(first.mRef, second.mRef);
        }

        friend std::ostream& operator<<( std::ostream& o, const FluidTensorView& t ) {
            o << '[';
            //T* p = t.mRef + t.mDesc.start;
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
//      pointer data() { return mRef + mDesc.start; }
      FluidTensorSlice<N> mDesc;
      pointer mRef;
    };

    template<typename T>
    class FluidTensorView<T,0>
    {
    public:
        using value_type = T;
        using const_value_type = const T;
        using pointer = T*;
        using reference = T&;

        FluidTensorView() = delete;

        FluidTensorView(const FluidTensorSlice<0> &s, pointer x)
            : elem(x + s.start), mStart(s.start) {}

        FluidTensorView& operator=(value_type& x)
        {
            *elem = x;
            return *this;
        }

        template<typename U>
        FluidTensorView& operator=(U& x)
        {
            static_assert(std::is_convertible<T,U>(),"Can't convert");
            *elem = x;
            return *this;
        }





        value_type operator()(){return *elem;}
        const_value_type operator()() const {return *elem;}

        operator value_type&() {return *elem;};
        operator const_value_type&() const {return *elem;}

        friend std::ostream& operator<<(std::ostream& o, const FluidTensorView& t)
        {
            o<< t();
            return o;
        }
    private:
        pointer elem;
        size_t mStart;
    };//View<T,0>

} //namespace fluid
