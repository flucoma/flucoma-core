#pragma once

#include "../common/MemoryBufferAdaptor.hpp"
#include "../common/BufferAdaptor.hpp"
#include "../common/FluidBaseClient.hpp"
#include "../common/OfflineClient.hpp"
#include "../common/ParameterTypes.hpp"
#include "../common/ParameterSet.hpp"
#include "../common/SpikesToTimes.hpp"
#include "../../data/FluidTensor.hpp"
#include "../../data/TensorTypes.hpp"

#include <vector>
#include <thread>

namespace fluid{
namespace client{

namespace impl{
//////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename B>
auto constexpr makeWrapperInputs(B b)
{
    return defineParameters(std::forward<B>(b),
          LongParam("startFrame", "Source Offset",0, Min(0)),
          LongParam("numFrames","Number of Frames", -1),
          LongParam("startChan","Start Channel",0, Min(0)),
          LongParam("numChans","Number of Channels", -1)
          );
}

template<typename...B>
auto constexpr makeWrapperOutputs(B...b)
{
  return defineParameters(std::forward<B>(b)...);
}

template<typename T,size_t N, size_t...Is>
auto constexpr spitIns(T(&a)[N],std::index_sequence<Is...>)
{
    return makeWrapperInputs(std::forward<T>(a[Is])...);
}

template<typename T,size_t N, size_t...Is>
auto constexpr spitOuts(T(&a)[N],std::index_sequence<Is...>)
{
    return makeWrapperOutputs(std::forward<T>(a[Is])...);
}
//////////////////////////////////////////////////////////////////////////////////////////////////////
using InputBufferSpec = ParamSpec<InputBufferT,Fixed<false>>;
using BufferSpec = ParamSpec<BufferT,Fixed<false>>;
//////////////////////////////////////////////////////////////////////////////////////////////////////

template<size_t...Is, size_t...Js,typename...Ts, typename...Us>
constexpr auto joinParameterDescriptors(ParameterDescriptorSet<std::index_sequence<Is...>,std::tuple<Ts...>> x,ParameterDescriptorSet<std::index_sequence<Js...>,std::tuple<Us...>> y)
{
  return ParameterDescriptorSet<
            typename JoinOffsetSequence<std::index_sequence<Is...>, std::index_sequence<Js...>>::type,
        std::tuple<Ts...,Us...>> {std::tuple_cat(x.descriptors(),y.descriptors())};
}

//////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename HostMatrix,typename HostVectorView>
struct StreamingControl;

template<template <typename, typename> class AdaptorType, class RTClient, typename ParamType, ParamType& PD, size_t Ins, size_t Outs>
class NRTClientWrapper: public OfflineIn, public OfflineOut
{
public:
  //Host buffers are always float32 (?)
  using HostVector     =  FluidTensor<float,1>;
  using HostMatrix     =  FluidTensor<float,2>;
  using HostVectorView =  FluidTensorView<float,1>;
  using HostMatrixView =  FluidTensorView<float,2>;
  static constexpr auto isControl = std::is_same<AdaptorType<HostMatrix,HostVectorView>,StreamingControl<HostMatrix,HostVectorView>>();
  static constexpr auto  decideOuts  =  isControl ? 1: Outs;

  using ParamDescType = ParamType;
  using ParamSetType = ParameterSet<ParamDescType>;
  using ParamSetViewType = ParameterSetView<ParamDescType>;
  using RTParamDescType = typename RTClient::ParamDescType;
  using RTParamSetViewType = ParameterSetView<typename RTClient::ParamDescType>;
  
  using MessageSetType = typename RTClient::MessageSetType;
  
  constexpr static ParamDescType& getParameterDescriptors() { return PD; }

  //The client will be accessing its parameter by a bunch of indices that need ofsetting now
//  using Client = RTClient<impl::ParameterSet_Offset<Params,ParamOffset>,T,U>;
// None of that for outputs though

  static constexpr size_t ParamOffset  = (Ins*5) + decideOuts;
  using WrappedClient = RTClient;//<ParameterSet_Offset<Params,ParamOffset>,T>;

  static auto& getMessageDescriptors() { return RTClient::getMessageDescriptors(); }
  
  NRTClientWrapper(ParamSetViewType& p)
    : mParams{p}
    , mRealTimeParams{RTClient::getParameterDescriptors(), p.template subset<ParamOffset>()}
    , mClient{mRealTimeParams}
  {}

  template <std::size_t N> auto& get() noexcept { return mParams.get().template get<N>(); }
//  template <std::size_t N> bool changed() noexcept { return mParams.template changed<N>(); }

  size_t audioChannelsIn()    const noexcept { return 0; }
  size_t audioChannelsOut()   const noexcept { return 0; }
  size_t controlChannelsIn()  const noexcept { return 0; }
  size_t controlChannelsOut() const noexcept { return 0; }
  ///Map delegate audio / control channels to audio buffers
  size_t audioBuffersIn()  const noexcept { return mClient.audioChannelsIn();  }
  size_t audioBuffersOut() const noexcept { return isControl ? 1 : mClient.audioChannelsOut(); }

  void setParams(ParamSetViewType& p)
  {
    mParams = p;
    mRealTimeParams = RTParamSetViewType(RTClient::getParameterDescriptors(), p.template subset<ParamOffset>());
    mClient.setParams(mRealTimeParams);
  }

  template<size_t N,typename...Args>
  decltype(auto) invoke(Args&&...args)
  {
    return invokeDelegate<N>(std::forward<Args>(args)...);
  }

  Result process(FluidContext& c)
  {
    auto constexpr inputCounter = std::make_index_sequence<Ins>();
    auto constexpr outputCounter = std::make_index_sequence<decideOuts>();

    auto inputBuffers  = fetchInputBuffers(inputCounter);
    auto outputBuffers = fetchOutputBuffers(outputCounter);

    std::array<intptr_t,Ins> inFrames;
    std::array<intptr_t,Ins> inChans;

    //check buffers exist
    int count = 0;
    for(auto&& b: inputBuffers)
    {
      
      intptr_t requestedFrames= b.nFrames;
      intptr_t requestedChans= b.nChans;
      
      auto rangeCheck = bufferRangeCheck(b.buffer, b.startFrame, requestedFrames, b.startChan, requestedChans);
      
      if(!rangeCheck.ok()) return rangeCheck;

      inFrames[count] = requestedFrames;
      inChans[count] =  requestedChans;
      mClient.sampleRate(BufferAdaptor::ReadAccess(b.buffer).sampleRate());
      count++;
    }
    
    if(std::all_of(outputBuffers.begin(), outputBuffers.end(),[](auto& b){
      
      if(!b) return true;
      
      BufferAdaptor::Access buf(b);
      return !buf.exists();
    }))
      return {Result::Status::kError, "No valid output has been set" }; //error
    
    
    Result r{Result::Status::kOk,""};
    
    //Remove non-existent output buffers from the output buffers vector, so clients don't try and use them
    std::transform(outputBuffers.begin(), outputBuffers.end(),outputBuffers.begin(), [&r](auto& b)->BufferAdaptor*
    {
      
      if(!b) return nullptr;
      BufferAdaptor::Access buf(b);
      if(! buf.exists())
      {
        r.set(Result::Status::kWarning);
        r.addMessage("One or more of your output buffers doesn't exist\n");
      }
      return buf.exists()? b : nullptr;
    }); 

    

    size_t numFrames   = *std::min_element(inFrames.begin(),inFrames.end());
    size_t numChannels = *std::min_element(inChans.begin(), inChans.end());
    
    Result processResult = AdaptorType<HostMatrix,HostVectorView>::process(mClient,inputBuffers,outputBuffers,numFrames,numChannels,c);

    if(!processResult.ok())
    {
      r.set(processResult.status());
      r.addMessage(processResult.message());
    }

    return r;
  }
private:

  template<size_t I>
  BufferProcessSpec fetchInputBuffer()
  {
    return {get<I>().get(),get<I+1>(), get<I+2>(),get<I+3>(),get<I+4>()};
  }

  template<size_t...Is>
  std::array<BufferProcessSpec, sizeof...(Is)> fetchInputBuffers(std::index_sequence<Is...>)
  {
    return {fetchInputBuffer<Is*5>()...};
  }

  template<size_t...Is>
  std::array<BufferAdaptor*,sizeof...(Is)> fetchOutputBuffers(std::index_sequence<Is...>)
  {
    return {get<Is + (Ins*5)>().get()...};
  }


  //We need to delegate invoke calls for message back to the actual RT client,
  //where the actual functions live: peel off the 1st client argument, and replace with our client instance
  template<size_t N,typename NRTClient, typename...Args>
  decltype(auto) invokeDelegate(NRTClient&, Args&&...args)
  {
    return mClient.template invoke<N>(mClient, std::forward<Args>(args)...);
  }

  RTParamSetViewType    mRealTimeParams;
  std::reference_wrapper<ParamSetViewType>     mParams;
  WrappedClient         mClient;
};
//////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename HostMatrix, typename HostVectorView>
struct Streaming
{
  template <typename Client,typename InputList, typename OutputList>
  static Result process(Client& client, InputList& inputBuffers,OutputList& outputBuffers, size_t nFrames, size_t nChans, FluidContext& c)
  {
    //To account for process latency we need to copy the buffers with padding
    std::vector<HostMatrix> outputData;
    std::vector<HostMatrix> inputData;

    outputData.reserve(outputBuffers.size());
    inputData.reserve(inputBuffers.size());

    size_t padding = client.latency();

    std::fill_n(std::back_inserter(outputData), outputBuffers.size(), HostMatrix(nChans,nFrames + padding));
    std::fill_n(std::back_inserter(inputData), inputBuffers.size(), HostMatrix(nChans,nFrames + padding));

    double sampleRate{0};

    for(int i = 0; i < nChans; ++i)
    {
      std::vector<HostVectorView> inputs;
      inputs.reserve(inputBuffers.size());
      for(int j = 0; j < inputBuffers.size(); ++j)
      {
        BufferAdaptor::ReadAccess thisInput(inputBuffers[j].buffer);
        if (i==0 && j==0) sampleRate = thisInput.sampleRate();
        inputData[j].row(i)(Slice(0,nFrames)) = thisInput.samps(inputBuffers[j].startFrame,nFrames,inputBuffers[j].startChan + i);
        inputs.emplace_back(inputData[j].row(i));
      }

      std::vector<HostVectorView> outputs;
      outputs.reserve(outputBuffers.size());
      for(int j = 0; j < outputBuffers.size(); ++j)
        outputs.emplace_back(outputData[j].row(i));
      
      if(c.task()) c.task()->iterationUpdate(i, nChans);
      
      client.process(inputs,outputs,c, true);
    }

    for(int i = 0; i < outputBuffers.size(); ++i)
    {
      if(!outputBuffers[i]) continue;
      BufferAdaptor::Access thisOutput(outputBuffers[i]);
      Result r = thisOutput.resize(nFrames,nChans,sampleRate);
      if(!r.ok()) return r;
      for(int j = 0; j < nChans; ++j)
        thisOutput.samps(j) = outputData[i].row(j)(Slice(padding));
    }
    
    return {};
  }
};
//////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename HostMatrix,typename HostVectorView>
struct StreamingControl
{
  template <typename Client,typename InputList, typename OutputList>
  static Result process(Client& client, InputList& inputBuffers,OutputList& outputBuffers, size_t nFrames, size_t nChans, FluidContext& c)
  {
        //To account for process latency we need to copy the buffers with padding
      std::vector<HostMatrix> inputData;
      size_t nFeatures = client.controlChannelsOut();
//      outputData.reserve(nFeatures);
      inputData.reserve(inputBuffers.size());

      size_t padding = client.latency();
      size_t controlRate = client.controlRate();
      size_t nHops = (nFrames + padding) / controlRate;

      //in contrast to the plain streaming case, we're going to call process() iteratively with a vector size = the control vector size, so we get KR where expected
      //TODO make this whole mess less baroque and opaque

      std::fill_n(std::back_inserter(inputData), inputBuffers.size(), HostMatrix(nChans,nFrames+padding));
      HostMatrix outputData(nChans * nFeatures, nHops);
      double sampleRate{0};
      //Copy input data
      for(int i = 0; i < nChans; ++i)
      {
        for(int j = 0; j < inputBuffers.size(); ++j)
        {
          BufferAdaptor::ReadAccess thisInput(inputBuffers[j].buffer);
          if(i==0 && j==0) sampleRate = thisInput.sampleRate();
          inputData[j].row(i)(Slice(0,nFrames)) = thisInput.samps(inputBuffers[j].startFrame,nFrames,inputBuffers[j].startChan + i);
        }
      }
    FluidTask* task = c.task();
    FluidContext dummyContext;
    for(int i = 0; i < nChans; ++i)
    {
      for(int j = 0; j < nHops; ++j )
      {
        size_t t = j * controlRate;
        std::vector<HostVectorView> inputs;
        inputs.reserve(inputBuffers.size());
        std::vector<HostVectorView> outputs;
        outputs.reserve(outputBuffers.size());
        for(int k = 0; k < inputBuffers.size(); ++k)
          inputs.emplace_back(inputData[k].row(i)(Slice(t,controlRate)));

        for(int k = 0; k < nFeatures; ++k)
          outputs.emplace_back(outputData.row(k + i*nFeatures)(Slice(j,1)));

        client.process(inputs,outputs,dummyContext, true);
        
        if(task && !task->processUpdate(j + 1 + (nHops * i),nHops * nChans)) break;
      }
    }
    
    BufferAdaptor::Access thisOutput(outputBuffers[0]);
    Result resizeResult = thisOutput.resize(nHops - 1,nChans * nFeatures,sampleRate / controlRate);
    if(!resizeResult.ok()) return resizeResult;

    for(int i = 0; i < nFeatures; ++i)
    {
      for(int j = 0; j < nChans; ++j)
        thisOutput.samps(i + j * nFeatures) = outputData.row(i + j * nFeatures)(Slice(1));
    }
    
    return {};
    
  }
};


//////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename HostMatrix,typename HostVectorView>
struct Slicing
{
  template <typename Client,typename InputList, typename OutputList>
  static Result process(Client& client, InputList& inputBuffers,OutputList& outputBuffers, size_t nFrames, size_t nChans, FluidContext& c)
  {

    assert(inputBuffers.size() == 1);
    assert(outputBuffers.size() == 1);
    size_t padding = client.latency();
    HostMatrix monoSource(1,nFrames + padding);

    BufferAdaptor::ReadAccess src(inputBuffers[0].buffer);
    // Make a mono sum;
    for (size_t i = 0; i < nChans; ++i)
      monoSource.row(0)(Slice(0,nFrames)).apply(src.samps(i), [](float &x, float y) { x += y; });

    HostMatrix onsetPoints(1,nFrames + padding);

    std::vector<HostVectorView> input  {monoSource.row(0)};
    std::vector<HostVectorView> output {onsetPoints.row(0)};

    client.process(input,output,c,true);

    return impl::spikesToTimes(onsetPoints(0,Slice(padding,nFrames)), outputBuffers[0], 1, inputBuffers[0].startFrame, nFrames,src.sampleRate());
  }
};

} //namespace impl
//////////////////////////////////////////////////////////////////////////////////////////////////////

template<class RTClient, typename Params, Params& PD, size_t Ins, size_t Outs>
using NRTStreamAdaptor = impl::NRTClientWrapper<impl::Streaming, RTClient, Params, PD, Ins, Outs>;

template<class RTClient,typename Params, Params& PD, size_t Ins, size_t Outs>
using NRTSliceAdaptor = impl::NRTClientWrapper<impl::Slicing, RTClient, Params, PD, Ins, Outs>;

template<class RTClient,typename Params, Params& PD, size_t Ins, size_t Outs>
using NRTControlAdaptor = impl::NRTClientWrapper<impl::StreamingControl, RTClient, Params, PD, Ins, Outs>;


//////////////////////////////////////////////////////////////////////////////////////////////////////

template<template <typename T> class RTClient, size_t Ms>
auto constexpr makeNRTParams(impl::InputBufferSpec&& in, impl::BufferSpec(&& out)[Ms])
{
  return impl::joinParameterDescriptors(impl::joinParameterDescriptors(impl::makeWrapperInputs(in), impl::spitOuts(out, std::make_index_sequence<Ms>())), RTClient<double>::getParameterDescriptors());
}
  
//////////////////////////////////////////////////////////////////////////////////////////////////////
  
template<typename NRTClient>
class NRTThreadingAdaptor : public OfflineIn, public OfflineOut
{
public:
  using ClientPointer = typename std::shared_ptr<NRTClient>;
  using ParamDescType = typename NRTClient::ParamDescType;
  using ParamSetType = typename NRTClient::ParamSetType;
  using ParamSetViewType = typename NRTClient::ParamSetViewType;
  using MessageSetType = typename NRTClient::MessageSetType;
  
  constexpr static ParamDescType& getParameterDescriptors() { return NRTClient::getParameterDescriptors(); }
  constexpr static auto& getMessageDescriptors() { return NRTClient::getMessageDescriptors();}

  size_t audioChannelsIn()    const noexcept { return 0; }
  size_t audioChannelsOut()   const noexcept { return 0; }
  size_t controlChannelsIn()  const noexcept { return 0; }
  size_t controlChannelsOut() const noexcept { return 0; }
  size_t audioBuffersIn()  const noexcept { return ParamDescType:: template NumOf<InputBufferT>();   }
  size_t audioBuffersOut() const noexcept { return ParamDescType:: template NumOf<BufferT>();  }

  NRTThreadingAdaptor(ParamSetType& p)
   : mHostParams{p}, mClient{new NRTClient{mHostParams}}
  {}
    
  ~NRTThreadingAdaptor()
  {
    if (mThreadedTask)
    {
      mThreadedTask->cancel(true);
      mThreadedTask.release();
    }
  }
  
  Result process()
  {
    if (mThreadedTask)
      return {Result::Status::kError, "already processing"};
    
    Result result;
    mThreadedTask = std::unique_ptr<ThreadedTask>(new ThreadedTask(mClient,mHostParams, mSynchronous, result));
    
    if (mSynchronous)
      mThreadedTask = nullptr;
    
    return result;
  }
  
  template<size_t N, typename T, typename...Args>
  decltype(auto) invoke(T& client, Args&&...args)
  {
    assert(mClient.get());
    using ReturnType = typename MessageSetType::template MessageDescriptorAt<T,N>::ReturnType;
    if (mThreadedTask)
      return ReturnType{Result::Status::kError, "Already processing"};
    return mClient-> template invoke<N>(client, std::forward<Args>(args)...);
  }
  
    
  ProcessState checkProgress(Result& result)
  {
    if (mThreadedTask)
    {
      auto state = mThreadedTask->checkProgress(result);
      
      if (state == kDone)
        mThreadedTask = nullptr;
      
      return state;
    }

    return kNoProcess;
  }
    
  void setSynchronous(bool synchronous)
  {
    mSynchronous = synchronous;
  }
  
  double progress()
  {
    return mThreadedTask ? mThreadedTask->mTask.progress() : 0.0;
  }
  
  void cancel()
  {
    if (mThreadedTask)
      mThreadedTask->cancel(false);
  }
  
  bool done()
  {
    return mThreadedTask ? mThreadedTask->mState == kDone : false;
  }
  
private:
    


  struct ThreadedTask
  {
    template<size_t N, typename T>
    struct BufferCopy
    {
      void operator()(typename T::type& param)
      {
        if (param)
          param = typename T::type(new MemoryBufferAdaptor(param));
      }
    };
    
    template<size_t N, typename T>
    struct BufferCopyBack
    {
      void operator()(typename T::type& param)
      {
        if(param) static_cast<MemoryBufferAdaptor*>(param.get())->copyToOrigin();
      }
    };
    
    template<size_t N, typename T>
    struct BufferDelete
    {
      void operator()(typename T::type& param)
      {
        param.reset();
      }
    };
      
    ThreadedTask(ClientPointer client, ParamSetType& hostParams,  bool synchronous, Result &result)
    : mState(kNoProcess), mClient(client), mProcessParams(hostParams), mContext{mTask}
    {
      
     assert(mClient.get() != nullptr); //right?
      
      if (synchronous)
      {
        result = process();
      }
      else
      {
        auto entry = [](ThreadedTask* owner) { owner->process(); };        
        mProcessParams.template forEachParamType<BufferT, BufferCopy>();
        mProcessParams.template forEachParamType<InputBufferT, BufferCopy>();
        mClient->setParams(mProcessParams);
        mState = kProcessing;
        mThread = std::thread(entry, this);
        result = Result();
      }
    }
    
    Result process()
    {
    
      assert(mClient.get() != nullptr); //right?
    
      mState = kProcessing;
      Result r = mClient->process(mContext);
      mState = kDone;
      
      if (mDetached)
        delete this;
      
      return r;
    }
      
    void cancel(bool detach)
    {
      mTask.cancel();

      mDetached = detach;
      
      if (detach && mThread.joinable())
        mThread.detach();
    }
      
    ProcessState checkProgress(Result& result)
    {
      ProcessState state = mState;
      
      if (state == kDone)
      {
        if (mThread.get_id() != std::thread::id())
          mThread.join();
        
        if (!mTask.cancelled())
        {
          mProcessParams.template forEachParamType<BufferT, BufferCopyBack>();
          result = mResult;
        }
        else
          result = {Result::Status::kCancelled,""};
        
        mProcessParams.template forEachParamType<BufferT, BufferDelete>();
        mState = kNoProcess;
      }
      
      return state;
    }
    
    ParamSetType mProcessParams;
    ProcessState mState;
    std::thread mThread;
      
    Result mResult;
    ClientPointer mClient;
    FluidTask mTask;
    FluidContext mContext;
    bool mDetached = false;
  };
  
  ParamSetType& mHostParams;
  bool mSynchronous = false;
  std::unique_ptr<ThreadedTask> mThreadedTask;
  ClientPointer mClient;
};

} //namespace client
} //namespace fluid
