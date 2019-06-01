#ifndef CUDA_TIMER_H
#define CUDA_TIMER_H

#include <numeric>

#include "Gpu_manager.h"
#include "Module.h"

namespace Support
{
   class Cuda_timer : public ABCS::Module // ABC
   {
      public:
      Cuda_timer(std::string _name, module_type _type);
      ~Cuda_timer();

      virtual int Init() = 0;

      virtual void FreeDeviceMem() { gpuMgr.FreeDeviceMem(GetWholeName()); }
      virtual void FreeDeviceStreams()
      {
         gpuMgr.FreeDeviceStreams(GetWholeName());
      }
      virtual void Cleanup(bool quiet = false) { gpuMgr.Cleanup(quiet); }
      int          GPUAlloc(void** devPtr, size_t size)
      {
         ttlDeviceMem += size;
         return gpuMgr.DeviceAlloc(devPtr, size, GetWholeName());
      };
      int GPUAllocPitch(void** devPtr, size_t* pitch, size_t width, size_t height)
      {
         return gpuMgr.DeviceAllocPitch(devPtr, pitch, width, height, GetWholeName());
      }
      int GPUCreateStream(cudaStream_t* pStream)
      {
         return gpuMgr.CreateStream(pStream, GetWholeName());
      }

      void  ProfileEnable(bool enable) { m_profileEnable = enable; }
      void  ProfileReport();
      float TotalTime();

      typedef uint16_t        DEBUG_MASK;
      static const DEBUG_MASK DEBUG_MASK_NONE       = 0x0000;
      static const DEBUG_MASK DEBUG_MASK_WRITE_CSV  = 0x8000;
      static const DEBUG_MASK DEBUG_MASK_EVERYTHING = 0xffff;

      void SetDebugMask(DEBUG_MASK mask) { m_debugMask = mask; }

      static float SumTime(float info[], uint32_t total)
      {
         float ttl = 0;
         return accumulate(info, info + total, ttl);
      }

      uint8_t GetEventCount() { return m_profileNum; }
      void    SetStream(cudaStream_t pStream) { m_gpuStream = pStream; }

      protected:
      inline void        ProfileEventStart();
      inline cudaError_t ProfileEventStart2();
      inline void        ProfileEvent(const char* desc);
      inline cudaError_t ProfileEvent2(const char* desc);

      void CreateDebugBuffer(size_t size);
      void DebugDump(Complex64* src, uint32_t batch_size, uint32_t beam_cnt, const char* filename);

      private:
      void        ProfileEventEx(const char* desc);
      cudaError_t ProfileEventEx2(const char* desc);

      protected:
      cudaStream_t m_gpuStream;

      // Deprecated - moved to GPU Manager
      uint32_t ttlDeviceMem;
      uint32_t ttlHostMem;

      GPUManager& gpuMgr = GPUManager::Instance();

      // Profile Events
      static const int PROFILE_EVENT_MAX = 25;
      bool             m_profileEnable   = false;
      uint8_t          m_profileNum      = 0;
      const char*      m_profileDesc[PROFILE_EVENT_MAX];
      cudaEvent_t      m_profileEvent[PROFILE_EVENT_MAX];

      // Debug Support
      DEBUG_MASK m_debugMask     = 0;
      Complex64* m_h_debugBuffer = nullptr;
   };

   inline void Cuda_timer::ProfileEvent(const char* desc)
   {
      if(m_profileEnable)
         ProfileEventEx(desc);
   }

   inline cudaError_t Cuda_timer::ProfileEvent2(const char* desc)
   {
      cudaError_t error = cudaSuccess;
      if(m_profileEnable)
         error = (ProfileEventEx2(desc));
      return error;
   }

   inline void Cuda_timer::ProfileEventStart()
   {
      m_profileNum = 0;
      ProfileEvent("Start");
   }

   inline cudaError_t Cuda_timer::ProfileEventStart2()
   {
      m_profileNum = 0;
      return (ProfileEvent2("Start"));
   }

}; // namespace Support

#endif /* CUDA_TIMER_H */