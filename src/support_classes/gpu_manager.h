#ifndef GPU_MANAGER_H
#define GPU_MANAGER_H

#include <cstddef>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <inttypes.h>
#include <map>
#include <mutex>
#include <stdio.h>
#include <string>
#include <vector>

namespace Management
{
   typedef struct
   {
      void*  ptr;
      size_t size;
   } AllocInfo;

   /**
    * \class gpu_manager
    *
    * \brief Manages GPU resources
    *
    *  The gpu_manager tracks resource allocations on all GPUs in the system.
    * Resources include memory, streams, handles, plans, etc. It acts similar to
    * a smart pointer in that the gpu_manager owns the deallocation of each
    * resource, and not the thread that created them. The threads are only
    * required to call the allocation functions, and those allocations are
    * registered by their respective types. Nothing the gpu_manager does is
    * intended to be used in the fast path. Allocations use a mutex to do the
    * bookeeping, so they should not be done for time-critical functions.
    */
   class gpu_manager
   {
      public:
      static gpu_manager& Instance();

      int SetDevice(uint8_t device);

      int DeviceAlloc(void** devPtr, size_t size, std::string name);
      int DeviceAllocPitch(void** devPtr, size_t* pitch, size_t width, size_t height, std::string name);
      int CreateStream(cudaStream_t* pStream, std::string name);

      void FreeDeviceMem(std::string name = "ALL");
      void FreeDeviceMem(const std::string name, void* ptr);
      void FreeDeviceStreams(std::string name = "ALL");
      void FreeStream(std::string name, cudaStream_t s);
      void FreeStreams(void);

      void PrintDeviceInfo(void);
      void PrintResourceUsage(void);
      void Cleanup(bool quiet = false);
      bool IsClean(void);

      private:
      gpu_manager(){};
      ~gpu_manager(){};

      std::mutex alloc_lock;

      std::map<std::pair<std::string, uint8_t>, std::vector<AllocInfo>> dalloc_map;
      std::map<std::pair<std::string, uint8_t>, std::vector<cudaStream_t>> dstream_map;
      std::map<uint8_t, uint64_t> d_ttl_alloc;
   };

}; // namespace Management

#endif /* GPU_MANAGER_H */