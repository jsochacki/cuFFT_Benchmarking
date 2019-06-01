#include "gpu_manager.h"

namespace Management
{
   gpu_manager& gpu_manager::Instance()
   {
      static gpu_manager instance;
      return instance;
   }

   //! \brief Sets the GPU device ID for the calling thread
   //! //-in device - Device ID to set to (0-15)
   //!
   //! //-out - non-zero value indicates fatal error
   int gpu_manager::SetDevice(uint8_t device)
   {
      cudaError_t err = cudaSetDevice(device);
      if(err != cudaSuccess)
      {
         printf("FATAL ERROR: Failed to set CUDA device to %u", device);
         return -1;
      }

      return 0;
   }

   //! \brief Allocate linear memory on the GPU
   //! //-in name - Name of the calling object for association with allocation
   //! //-in size - Size of allocation in bytes
   //!
   //! //-out devPtr - Pointer address where memory is stored
   //! //-out - non-zero value indicates fatal error
   int gpu_manager::DeviceAlloc(void** devPtr, size_t size, std::string name)
   {
      cudaError_t err;

      err = cudaMalloc(devPtr, size);
      if(err != cudaSuccess)
      {
         printf("FATAL ERROR: cudaMalloc Error: %s from %s",
                cudaGetErrorString(err), name.c_str());
         return -1;
      }

      std::unique_lock<std::mutex> local_lock(alloc_lock);

      int device;
      if(cudaGetDevice(&device) != cudaSuccess)
      {
         printf("FATAL ERROR: Failed to get CUDA device ID for name %s", name.c_str());
         return -1;
      }

      dalloc_map[{name, device}].push_back({*devPtr, size});
      d_ttl_alloc[device] += size;
   }

   //! \brief Allocate pitched memory on the GPU
   //! //-in name - Name of the calling object for association with allocation
   //! //-in width - Requested width of allocat in bytes
   //! //-in height - Requested height of allocat in bytes
   //!
   //! //-out devPtr - Pointer address where memory is stored
   //! //-out pitch - Actual width of allocation in bytes
   //! //-out - non-zero value indicates fatal error
   int gpu_manager::DeviceAllocPitch(void** devPtr, size_t* pitch, size_t width, size_t height, std::string name)
   {
      cudaError_t err;

      err = cudaMallocPitch(devPtr, pitch, width, height);
      if(err != cudaSuccess)
      {
         printf("FATAL ERROR: cudaMallocPitch Error: %s from %s",
                cudaGetErrorString(err), name.c_str());
         return -1;
      }

      std::unique_lock<std::mutex> local_lock(alloc_lock);

      int device;
      if(cudaGetDevice(&device) != cudaSuccess)
      {
         printf("FATAL ERROR: Failed to get CUDA device ID for %s", name.c_str());
         return -1;
      }

      dalloc_map[{name, device}].push_back({*devPtr, *pitch * height});
      d_ttl_alloc[device] += *pitch * height;
   }

   //! \brief Create a stream on the GPU
   //! //-in name - Name of calling module
   //!
   //! //-out pStream - Pointer address where the stream is stored
   //! //-out - non-zero value indicates fatal error
   int gpu_manager::CreateStream(cudaStream_t* pStream, std::string name)
   {
      cudaError_t err = cudaStreamCreate(pStream);
      if(err != cudaSuccess)
      {
         printf("FATAL ERROR: cudaStreamCreate Error: %s from %s",
                cudaGetErrorString(err), name.c_str());
         return -1;
      }

      std::unique_lock<std::mutex> local_lock(alloc_lock);
      int                          device;
      if(cudaGetDevice(&device) != cudaSuccess)
      {
         printf("FATAL ERROR: Failed to get CUDA device ID for %s", name.c_str());
         return -1;
      }

      dstream_map[{name, device}].push_back(*pStream);
   }

   //! \brief Free device memory from a single user
   //! //-in name - of module or ALL to free all memory the gpu_manager controls
   //!
   //! //-out - none
   void gpu_manager::FreeDeviceMem(std::string name)
   {
      std::unique_lock<std::mutex> local_lock(alloc_lock);
      for(auto it = dalloc_map.cbegin(); it != dalloc_map.cend();)
      {
         if((name == "ALL") || (it->first.first == name))
         {
            uint64_t cnt    = 0;
            uint8_t  device = it->first.second;
            for(auto& v : it->second)
            {
               cudaSetDevice(device);
               cudaFree(v.ptr);
               cnt += v.size;
               d_ttl_alloc[device] -= v.size;
            }

            dalloc_map.erase(it++);

            if(d_ttl_alloc[device] == 0)
            {
               d_ttl_alloc.erase(device);
            }

            if(name == "ALL")
            {
               printf(
                   "DEBUG: Freed all device memory from GPU manager (%.2fMB "
                   "freed)",
                   cnt / 1e6);
            }
            else
            {
               printf(
                   "DEBUG: Freed all device memory for %s from GPU manager "
                   "(%.2fMB "
                   "freed)",
                   name.c_str(), cnt / 1e6);
            }
         }
         else
         {
            ++it;
         }
      }
   }

   //! \brief Free a single buffer on a device
   //! //-in name - of module
   //! //-in ptr - pointer to free
   //!
   //! //-out - none
   void gpu_manager::FreeDeviceMem(const std::string name, void* ptr)
   {
      std::unique_lock<std::mutex> local_lock(alloc_lock);
      for(auto it = dalloc_map.cbegin(); it != dalloc_map.cend();)
      {
         if(it->first.first == name)
         {
            uint8_t device = it->first.second;
            for(auto& v : it->second)
            {
               if(v.ptr == ptr)
               {
                  cudaSetDevice(device);
                  cudaFree(v.ptr);
                  d_ttl_alloc[device] -= v.size;
                  return;
               }
            }
         }
      }

      printf("NON-FATAL ERROR: Pointer %p not found with name %s", ptr, name.c_str());
   }

   //! \brief Free device streams allocated on GPU
   //! //-in name - of module to free all streams on
   //!
   //! //-out - none
   void gpu_manager::FreeDeviceStreams(std::string name)
   {
      if(name == "ALL")
      {
         FreeStreams();
         printf("DEBUG: Freed all GPU streams.");
      }
      else
      {
         std::unique_lock<std::mutex> local_lock(alloc_lock);
         for(auto it = dstream_map.cbegin(); it != dstream_map.cend();)
         {
            if(it->first.first == name)
            {
               cudaSetDevice(it->first.second);
               for(auto& v : it->second)
               {
                  cudaStreamDestroy(v);
               }
               dstream_map.erase(it++);

               printf("DEBUG: Freed all streams for %s from GPU manager", name.c_str());
            }
            else
            {
               ++it;
            }
         }
      }
   }

   //! \brief Free a single stream
   //! //-in name - of module to free stream s on
   //! //-in s - stream
   //!
   //! //-out - none
   void gpu_manager::FreeStream(std::string name, cudaStream_t s)
   {
      std::unique_lock<std::mutex> local_lock(alloc_lock);
      for(auto it = dstream_map.cbegin(); it != dstream_map.cend(); it++)
      {
         if(it->first.first == name)
         {
            for(auto& v : it->second)
            {
               if(v == s)
               {
                  cudaSetDevice(it->first.second);
                  cudaStreamDestroy(v);
                  return;
               }
            }
         }
      }

      printf("NON-FATAL ERROR: Couldn't find stream to delete for name %s", name.c_str());
   }

   //! \brief Free all streams allocated on GPU
   //! //-note frees all streams on gpus that have streams in the dstream_map
   //!
   //! //-out - none
   void gpu_manager::FreeStreams(void)
   {
      std::unique_lock<std::mutex> local_lock(alloc_lock);
      for(auto it = dstream_map.cbegin(); it != dstream_map.cend();)
      {
         cudaSetDevice(it->first.second);
         for(auto& v : it->second)
         {
            cudaStreamDestroy(v);
         }
         dstream_map.erase(it++);
      }
   }

   //! \brief Prints information about each GPU device
   //!
   //! //-out - none
   void gpu_manager::PrintDeviceInfo(void)
   {
      int numGpus = 0;
      cudaGetDeviceCount(&numGpus);

      printf("*** %d GPU devices found ***", numGpus);
      for(uint32_t i = 0; i < numGpus; i++)
      {
         cudaDeviceProp p;
         cudaGetDeviceProperties(&p, i);
         printf("GPU Device %u", i);
         printf("   Name: %s", p.name);
         printf("   Compute Capability: %u.%u", p.major, p.minor);
         printf("   Mem Clock (MHz): %.0f", p.memoryClockRate / 1e3);
         printf("   Mem Bus Width (bits): %d", p.memoryBusWidth);
         printf("   Shared Memory Per Block: %zu bytes", p.sharedMemPerBlock);
         printf("   Peak Memory Bandwidth (GB/s): %.0f",
                2.0 * p.memoryClockRate * (p.memoryBusWidth / 8) / 1.0e6);
         printf("   Total Mem (GB): %.2f", p.totalGlobalMem / 1e9);
         printf("   SM Count: %d", p.multiProcessorCount);
         printf("   SM Clock (MHz): %.0f", p.clockRate / 1e3);
         printf("   ECC Enabled: %d", p.ECCEnabled);
      }
   }

   //! \brief Prints total memory useage and module allocations
   //!
   void gpu_manager::PrintResourceUsage(void)
   {
      printf("NOTICE: Total memory allocation per GPU:");
      for(auto& i : d_ttl_alloc)
      {
         printf("NOTICE: Device %2u: %.2fMB", i.first, i.second / 1e6);
      }

      printf("NOTICE: Memory allocation per core:");
      for(auto& i : dalloc_map)
      {
         uint64_t ttl = 0;
         uint32_t cnt = 0;
         for(auto& n : i.second)
         {
            ttl += n.size;
            cnt++;
         }
         printf("NOTICE: Object %s on device %2u: %.2fMB over %u allocations",
                i.first.first.c_str(), i.first.second, ttl / 1e6, cnt);
      }

      printf("NOTICE: Total streams used:");
      for(auto& i : dstream_map)
      {
         printf("NOTICE: Object %s on device %2u: %zu streams",
                i.first.first.c_str(), i.first.second, i.second.size());
      }
   }

   //! \brief Clean up all resources
   //!
   //! //-out - non-zero value indicates fatal error
   void gpu_manager::Cleanup(bool quiet)
   {
      if(!quiet)
      {
         printf("Freeing all device memory");
      }
      FreeDeviceMem();
      FreeStreams();
   }

   //! \brief Check if all resources have been cleaned up
   //!
   //! //-out - true if no gpu resources are allocated at the moment
   bool gpu_manager::IsClean(void)
   {
      std::unique_lock<std::mutex> local_lock(alloc_lock);
      if(dalloc_map.empty() && dstream_map.empty() && d_ttl_alloc.empty())
      {
         return true;
      }
      printf("NOTICE: GPU Manager is not clean.");
      PrintResourceUsage();
      return false;
   }

}; // namespace Management