#include "Cuda_timer.h"

#include "GpuUtils.h"

namespace Support
{
   const char* const ABSCS::Module::typeStr[3] = {"transmitter", "receiver",
                                                  "both"};

   Cuda_timer::Cuda_timer(std::string _name, module_type _type) :
      Module(_name, _type)
   {
      // Create events for profile purposes
      for(int i = 0; i < PROFILE_EVENT_MAX; i++)
      {
         checkCuda(cudaEventCreate(&m_profileEvent[i]));
      }
   }

   Cuda_timer::~Cuda_timer()
   {
      for(int i = 0; i < PROFILE_EVENT_MAX; i++)
      {
         checkCuda(cudaEventDestroy(m_profileEvent[i]));
      }

      if(m_h_debugBuffer != nullptr)
      {
         checkCuda(cudaFreeHost(m_h_debugBuffer));
         m_h_debugBuffer = nullptr;
      }

      FreeDeviceMem();
   }

   //! \brief Record time of event completion
   //!
   void Cuda_timer::ProfileEventEx(const char* desc)
   {
      if(m_profileNum < PROFILE_EVENT_MAX)
      {
         checkCuda(cudaEventRecord(m_profileEvent[m_profileNum], m_gpuStream));
         m_profileDesc[m_profileNum++] = desc;
      }
   }

   cudaError_t Cuda_timer::ProfileEventEx2(const char* desc)
   {
      cudaError_t error;
      if(m_profileNum < PROFILE_EVENT_MAX)
      {
         error = cudaEventRecord(m_profileEvent[m_profileNum], m_gpuStream);
         if(error != 0)
            TB_ERROR("ERROR IS %d, in stream %llu", error, m_gpuStream);
         checkCuda(error);
         m_profileDesc[m_profileNum++] = desc;
      }
      return error;
   }

   //! \brief Diagnostic routine to obtain the GPU time needed to complete
   //!  the specified event.
   //!
   void Cuda_timer::ProfileReport()
   {
      // Need to wait for last event to finish or you will FATAL with CUDA
      // runtime error
      checkCuda(cudaEventSynchronize(m_profileEvent[m_profileNum - 1]));
      float ms;
      for(int i = 1; i < m_profileNum; i++)
      {
         checkCuda(cudaEventElapsedTime(&ms, m_profileEvent[i - 1], m_profileEvent[i]));
         TB_INFO("% 20s: %6.5f ms", m_profileDesc[i], ms);
      }
   }

   //! \brief Diagnostic routine to obtain the total GPU time needed to complete
   //!  the most recently scheduled work.
   //!
   float Cuda_timer::TotalTime()
   {
      float ms = 0.0;
      if(m_profileNum > 0)
      {
         checkCuda(cudaEventElapsedTime(&ms, m_profileEvent[0],
                                        m_profileEvent[m_profileNum - 1]));
      }
      return ms;
   }

   //! \brief Diagnostic routine to create a host memory buffer for the purpose
   //!  of saving internal memory state information
   //!
   void Cuda_timer::CreateDebugBuffer(size_t size)
   {
      checkCuda(cudaHostAlloc(&m_h_debugBuffer, size, cudaHostAllocDefault));
   }

   //! \brief Diagnostic routine to dump a block of memory to a CSV file
   void Cuda_timer::DebugDump(Complex64* src, uint32_t batch_size, uint32_t beam_cnt, const char* filename)
   {
      checkCuda(cudaDeviceSynchronize());

      checkCuda(cudaMemcpy(m_h_debugBuffer, src, batch_size * beam_cnt * sizeof(Complex64),
                           cudaMemcpyDeviceToHost));

      if(m_debugMask & DEBUG_MASK_WRITE_CSV)
      {
         FILE* fp;

         fp = fopen(filename, "w");
         if(fp == NULL)
         {
            TB_ERROR("Can't open %s", filename);
            return;
         }

         for(uint32_t g = 0; g < batch_size; g++)
         {
            for(uint32_t k = 0; k < beam_cnt; k++)
            {
               Complex64* ptr = &m_h_debugBuffer[k * batch_size + g];
               (ptr->i != 0.0) ? fprintf(fp, "%4.2f,", ptr->i)
                               : fprintf(fp, "   0,");
               (ptr->q != 0.0) ? fprintf(fp, "%4.2f, ", ptr->q)
                               : fprintf(fp, "   0, ");
            }
            fprintf(fp, "\n");
         }
         fclose(fp);
      }
   }

}; // namespace Support