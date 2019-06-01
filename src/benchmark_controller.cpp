// includes, system
#include <cuFFT_functions.h>
#include <gpu_manager.h>

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
   Management::gpu_manager& local_gpu_manager = Management::gpu_manager::Instance();
   local_gpu_manager.PrintDeviceInfo();
   runTest(argc, argv);
}