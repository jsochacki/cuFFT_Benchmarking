// includes, system
#include <Gpu_manager.h>
#include <cuFFT_functions.h>

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
   Management::Gpu_manager& local_gpu_manager = Management::Gpu_manager::Instance();
   local_gpu_manager.PrintDeviceInfo();
   runTest(argc, argv);
}