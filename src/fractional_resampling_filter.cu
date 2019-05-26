#include "fractional_resampling_filter.h"

//#include <iostream> //Contains cout, etc..

//Three notes
//1 since you are never calling any of these from a .cpp file at this point
//you do not need extern "C" linkage
//2 by including the definition of the __constant__ here and only here and using a setting
//function you do not need to worry about linkage of separate translational units
//by the nvcc compiler so can ignore the "separate compilation" compile flow
//Again , since the calling file is a .cu file you do not need to wrap kernel calls
//if the calling file were a .cpp file you would need to wrap the kernel calls
//as the <<<>>> is not compilable by g++, implement extern "C" linkage to prevent mangling
//and #include "cuda_runtime.h" in your .cpp file for declarations

__constant__ float d_rev_coeffs[K_MAC * (POLYNOMIAL_ORDER_MAC + 1)];

__host__ cudaError_t initialize_frf_coefficients_in_constant_memory(float *coefficients_vector)
{
   return cudaMemcpyToSymbol(d_rev_coeffs,
                             coefficients_vector,
                             (K_MAC * (POLYNOMIAL_ORDER_MAC + 1)) * sizeof(float),
                             0,
                             cudaMemcpyHostToDevice);
}

__host__ uint32_t complex_test_signal_generator(uint32_t Oversample_Rate,
                          uint32_t Number_Of_Cycles,
                          float Digital_Frequency,
                          complex64 **output)
{
    *output = NULL;
    *output = (complex64 *) calloc(Number_Of_Cycles * Oversample_Rate, sizeof *output);
    if(*output == NULL)
    {
        return 0;
    }

    for(uint32_t n = 0; n < (Number_Of_Cycles * Oversample_Rate); n++)
    {
        (*output)[n].i = std::cos(2 * MY_PI * Digital_Frequency
                              * static_cast<float>(n));
        (*output)[n].q = std::sin(2 * MY_PI * Digital_Frequency
                                      * static_cast<float>(n));
    }
    return (Number_Of_Cycles * Oversample_Rate);
}

__host__ uint32_t calculate_FRF_output_length(uint32_t NUMBER_OF_INPUT_SAMPLES,
                                          uint32_t input_counter,
                                          uint64_t NCO,
                                          uint64_t NCO_Size,
                                          uint64_t NCO_Step_Size)
{
    // So length(y) - input counter is the number of inputs you have to process
    // since y is the length of the input signal and input_counter is equal to
    // how many immediate shifts you will do when you start the filter based on
    // the previous run
    // you multiply this by the NCO_Size and subtract 1 to get the number of NCO
    // counts you have to do to complete all input processing (the -1 is because
    // the NCO starts at 0 and the first count puts you at 1 already, think of
    // other stuff in there for all the other runs
    return (std::floor((
            static_cast<double>(((static_cast<uint64_t>(NUMBER_OF_INPUT_SAMPLES)
                                  - static_cast<uint64_t>(input_counter))
                                 * NCO_Size)
                                - NCO - 1ULL)
                        / static_cast<double>(NCO_Step_Size))
                       + 1 ));
}

__host__ uint64_t calculate_FRF_NCO_step_size(uint64_t NCO_Size, int64_t PPM_Offset)
{
    return (static_cast<uint64_t>(
               static_cast<int64_t>(NCO_Size) +
               ((static_cast<int64_t>(NCO_Size) * PPM_Offset)
                / (1000000LL))    ));
}

//Create the input vector for the filter
__global__ void create_complex_input_vector(complex64* __restrict__ input_signal,
                                            complex64* __restrict__ reverse_state,
                                            uint32_t NUMBER_OF_INPUT_SAMPLES,
                                            uint16_t K,
                                            complex64* __restrict__ input_vector)
{
   uint32_t idx = (static_cast<uint32_t>(blockIdx.x)
                       * static_cast<uint32_t>(blockDim.x))
                      + static_cast<uint32_t>(threadIdx.x);

   if(idx < (K - 1))
   {
       // it is K - 2 because reverse_stat is K - 1 long
       // and we want to start one lower than that to auto shift the input
       input_vector[idx].i = reverse_state[idx].i;
       input_vector[idx].q = reverse_state[idx].q;
       input_vector[idx + (K - 1)].i = input_signal[idx].i;
       input_vector[idx + (K - 1)].q = input_signal[idx].q;
   }
   //More secure but unnecessary technically
   //else if((idx >= (K - 1)) && (idx < NUMBER_OF_INPUT_SAMPLES))
   else if(idx < NUMBER_OF_INPUT_SAMPLES)
   {
       input_vector[idx + (K - 1)].i = input_signal[idx].i;
       input_vector[idx + (K - 1)].q = input_signal[idx].q;
   }
}

__global__ void complex_fractional_resampling_filter(const complex64* __restrict__ input_vector,
                                                     uint32_t N_Outputs,
                                                     uint64_t NCO,
                                                     uint64_t NCO_Step_Size,
                                                     uint32_t input_counter,
                                                     complex64* __restrict__ output_signal)
{
   uint32_t idx = (static_cast<uint32_t>(blockIdx.x)
                       * static_cast<uint32_t>(blockDim.x))
                      + static_cast<uint32_t>(threadIdx.x);

   if(idx < N_Outputs)
   {
      ////K_MAC * NCOLUMNS_MAC == 186
//      extern __shared__ float s_coeffs[];

      float NCO_state =
                 (static_cast<float>( ((NCO + (static_cast<uint64_t>(idx) * NCO_Step_Size)) % static_cast<uint64_t>(NCO_Size_MAC))) /
                  static_cast<float>(NCO_Bit_Truncation_MAC)) /
                      static_cast<float>(R_MAC);

      uint32_t in_index_base = input_counter
                       + static_cast<uint32_t>((NCO + (static_cast<uint64_t>(idx) * NCO_Step_Size))
                                                / (NCO_Size_MAC));

//      if(threadIdx.x < (K_MAC * NCOLUMNS_MAC))
//      {
//         s_coeffs[threadIdx.x] = d_rev_coeffs[threadIdx.x];
//      }

      float ival = 0.0f;
      float qval = 0.0f;
      float NCO_statepow2, NCO_statepow4;

      NCO_statepow2 = NCO_state * NCO_state;
      NCO_statepow4 = NCO_statepow2 * NCO_statepow2;

//      __syncthreads();

      for(unsigned int k = 0; k < K_MAC; k++)
      {
         float H;

//         H = ((s_coeffs[(k * NCOLUMNS_MAC) + 0] * NCO_state) + s_coeffs[(k * NCOLUMNS_MAC) + 1]) * NCO_statepow4 +
//             ((s_coeffs[(k * NCOLUMNS_MAC) + 2] * NCO_state) + s_coeffs[(k * NCOLUMNS_MAC) + 3]) * NCO_statepow2 +
//              (s_coeffs[(k * NCOLUMNS_MAC) + 4] * NCO_state) + s_coeffs[(k * NCOLUMNS_MAC) + 5];

         H = ((d_rev_coeffs[(k * NCOLUMNS_MAC) + 0] * NCO_state) + d_rev_coeffs[(k * NCOLUMNS_MAC) + 1]) * NCO_statepow4 +
             ((d_rev_coeffs[(k * NCOLUMNS_MAC) + 2] * NCO_state) + d_rev_coeffs[(k * NCOLUMNS_MAC) + 3]) * NCO_statepow2 +
              (d_rev_coeffs[(k * NCOLUMNS_MAC) + 4] * NCO_state) + d_rev_coeffs[(k * NCOLUMNS_MAC) + 5];

         ival += H * input_vector[in_index_base + k].i;
         qval += H * input_vector[in_index_base + k].q;
      }
      //transfer data to memory wholly and not by parts
      output_signal[idx].i = ival;
      output_signal[idx].q = qval;
   }
}

//Post run host housekeeping
__host__ int post_run_NCO_update(uint32_t N_Outputs)
{
    uint64_t TNCOVAL;

    iteration += 1;
    //since the next run starts with the previous NCO state we need to mover
    //the NCO state and input counter forward to their next states
    TNCOVAL = (NCO + (static_cast<uint64_t>(N_Outputs) * NCO_Step_Size));
    input_counter = input_counter + static_cast<uint32_t>(TNCOVAL / NCO_Size); //We want the truncated result, can use floor if you feel better that way
    NCO = (TNCOVAL % NCO_Size);
    input_counter = input_counter - NUMBER_OF_INPUT_SAMPLES;

    return 0;
}

//Post run GPU housekeeping
__global__ void post_run_complex_state_update(uint32_t NUMBER_OF_INPUT_SAMPLES,
                                      uint16_t K,
                                      complex64* __restrict__ reverse_state,
                                      complex64* __restrict__ input_vector)
{

   uint32_t idx = static_cast<uint32_t>(threadIdx.x);

   //Move filter state to end state
   if(idx < (K - 1))
   {
       //auto drop the last value in the filter state
       // to keep it NUMBER_OF_INPUT_SAMPLES => NUMBER_OF_INPUT_SAMPLES - 1
       //and state length must go to K
       reverse_state[idx].i =
           input_vector[NUMBER_OF_INPUT_SAMPLES + idx].i;
       reverse_state[idx].q =
           input_vector[NUMBER_OF_INPUT_SAMPLES + idx].q;
       //Input_Vector_Length = NUMBER_OF_INPUT_SAMPLES + K - 1;
       //((Input_Vector_Length - 1) - (K - 1)) + k
   }
}

__host__ void initialize_state(complex64 *reverse_state, uint16_t K)
{
   for(uint16_t index = 0; index < (K - 1); index++)
   {
      reverse_state[index].i = 0.0f;
      reverse_state[index].q = 0.0f;
   }
}
