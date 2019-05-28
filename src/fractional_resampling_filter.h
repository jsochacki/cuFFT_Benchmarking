#ifndef FRACTIONAL_RESAMPLING_FILTER_H_
#define FRACTIONAL_RESAMPLING_FILTER_H_

// required for compile if file is a .cpp file
//#include <cuda_runtime.h>

// required for types used
#include <stdint.h>

typedef struct
{
   float i;
   float q;
} complex64;

// TODO just for development, can delete
const float MY_PI = 3.14159265358979323846;

#define CU_FUNC_CALLED_FROM_CPP extern "C"

#define R_MAC 1024
#define K_MAC 31
#define POLYNOMIAL_ORDER_MAC 5
// POLYNOMIAL_ORDER + 1
#define NCOLUMNS_MAC 6
#define TPS_MAC 32U
#define SPB_MAC 24U

#define NCO_Size_MAC 1UL << 41
#define NCO_Bit_Truncation_MAC 1UL << 31

// Filter design parameters
extern const uint16_t R;
extern const uint16_t K;
extern const uint16_t POLYNOMIAL_ORDER;

extern uint32_t NUMBER_OF_INPUT_SAMPLES;
extern uint32_t NUMBER_OF_OUTPUT_SAMPLES;

// The NCO and all associated NCO data needs to be uint64
extern uint64_t NCO_Size;
extern uint64_t NCO_Bit_Truncation;
extern uint64_t NCO;

extern uint32_t input_counter;
extern uint32_t output_counter;

extern uint64_t NCO_Step_Size;

extern uint64_t iteration;

extern int64_t PPM_Offset;

// Load in coefficients
extern __constant__ float d_coeffs[K_MAC * (POLYNOMIAL_ORDER_MAC + 1)];

// TODO, just to verify proper read in
extern const float coeffs[K_MAC * (POLYNOMIAL_ORDER_MAC + 1)];

__host__ cudaError_t initialize_frf_coefficients_in_constant_memory(float*);

__host__ uint32_t complex_test_signal_generator(uint32_t, uint32_t, float, complex64**);

__host__ uint32_t calculate_FRF_output_length(uint32_t, uint32_t, uint64_t, uint64_t, uint64_t);

__host__ uint64_t calculate_FRF_NCO_step_size(uint64_t, int64_t);

__global__ void create_complex_input_vector(complex64* __restrict__,
                                            complex64* __restrict__,
                                            uint32_t,
                                            uint16_t,
                                            complex64* __restrict__);

__global__ void complex_fractional_resampling_filter(const complex64* __restrict__,
                                                     uint32_t,
                                                     uint64_t,
                                                     uint64_t,
                                                     uint32_t,
                                                     complex64* __restrict__);

__host__ int post_run_NCO_update(uint32_t);
__global__ void
    post_run_complex_state_update(uint32_t, uint16_t, complex64* __restrict__, complex64* __restrict__);

__host__ void initialize_state(complex64*, uint16_t);

// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
inline cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
   if(result != cudaSuccess)
   {
      fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
      assert(result == cudaSuccess);
   }
#endif
   return result;
}

#endif /* FRACTIONAL_RESAMPLING_FILTER_CUH_ */
