// Includes just used in dev and debug
#include <iostream> //Contains cout, etc..
#include <stdlib.h> //Contains exit, and EXIT_FAILURE
#include <fstream> //Contains ofstream, etc..

const unsigned int NUMBER_OF_TEST_VECTORS = 10;

#include "fractional_resampling_filter.h"

//Filter design parameters
const uint16_t R = R_MAC;
const uint16_t K = K_MAC;
const uint16_t POLYNOMIAL_ORDER = POLYNOMIAL_ORDER_MAC;

//For unit test only
uint32_t NUMBER_OF_INPUT_SAMPLES = 3200UL;
uint32_t NUMBER_OF_OUTPUT_SAMPLES = 90000UL;  //81903 should be the largest

// The NCO and all associated NCO data needs to be uint64
uint64_t NCO_Size = NCO_Size_MAC;
uint64_t NCO_Bit_Truncation = NCO_Bit_Truncation_MAC;
uint64_t NCO = 0ULL;

uint32_t input_counter = 0;

uint64_t NCO_Step_Size = 0ULL;

uint64_t iteration = 1;

uint32_t N_Outputs;

int64_t PPM_Offset = 0LL;

int main(void)
{
   float local_coeffs[K_MAC * (POLYNOMIAL_ORDER_MAC + 1)] = {0.0f};

   float pretest = 10.0f;
   float *d_test;
   float posttest = 0.0f;

   checkCuda( cudaMalloc((void **)&d_test,
                        sizeof(float) * 1));

      checkCuda( cudaMemcpy(d_complex_reverse_state, reverse_state,
   (sizeof(complex64)) * (static_cast<unsigned long int>(K) - 1UL),
   cudaMemcpyHostToDevice));

   checkCuda( cudaMemcpyFromSymbol(d_coeffs,
               local_coeffs,
               (K_MAC * (POLYNOMIAL_ORDER_MAC + 1)) * sizeof(float),
               0,
               cudaMemcpyDeviceToHost));

   for(uint16_t index = 0; index < (K_MAC * (POLYNOMIAL_ORDER_MAC + 1)); ++index)
   {
      std::cout << "local_coeffs[" << index << "] = " << local_coeffs[index] << std::endl;
      std::cout << "coeffs[" << index << "] = " << coeffs[index] << std::endl;
   }

}
