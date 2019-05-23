// Includes just used in dev and debug
#include <iostream> //Contains cout, etc..
#include <stdlib.h> //Contains exit, and EXIT_FAILURE
#include <fstream> //Contains ofstream, etc..

#include "test_vector_utilities.hpp"

const unsigned int NUMBER_OF_TEST_VECTORS = 10;

#include "GpuFractionalResamplingFilter.h"

//Filter design parameters
const uint16_t R = R_MAC;
const uint16_t K = K_MAC;
const uint16_t POLYNOMIAL_ORDER = POLYNOMIAL_ORDER_MAC;

//For unit test only
uint32_t NUMBER_OF_INPUT_SAMPLES = 3200UL;
uint32_t NUMBER_OF_OUTPUT_SAMPLES = 90000UL;

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
   //unit test set up section
   std::string test_vector_path("../src/");
   std::vector<std::string> test_vector_names(10, "frf_test_vector_");
   std::string current_test_vector_file_name, coefficients_file_name;

   std::vector<complex64*> test_vectors(NUMBER_OF_TEST_VECTORS, NULL);

   std::vector<uint32_tvi*> test_vector_information(NUMBER_OF_TEST_VECTORS);

   std::string frf_coefficients_name("fractional_resampling_filter_coefficients.txt");
   uint16_fci *coefficients_information = new uint16_fci();
   float *coefficients_vector;

   for(unsigned int n = 0; n < NUMBER_OF_TEST_VECTORS; n++)
   {
      test_vector_names[n].append(std::to_string(n + 1) + std::string(".txt"));
   }

   for(std::string str : test_vector_names)
          std::cout<< str <<std::endl;

   for(unsigned int n = 0; n < NUMBER_OF_TEST_VECTORS; n++)
   {
      test_vector_information[n] = new uint32_tvi();
      current_test_vector_file_name = test_vector_path + test_vector_names[n];

      if(!read_in_complex_test_vector<complex64, uint32_t>(current_test_vector_file_name,
                                                           test_vector_information[n],
                                                           &test_vectors[n]))
      {
          std::cout << "fatal error" << std::endl;
      }
   }

   coefficients_file_name = test_vector_path + frf_coefficients_name;

   if(!read_in_filter_coefficients<float, uint16_t>(coefficients_file_name,
                                                   coefficients_information,
                                                   &coefficients_vector))
   {
      std::cout << "fatal error" << std::endl;
   }

   checkCuda( initialize_frf_coefficients_in_constant_memory(coefficients_vector));

   //Generate input for testing purposes
   complex64 *complex_input_signal;

   uint32_t N_Inputs, Oversample_Rate, Number_Of_Cycles;
   float Digital_Frequency;

   N_Inputs = 0UL;
   Oversample_Rate = 32UL;
   Number_Of_Cycles = 100UL;
   Digital_Frequency =  static_cast<float>(1) / static_cast<float>(Oversample_Rate);

   N_Inputs = complex_test_signal_generator(Oversample_Rate,
                                        Number_Of_Cycles,
                                        Digital_Frequency,
                                        &complex_input_signal);

    // Start filter function section

   //host variables
    complex64 *complex_input_vector, *complex_output_signal;
    uint32_t complex_float_data_size = sizeof(complex64);
    uint32_t complex_input_vector_length =
             static_cast<uint32_t>(
                (NUMBER_OF_INPUT_SAMPLES
                 + static_cast<uint32_t>(K)
                 - 1UL)            );

    //device variables
    complex64 *d_complex_input_signal, *d_complex_input_vector;
    complex64 *d_complex_output_signal, *d_complex_reverse_state;

    if(NUMBER_OF_INPUT_SAMPLES != N_Inputs)
    {
        exit(0);
    }

    cudaEvent_t start, stop;
    checkCuda( cudaEventCreate(&start));
    checkCuda( cudaEventCreate(&stop));

    checkCuda( cudaMalloc((void **)&d_complex_input_signal,
                          complex_float_data_size * NUMBER_OF_INPUT_SAMPLES));
    checkCuda( cudaMalloc((void **)&d_complex_input_vector,
                          complex_float_data_size * complex_input_vector_length));
    checkCuda( cudaMalloc((void **)&d_complex_output_signal,
                          complex_float_data_size * NUMBER_OF_OUTPUT_SAMPLES));
    checkCuda( cudaMalloc((void **)&d_complex_reverse_state,
                          complex_float_data_size * (K - 1)));

    complex_input_vector = (complex64 *) malloc(complex_float_data_size * complex_input_vector_length); //new float[NUMBER_OF_INPUT_SAMPLES + K - 1]();
    complex_output_signal = (complex64 *) malloc(complex_float_data_size * NUMBER_OF_OUTPUT_SAMPLES); //new float[NUMBER_OF_OUTPUT_SAMPLES]();

    //So you can initialize the device copy to zero
    complex64 *reverse_state;
    reverse_state = (complex64 *) calloc(static_cast<unsigned long int>(K) - 1UL,
                                         sizeof(complex64));
    initialize_state(reverse_state, K);
    ///
    //Start loop right here

    std::vector<std::ofstream*> output_files(10);
    for(unsigned int nvec = 0; nvec < NUMBER_OF_TEST_VECTORS; nvec++)
    {
       output_files[nvec] = new std::ofstream();
       output_files[nvec]->open(static_cast<std::string>("result_") + test_vector_names[nvec], std::ios::out);

       checkCuda( cudaMemcpy(d_complex_reverse_state, reverse_state,
                             (sizeof(complex64)) * (static_cast<unsigned long int>(K) - 1UL),
                             cudaMemcpyHostToDevice));

       PPM_Offset = test_vector_information[nvec]->PPM;
       NCO_Step_Size = calculate_FRF_NCO_step_size(NCO_Size, PPM_Offset);

       //reinitialize the filter
       NCO = 0ULL;
       input_counter = 0;
       iteration = 1;

       uint32_t sum_N_Outputs, start_index;
       sum_N_Outputs = 0;
       start_index = 0;
       float accumulator;
       accumulator = 0.0f;

       for(uint16_t nnn = 0; nnn < 8; nnn++)
       {

           //Calculate the number of outputs we will get based on current filter state
           N_Outputs = calculate_FRF_output_length(NUMBER_OF_INPUT_SAMPLES,
                                                   input_counter,
                                                   NCO,
                                                   NCO_Size,
                                                   NCO_Step_Size);

           if(N_Outputs > NUMBER_OF_OUTPUT_SAMPLES)
           {
              std::cout << "error, The PPM offset supplied generates more output "
                        << "samples than we have allowed for in the output buffer.\n"
                        << " Please correct this.  Exiting now.\n" << std::endl;
              break;
           }
           sum_N_Outputs += N_Outputs;

           checkCuda( cudaEventRecord(start));
   //        checkCuda( cudaMemcpy(d_input_signal, input_signal, float_data_size * NUMBER_OF_INPUT_SAMPLES, cudaMemcpyHostToDevice));
           checkCuda( cudaMemcpy(d_complex_input_signal, complex_input_signal, complex_float_data_size * NUMBER_OF_INPUT_SAMPLES, cudaMemcpyHostToDevice));
           checkCuda( cudaEventRecord(stop));

           checkCuda( cudaEventSynchronize(stop));
           float milliseconds_host_to_device = 0;
           checkCuda( cudaEventElapsedTime(&milliseconds_host_to_device, start, stop));

           checkCuda( cudaEventRecord(start));

           create_complex_input_vector<<<196,1024>>>(d_complex_input_signal,
                                                     d_complex_reverse_state,
                                                     NUMBER_OF_INPUT_SAMPLES,
                                                     K,
                                                     d_complex_input_vector);

           checkCuda( cudaEventRecord(stop));

           checkCuda( cudaEventSynchronize(stop));
           float milliseconds_create_input_vector = 0;
           checkCuda( cudaEventElapsedTime(&milliseconds_create_input_vector, start, stop));

           uint16_t threads_per_block = TPS_MAC * SPB_MAC; //1024
           uint16_t blocks_per_vector = 1 + ((N_Outputs - 1UL)
                       / (threads_per_block / TPS_MAC));
           checkCuda( cudaEventRecord(start));
//           for(unsigned int loop = 0; loop < 1000; loop++)
//           {

           complex_fractional_resampling_filter
                          <<<blocks_per_vector,
                             threads_per_block,
                             (K * (POLYNOMIAL_ORDER + 1))*sizeof(float)>>>
                                (d_complex_input_vector,
                                 N_Outputs,
                                 NCO,
                                 NCO_Step_Size,
                                 input_counter,
                                 d_complex_output_signal);


//           }
           checkCuda( cudaEventRecord(stop));

           checkCuda( cudaEventSynchronize(stop));
           float milliseconds_run_frf = 0;
           checkCuda( cudaEventElapsedTime(&milliseconds_run_frf, start, stop));
           milliseconds_run_frf = milliseconds_run_frf;// / 1000.0f;

           checkCuda( cudaEventRecord(start));
           checkCuda( cudaMemcpy(complex_output_signal, d_complex_output_signal, complex_float_data_size * N_Outputs, cudaMemcpyDeviceToHost));
           checkCuda( cudaEventRecord(stop));

           checkCuda( cudaEventSynchronize(stop));
           float milliseconds_device_to_host = 0;
           checkCuda( cudaEventElapsedTime(&milliseconds_device_to_host, start, stop));

           //make sure to throw a blocking call after this at some point before the next kernel call
           post_run_complex_state_update<<<1,K>>>(NUMBER_OF_INPUT_SAMPLES,
                                          K,
                                          d_complex_reverse_state,
                                          d_complex_input_vector);

           if(post_run_NCO_update(N_Outputs))
           {
               exit(0);
           }

           for(uint32_t sample = 0; sample < N_Outputs; sample++)
           {
              float A, B;
              A = sqrtf((test_vectors[nvec][start_index + sample].i *
                         test_vectors[nvec][start_index + sample].i)
                      + (test_vectors[nvec][start_index + sample].q *
                         test_vectors[nvec][start_index + sample].q));
              B = sqrtf((complex_output_signal[sample].i *
                         complex_output_signal[sample].i)
                      + (complex_output_signal[sample].q *
                         complex_output_signal[sample].q));
              accumulator += ((A - B) * (A - B));
           }
           start_index += N_Outputs;

           std::cout << "last run took " << milliseconds_host_to_device
                    << " to run host to device copy" << std::endl;

           std::cout << "last run took " << milliseconds_create_input_vector
                    << " to create the input vector" << std::endl;

           std::cout << "last run took " << milliseconds_run_frf
                    << " to run the frf filter" << std::endl;

           std::cout << "last run took " << milliseconds_device_to_host
                    << " to run device to host copy" << std::endl;

       }

       //Cant do this, length will change from run to run, the test_vector_length is
       //the cumulative length and is not just 8x one value but is the cumulation
       //of N_Outputs across 8 itterations
      if(sum_N_Outputs != test_vector_information[nvec]->test_vector_length)
      {
        std::cout << "error, The FRF functions are generating a different number "
                  << "of output samples than the test vectors have in them.\n"
                  << " Please correct this.  Exiting now.\n" << std::endl;
        break;
      }

      accumulator /= N_Outputs;

      std::cout << " the mean squared error is " << accumulator << std::endl;
      std::cout << " the mean squared error is " << 10*log10f(accumulator) << " dB" << std::endl;

      output_files[nvec]->close();
   }

    free(complex_input_vector); free(complex_input_signal); free(complex_output_signal);

    cudaFree(d_complex_input_signal); cudaFree(d_complex_input_vector);
    cudaFree(d_complex_output_signal); cudaFree(d_complex_reverse_state);
}
