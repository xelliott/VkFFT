// general parts
#include <chrono>
#include <iostream>
#include <memory>
#include <stdio.h>
#include <string.h>
#include <thread>
#include <vector>
#ifndef __STDC_FORMAT_MACROS
#define __STDC_FORMAT_MACROS
#endif
#include <complex>
#include <inttypes.h>

#if (VKFFT_BACKEND == 0)
#include "glslang_c_interface.h"
#include "vulkan/vulkan.h"
#elif (VKFFT_BACKEND == 1)
#include <cuComplex.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <nvrtc.h>
#elif (VKFFT_BACKEND == 2)
#ifndef __HIP_PLATFORM_HCC__
#define __HIP_PLATFORM_HCC__
#endif
#include <hip/hip_complex.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <hip/hiprtc.h>
#elif (VKFFT_BACKEND == 3)
#ifndef CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#endif
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#elif (VKFFT_BACKEND == 4)
#include <ze_api.h>
#endif
#include "utils_VkFFT.h"
#include "vkFFT.h"

VkFFTResult sample_52_convolution_VkFFT_single_2d_batched_r2c(
    VkGPU *vkGPU, uint64_t file_output, FILE *output,
    uint64_t isCompilerInitialized) {
  auto pex = 0.5;
  VkFFTResult resFFT = VKFFT_SUCCESS;
#if (VKFFT_BACKEND == 0)
  VkResult res = VK_SUCCESS;
#elif (VKFFT_BACKEND == 1)
  cudaError_t res = cudaSuccess;
#elif (VKFFT_BACKEND == 2)
  hipError_t res = hipSuccess;
#elif (VKFFT_BACKEND == 3)
  cl_int res = CL_SUCCESS;
#elif (VKFFT_BACKEND == 4)
  ze_result_t res = ZE_RESULT_SUCCESS;
#endif
  if (file_output)
    fprintf(output,
            "52 - VkFFT batched convolution example with identitiy kernel\n");
  printf("52 - VkFFT batched convolution example with identitiy kernel\n");
  // Configuration + FFT application.
  VkFFTConfiguration configuration = {};
  VkFFTApplication app_kernel = {};
  // Convolution sample code
  // Setting up FFT configuration. FFT is performed in-place with no performance
  // loss.

  configuration.FFTdim = 1;   // FFT dimension, 1D, 2D or 3D (default 1).
  configuration.size[0] = 4096; // Multidimensional FFT dimensions sizes (default
                              // 1). For best performance (and stability), order
                              // dimensions in descendant size order as: x>y>z.
  configuration.size[1] = 1;
  configuration.size[2] = 1;

  using floatT = double;
  using complexT = cuDoubleComplex;
  configuration.doublePrecision = 1;
  configuration.kernelConvolution =
      true; // specify if this plan is used to create kernel for convolution
  configuration.performR2C =
      true; // Perform R2C/C2R transform. Can be combined with all other
            // options. Reduces memory requirements by a factor of 2. Requires
            // special input data alignment: for x*y*z system pad x*y plane to
            // (x+2)*y with last 2*y elements reserved, total array dimensions
            // are (x*y+2y)*z. Memory layout after R2C and before C2R can be
            // found on github.
  configuration.coordinateFeatures =
      1; // Specify dimensionality of the input feature vector (default 1). Each
         // component is stored not as a vector, but as a separate system and
         // padded on it's own according to other options (i.e. for x*y system
         // of 3-vector, first x*y elements correspond to the first dimension,
         // then goes x*y for the second, etc).
  // coordinateFeatures number is an important constant for convolution. If we
  // perform 1x1 convolution, it is equal to number of features, but
  // matrixConvolution should be equal to 1. For matrix convolution, it must be
  // equal to matrixConvolution parameter. If we perform 2x2 convolution, it is
  // equal to 3 for symmetric kernel (stored as xx, xy, yy) and 4 for
  // nonsymmetric (stored as xx, xy, yx, yy). Similarly, 6 (stored as xx, xy,
  // xz, yy, yz, zz) and 9 (stored as xx, xy, xz, yx, yy, yz, zx, zy, zz) for
  // 3x3 convolutions.
  configuration.normalize = 1; // normalize iFFT

  configuration.numberBatches = 1;
  // After this, configuration file contains pointers to Vulkan objects needed
  // to work with the GPU: VkDevice* device - created device, [uint64_t
  // *bufferSize, VkBuffer *buffer, VkDeviceMemory* bufferDeviceMemory] -
  // allocated GPU memory FFT is performed on. [uint64_t *kernelSize, VkBuffer
  // *kernel, VkDeviceMemory* kernelDeviceMemory] - allocated GPU memory, where
  // kernel for convolution is stored.
  configuration.device = &vkGPU->device;
  // In this example, we perform a convolution for a real vectorfield (3vector)
  // with a symmetric kernel (6 values). We use configuration to initialize
  // convolution kernel first from real data, then we create
  // convolution_configuration for convolution. The buffer object from
  // configuration is passed to convolution_configuration as kernel object.
  // 1. Kernel forward FFT.
  uint64_t kernelSize = sizeof(complexT) * (configuration.size[0]) *
                        configuration.size[1] * configuration.size[2];
  ;

  complexT *kernel = 0;
  res = cudaMalloc((void **)&kernel, kernelSize);
  if (res != cudaSuccess)
    return VKFFT_ERROR_FAILED_TO_ALLOCATE;
  configuration.buffer = (void **)&kernel;

  configuration.bufferSize = &kernelSize;

  if (file_output)
    fprintf(output, "Total memory needed for kernel: %" PRIu64 " MB\n",
            kernelSize / 1024 / 1024);
  printf("Total memory needed for kernel: %" PRIu64 " MB\n",
         kernelSize / 1024 / 1024);

  // Fill kernel on CPU.
  floatT *kernel_input = (floatT *)malloc(kernelSize);
  if (!kernel_input)
    return VKFFT_ERROR_MALLOC_FAILED;
  for (uint64_t k = 0; k < configuration.size[2]; k++) {
    for (uint64_t j = 0; j < configuration.size[1]; j++) {
      auto offset = j * (configuration.size[0] + 2) +
                    k * (configuration.size[0] + 2) * configuration.size[1];

      // Below is the test identity kernel for 1x1 nonsymmetric FFT,
      // multiplied by (f * configuration.coordinateFeatures + v + 1);
      for (uint64_t i = 0; i < configuration.size[0] / 2 + 1; i++) {

        kernel_input[2 * i + offset] = 0.0;
        kernel_input[2 * i + 1 + offset] = i * pex;
      }
      kernel_input[configuration.size[0] + 1 + offset] = 0;
      for (uint64_t i = configuration.size[0] - 1;
           i >= configuration.size[0] / 2 + 1; i--) {

        kernel_input[2 * i + offset] = 0.0;
        kernel_input[2 * i + 1 + offset] =
            ((int)i - (int)configuration.size[0]) * pex;
      }
    }
  }
  // Sample buffer transfer tool. Uses staging buffer of the same size as
  // destination buffer, which can be reduced if transfer is done sequentially
  // in small buffers.
  res = cudaMemcpy(kernel, kernel_input, kernelSize, cudaMemcpyHostToDevice);
  if (res != cudaSuccess)
    return VKFFT_ERROR_FAILED_TO_COPY;
  // Initialize application responsible for the kernel. This function loads
  // shaders, creates pipeline and configures FFT based on configuration file.
  // No buffer allocations inside VkFFT library.
  // resFFT = initializeVkFFT(&app_kernel, configuration);
  // if (resFFT != VKFFT_SUCCESS)
  //   return resFFT;
  // Sample forward FFT command buffer allocation + execution performed on
  // kernel. Second number determines how many times perform application in one
  // submit. FFT can also be appended to user defined command buffers.

  // Uncomment the line below if you want to perform kernel FFT. In this sample
  // we use predefined identitiy kernel.
  {
    // VkFFTLaunchParams launchParams = {};
    // performVulkanFFT(vkGPU, &app_kernel, &launchParams, -1, 1);
    std::cout << "Kernel\n";
    cudaMemcpy(kernel_input, kernel, kernelSize, cudaMemcpyDeviceToHost);
    for (uint64_t i = 0; i < configuration.size[0]; i++) {
      printf("%lu (%f, %f)\n", i, kernel_input[2 * i], kernel_input[2 * i + 1]);
    }
  }

  // The kernel has been trasnformed.

  // 2. Buffer convolution with transformed kernel.
  // Copy configuration, as it mostly remains unchanged. Change specific parts.
  auto forward_configuration = configuration;
  forward_configuration.kernelConvolution = false;
  forward_configuration.performConvolution = true;
  forward_configuration.numberBatches = 2;
  forward_configuration.isInputFormatted = true;
  forward_configuration.isOutputFormatted = true;

  // Allocate separate buffer for the input data.
  uint64_t inputBufferSize = forward_configuration.numberBatches *
                             sizeof(floatT) * forward_configuration.size[0] *
                             forward_configuration.size[1] *
                             forward_configuration.size[2];
  ;
  uint64_t bufferSize = forward_configuration.numberBatches * sizeof(complexT) *
                        (forward_configuration.size[0] / 2 + 1) *
                        forward_configuration.size[1] *
                        forward_configuration.size[2];
  ;

  floatT *inputBuffer = 0;
  floatT *outputBuffer = 0;
  complexT *buffer = 0;
  res = cudaMalloc((void **)&inputBuffer, inputBufferSize);
  if (res != cudaSuccess)
    return VKFFT_ERROR_FAILED_TO_ALLOCATE;
  res = cudaMalloc((void **)&outputBuffer, inputBufferSize);
  if (res != cudaSuccess)
    return VKFFT_ERROR_FAILED_TO_ALLOCATE;
  res = cudaMalloc((void **)&buffer, bufferSize);
  if (res != cudaSuccess)
    return VKFFT_ERROR_FAILED_TO_ALLOCATE;

  forward_configuration.inputBufferStride[0] = forward_configuration.size[0];
  // forward_configuration.inputBuffer = (void **)&inputBuffer;
  // forward_configuration.inputBufferSize = &inputBufferSize;
  forward_configuration.outputBufferStride[0] = forward_configuration.size[0];
  // forward_configuration.outputBuffer = (void **)&outputBuffer;
  // forward_configuration.outputBufferSize = &inputBufferSize;
  forward_configuration.bufferSize = &bufferSize;
  forward_configuration.buffer = (void **)&buffer;
  forward_configuration.kernelSize = &kernelSize;
  forward_configuration.kernel = (void **)&kernel;

  if (file_output)
    fprintf(output, "Total memory needed for buffer: %" PRIu64 " MB\n",
            bufferSize / 1024 / 1024);
  printf("Total memory needed for buffer: %" PRIu64 " MB\n",
         bufferSize / 1024 / 1024);
  // Fill data on CPU. It is best to perform all operations on GPU after initial
  // upload.
  floatT *buffer_input = (floatT *)malloc(inputBufferSize);
  if (!buffer_input)
    return VKFFT_ERROR_MALLOC_FAILED;
  std::cout << "input\n";
  for (uint64_t k = 0; k < forward_configuration.numberBatches; k++) {
    for (uint64_t j = 0; j < forward_configuration.size[1]; j++) {
      for (uint64_t i = 0; i < forward_configuration.size[0]; i++) {
        floatT x = i * 2 * M_PI / pex / forward_configuration.size[0];
        floatT val = (k + 1) * (std::sin(3 * pex * x) +
                                0.2 * std::cos(13 * pex * x + 3.0));
        buffer_input[i + j * forward_configuration.size[0] +
                     k * forward_configuration.size[0] *
                         forward_configuration.size[1]] = val;
        // std::sin(8 * x) - 0.3 * std::cos(14 * x) +
        // 0.2 * std::cos(11 * x + 5.0);
        std::cout << k << " " << i << " " << val << "\n";
      }
    }
  }
  // Transfer data to GPU using staging buffer.
  res = cudaMemcpy(inputBuffer, buffer_input, inputBufferSize,
                   cudaMemcpyHostToDevice);
  if (res != cudaSuccess)
    return VKFFT_ERROR_FAILED_TO_COPY;

  // Initialize application responsible for the convolution.
  VkFFTApplication app_forward{};
  forward_configuration.keepShaderCode = 1;
  resFFT = initializeVkFFT(&app_forward, forward_configuration);
  if (resFFT != VKFFT_SUCCESS) {
    std::cout << "Failed to initialize forward application\n";
    return resFFT;
  } else {
    std::cout << "Forward application initialized\n";
  }
  // Sample forward FFT command buffer allocation + execution performed on
  // kernel. FFT can also be appended to user defined command buffers.
  VkFFTLaunchParams launchParams = {};
  launchParams.inputBuffer = (void **)&inputBuffer;
  launchParams.outputBuffer = (void **)&outputBuffer;
  resFFT = performVulkanFFT(vkGPU, &app_forward, &launchParams, -1, 1);
  if (resFFT != VKFFT_SUCCESS) {
    std::cout << "Forward FFT failed " << resFFT << "\n";
    return resFFT;
  } else {
    std::cout << "Forward FFT finished " << resFFT << "\n";
  }

  floatT *buffer_output = (floatT *)malloc(bufferSize);
  if (!buffer_output)
    return VKFFT_ERROR_MALLOC_FAILED;
  // Transfer data from GPU using staging buffer.
  res = cudaMemcpy(buffer_output, buffer, bufferSize, cudaMemcpyDeviceToHost);
  if (res != cudaSuccess)
    return VKFFT_ERROR_FAILED_TO_COPY;

  // Print data, if needed.
  //  std::cout << "Forward Output:\n";
  //  for (uint64_t k = 0; k < forward_configuration.numberBatches; k++) {
  //    for (uint64_t j = 0; j < forward_configuration.size[1]; j++) {
  //      for (uint64_t i = 0; i < forward_configuration.size[0]; i++) {
  //        float x = i*2*M_PI/forward_configuration.size[0];

  //       if (file_output)
  //         fprintf(output, "%.6f ", buffer_output[i + j *
  //         forward_configuration.size[0] + k * forward_configuration.size[0] *
  //         forward_configuration.size[1]]);

  //       std::cout << buffer_output[i + j * forward_configuration.size[0] + k
  //       * forward_configuration.size[0] * forward_configuration.size[1]] << "
  //       " << 8*std::cos(8*float(x)) + (float)4.2*std::sin(14*float(x)) <<
  //       "\n";
  //     }
  //     std::cout << "\n";
  //   }
  // }

  // resFFT = performVulkanFFT(vkGPU, &app_convolution, &launchParams, 1, 1);
  // if (resFFT != VKFFT_SUCCESS) return resFFT;

  res = cudaMemcpy(buffer_input, outputBuffer, inputBufferSize,
                   cudaMemcpyDeviceToHost);
  if (res != cudaSuccess)
    return VKFFT_ERROR_FAILED_TO_COPY;
  std::cout << "Backward output:\n";
  for (uint64_t k = 0; k < forward_configuration.numberBatches; k++) {
    for (uint64_t j = 0; j < forward_configuration.size[1]; j++) {
      for (uint64_t i = 0; i < forward_configuration.size[0]; i++) {
        floatT x = i * 2 * M_PI / pex / forward_configuration.size[0];
        // auto reference = 8 * std::cos(8 * x) + 4.2 * std::sin(14 * x) -
        //                  2.2 * std::sin(11 * x + 5.0);
        auto reference =
            (k + 1) * (3 * pex * std::cos(3 * pex * x) -
                       0.2 * 13 * pex * std::sin(13 * pex * x + 3.0));
        auto result = buffer_input[i + j * forward_configuration.size[0] +
                                   k * forward_configuration.size[0] *
                                       forward_configuration.size[1]];
        if (!(std::abs(result - reference) < std::abs(reference) * 1e-4 ||
              (std::abs(reference) < 1e-5 && std::abs(result) < 1e-5))) {
          std::cout << "Error at " << k << " " << i << " " << result << " "
                    << reference << "\n";
        }
      }
    }
  }

  free(kernel_input);
  free(buffer_input);
  free(buffer_output);
  cudaFree(inputBuffer);
  cudaFree(buffer);
  cudaFree(kernel);
  // deleteVkFFT(&app_kernel);
  // deleteVkFFT(&app_convolution);
  deleteVkFFT(&app_forward);
  return resFFT;
}

VkFFTResult sample_53_convolution_VkFFT_single_2d_batched_r2c_pad(
    VkGPU *vkGPU, uint64_t file_output, FILE *output,
    uint64_t isCompilerInitialized) {
  auto pex = 0.5;
  VkFFTResult resFFT = VKFFT_SUCCESS;
#if (VKFFT_BACKEND == 0)
  VkResult res = VK_SUCCESS;
#elif (VKFFT_BACKEND == 1)
  cudaError_t res = cudaSuccess;
#elif (VKFFT_BACKEND == 2)
  hipError_t res = hipSuccess;
#elif (VKFFT_BACKEND == 3)
  cl_int res = CL_SUCCESS;
#elif (VKFFT_BACKEND == 4)
  ze_result_t res = ZE_RESULT_SUCCESS;
#endif
  if (file_output)
    fprintf(output,
            "52 - VkFFT batched convolution example with identitiy kernel\n");
  printf("52 - VkFFT batched convolution example with identitiy kernel\n");
  // Configuration + FFT application.
  VkFFTConfiguration configuration = {};
  VkFFTApplication app_kernel = {};
  // Convolution sample code
  // Setting up FFT configuration. FFT is performed in-place with no performance
  // loss.

  configuration.FFTdim = 1;   // FFT dimension, 1D, 2D or 3D (default 1).
  configuration.size[0] = 32; // Multidimensional FFT dimensions sizes (default
                              // 1). For best performance (and stability), order
                              // dimensions in descendant size order as: x>y>z.
  configuration.size[1] = 1;
  configuration.size[2] = 1;
  configuration.numberBatches = 2;

  using floatT = float;
  using complexT = cuFloatComplex;
  configuration.doublePrecision = 0;
  configuration.performR2C = true;
  configuration.normalize = 1; // normalize iFFT
  configuration.device = &vkGPU->device;

  configuration.inverseReturnToInputBuffer = true;
  configuration.isInputFormatted = true;
  configuration.inputBufferStride[0] = configuration.size[0];
  configuration.bufferStride[0] = configuration.size[0] / 2 + 1;
  uint64_t inputBufferSize = configuration.numberBatches * sizeof(floatT) *
                             configuration.size[0] * configuration.size[1] *
                             configuration.size[2];
  uint64_t bufferSize = configuration.numberBatches * sizeof(complexT) *
                        (configuration.size[0] / 2 + 1) *
                        configuration.size[1] * configuration.size[2];

  floatT *inputBuffer = 0;
  complexT *buffer = 0;
  res = cudaMalloc((void **)&inputBuffer, inputBufferSize);
  if (res != cudaSuccess)
    return VKFFT_ERROR_FAILED_TO_ALLOCATE;
  res = cudaMalloc((void **)&buffer, bufferSize);
  if (res != cudaSuccess)
    return VKFFT_ERROR_FAILED_TO_ALLOCATE;

  configuration.inputBufferSize = &inputBufferSize;
  configuration.bufferSize = &bufferSize;
  configuration.buffer = (void **)&buffer;

  // Fill data on CPU. It is best to perform all operations on GPU after initial
  // upload.
  floatT *buffer_input = (floatT *)malloc(inputBufferSize);
  if (!buffer_input)
    return VKFFT_ERROR_MALLOC_FAILED;
  std::cout << "input\n";
  for (uint64_t k = 0; k < configuration.numberBatches; k++) {
    for (uint64_t j = 0; j < configuration.size[1]; j++) {
      for (uint64_t i = 0; i < configuration.size[0]; i++) {
        floatT x = i * 2 * M_PI / pex / configuration.size[0];
        floatT val = (k + 1) * (std::sin(3 * pex * x) +
                                0.2 * std::cos(13 * pex * x + 3.0));
        buffer_input[i + j * configuration.size[0] +
                     k * configuration.size[0] * configuration.size[1]] = val;
        // std::sin(8 * x) - 0.3 * std::cos(14 * x) +
        // 0.2 * std::cos(11 * x + 5.0);
        std::cout << k << " " << i << " " << val << "\n";
      }
    }
  }
  // Transfer data to GPU using staging buffer.
  res = cudaMemcpy(inputBuffer, buffer_input, inputBufferSize,
                   cudaMemcpyHostToDevice);
  if (res != cudaSuccess)
    return VKFFT_ERROR_FAILED_TO_COPY;

  // Initialize application responsible for the convolution.
  VkFFTApplication app{};
  resFFT = initializeVkFFT(&app, configuration);
  if (resFFT != VKFFT_SUCCESS) {
    std::cout << "Failed to initialize application\n";
    return resFFT;
  }

  VkFFTApplication app_backward{};
  configuration.performZeropadding[0] = 1;
  configuration.frequencyZeroPadding = 1;
  configuration.fft_zeropad_left[0] = 14;
  configuration.fft_zeropad_right[0] = 17;
  configuration.makeInversePlanOnly = 1;
  resFFT = initializeVkFFT(&app_backward, configuration);
  if (resFFT != VKFFT_SUCCESS) {
    std::cout << "Failed to initialize backward application\n";
    return resFFT;
  }

  // Sample forward FFT command buffer allocation + execution performed on
  // kernel. FFT can also be appended to user defined command buffers.
  VkFFTLaunchParams launchParams = {};
  launchParams.inputBuffer = (void **)&inputBuffer;
  resFFT = performVulkanFFT(vkGPU, &app, &launchParams, -1, 1);
  if (resFFT != VKFFT_SUCCESS) {
    std::cout << "Forward FFT failed " << resFFT << "\n";
    return resFFT;
  } else {
    std::cout << "Forward FFT finished " << resFFT << "\n";
  }

  std::complex<floatT> *buffer_output =
      (std::complex<floatT> *)malloc(bufferSize);
  if (!buffer_output)
    return VKFFT_ERROR_MALLOC_FAILED;
  // Transfer data from GPU using staging buffer.
  res = cudaMemcpy(buffer_output, buffer, bufferSize, cudaMemcpyDeviceToHost);
  if (res != cudaSuccess)
    return VKFFT_ERROR_FAILED_TO_COPY;

  // Print forward FFT result.
  std::cout << "Forward Output:\n";
  for (uint64_t k = 0; k < configuration.numberBatches; k++) {
    for (uint64_t j = 0; j < configuration.size[1]; j++) {
      for (uint64_t i = 0; i < configuration.size[0] / 2 + 1; i++) {
        std::cout << k << " " << i << " "
                  << buffer_output[i + k * (configuration.size[0] / 2 + 1)]
                  << "\n";
      }
      std::cout << "\n";
    }
  }

  resFFT = performVulkanFFT(vkGPU, &app_backward, &launchParams, 1, 1);
  if (resFFT != VKFFT_SUCCESS) {
    std::cout << "Backward FFT failed " << resFFT << "\n";
    return resFFT;
  } else {
    std::cout << "Backward FFT finished " << resFFT << "\n";
  }

  // resFFT = performVulkanFFT(vkGPU, &app_convolution, &launchParams, 1, 1);
  // if (resFFT != VKFFT_SUCCESS) return resFFT;

  res = cudaMemcpy(buffer_input, inputBuffer, inputBufferSize,
                   cudaMemcpyDeviceToHost);
  if (res != cudaSuccess)
    return VKFFT_ERROR_FAILED_TO_COPY;
  std::cout << "Backward output:\n";
  for (uint64_t k = 0; k < configuration.numberBatches; k++) {
    for (uint64_t j = 0; j < configuration.size[1]; j++) {
      for (uint64_t i = 0; i < configuration.size[0]; i++) {
        floatT x = i * 2 * M_PI / pex / configuration.size[0];
        // auto reference = 8 * std::cos(8 * x) + 4.2 * std::sin(14 * x) -
        //                  2.2 * std::sin(11 * x + 5.0);
        auto reference = (k + 1) * (std::sin(3 * pex * x) +
                                    0.2 * std::cos(13 * pex * x + 3.0));
        auto result =
            buffer_input[i + j * configuration.size[0] +
                         k * configuration.size[0] * configuration.size[1]];
        if (!(std::abs(result - reference) < std::abs(reference) * 1e-4 ||
              (std::abs(reference) < 1e-5 && std::abs(result) < 1e-5))) {
          std::cout << "Error at " << k << " " << i << " " << result << " "
                    << reference << "\n";
        }
      }
    }
  }

  free(buffer_input);
  free(buffer_output);
  cudaFree(inputBuffer);
  cudaFree(buffer);
  // deleteVkFFT(&app_kernel);
  // deleteVkFFT(&app_convolution);
  deleteVkFFT(&app);
  deleteVkFFT(&app_backward);
  return resFFT;
}
