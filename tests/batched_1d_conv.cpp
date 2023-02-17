// general parts
#include <algorithm>
#include <chrono>
#include <iostream>
#include <memory>
#include <numeric>
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

template <class T> struct ReferenceData {
  std::vector<T> input;
  std::vector<T> output;
  std::vector<std::complex<T>> kernel;
};

template <class T> ReferenceData<T> generate_data(int n, int batch) {
  const double k0 = 0.5;

  ReferenceData<T> ref_data{std::vector<T>(n * batch),
                            std::vector<T>(n * batch),
                            std::vector<std::complex<T>>(n)};
  auto &input = ref_data.input;
  auto &output = ref_data.output;
  auto &kernel = ref_data.kernel;
  // std::vector<T> input(n * batch);
  // std::vector<T> output(n * batch);
  // std::vector<std::complex<T>> kernel(n / 2 + 1);

  using std::cos;
  using std::sin;
  for (int b = 0; b < batch; ++b) {
    for (int i = 0; i < n; ++i) {
      double x = i * 2 * M_PI / k0 / n;
      double y = b * 2 * M_PI / batch;
      double val = sin(3 * k0 * x) + 0.2 * cos(13 * k0 * x + 2 * y + 3.0) -
                   0.75 * cos((n / 2 - 1) * k0 * x - (b / 2 - 1) * y + 2.0);
      double dx_val = 3 * k0 * cos(3 * k0 * x) -
                      0.2 * 13 * k0 * sin(13 * k0 * x + 2 * y + 3.0) +
                      0.75 * (n / 2 - 1) * k0 *
                          sin((n / 2 - 1) * k0 * x - (b / 2 - 1) * y + 2.0);
      input[i + b * n] = static_cast<T>(val);
      output[i + b * n] = static_cast<T>(dx_val);
    }
  }

  for (int i = 0; i < n / 2; ++i) {
    kernel[i] = std::complex<T>(0, i * k0);
  }
  kernel[n / 2] = 0;
  for (int i = n / 2 + 1; i < n; ++i) {
    kernel[i] = std::complex<T>(0, static_cast<T>((i - n) * k0));
  }

  return ref_data;
}

template <class floatT>
VkFFTResult perform_batched_1d_r2c_convolution_VkFFT(VkGPU *vkGPU, int size,
                                                     int n_batch) {

  using complexT = std::conditional_t<std::is_same<floatT, float>::value,
                                      cuFloatComplex, cuDoubleComplex>;

  auto ref_data = generate_data<floatT>(size, n_batch);

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
  uint64_t kernelSize = sizeof(complexT) * size;

  complexT *kernel = 0;
  res = cudaMalloc((void **)&kernel, kernelSize);
  if (res != cudaSuccess)
    return VKFFT_ERROR_FAILED_TO_ALLOCATE;

  // printf("Total memory needed for kernel: %" PRIu64 " B\n",
  //        kernelSize);

  // Copy generated kernel on CPU to GPU
  res = cudaMemcpy(kernel, ref_data.kernel.data(), kernelSize,
                   cudaMemcpyHostToDevice);
  if (res != cudaSuccess)
    return VKFFT_ERROR_FAILED_TO_COPY;

  // 2. Buffer convolution with kernel.
  VkFFTConfiguration configuration = {};

  // Multidimensional FFT dimensions sizes
  configuration.FFTdim = 1; // FFT dimension, 1D, 2D or 3D (default 1).
  configuration.size[0] = size;
  configuration.size[1] = 1;
  configuration.size[2] = 1;

  configuration.doublePrecision = std::is_same<floatT, float>::value ? 0 : 1;
  configuration.performR2C = true;
  configuration.coordinateFeatures = 1;
  configuration.normalize = 1; // normalize iFFT

  // After this, configuration file contains pointers to Vulkan objects needed
  // to work with the GPU: VkDevice* device - created device, [uint64_t
  // *bufferSize, VkBuffer *buffer, VkDeviceMemory* bufferDeviceMemory] -
  // allocated GPU memory FFT is performed on. [uint64_t *kernelSize, VkBuffer
  // *kernel, VkDeviceMemory* kernelDeviceMemory] - allocated GPU memory, where
  // kernel for convolution is stored.
  configuration.device = &vkGPU->device;

  configuration.kernelConvolution = false;
  configuration.performConvolution = true;
  configuration.numberBatches = n_batch;
  configuration.isInputFormatted = true;
  configuration.isOutputFormatted = true;

  // Allocate separate buffer for the input data.
  uint64_t inputBufferSize =
      configuration.numberBatches * configuration.size[0] * sizeof(floatT);
  uint64_t bufferSize = configuration.numberBatches *
                        (configuration.size[0] / 2 + 1) * sizeof(complexT);

  floatT *inputBuffer = 0;
  complexT *buffer = 0;
  res = cudaMalloc((void **)&inputBuffer, inputBufferSize);
  if (res != cudaSuccess)
    return VKFFT_ERROR_FAILED_TO_ALLOCATE;
  res = cudaMalloc((void **)&buffer, bufferSize);
  if (res != cudaSuccess)
    return VKFFT_ERROR_FAILED_TO_ALLOCATE;

  configuration.inputBufferStride[0] = configuration.size[0];
  configuration.inputBuffer = (void **)&inputBuffer;
  // configuration.inputBufferSize = &inputBufferSize;
  configuration.outputBufferStride[0] = configuration.size[0];
  configuration.outputBuffer = (void **)&inputBuffer;
  // configuration.outputBufferSize = &inputBufferSize;
  configuration.bufferSize = &bufferSize;
  configuration.buffer = (void **)&buffer;
  configuration.kernelSize = &kernelSize;
  configuration.kernel = nullptr;
  configuration.numberKernels = 1;

  // printf("Total memory needed for buffer: %" PRIu64 " B\n",
  //        bufferSize);
  // printf("Total memory needed for input buffer: %" PRIu64 " B\n",
  //        inputBufferSize);

  // Transfer data to GPU using staging buffer.
  res = cudaMemcpy(inputBuffer, ref_data.input.data(), inputBufferSize,
                   cudaMemcpyHostToDevice);
  if (res != cudaSuccess)
    return VKFFT_ERROR_FAILED_TO_COPY;
  // for (int i = 0; i < size; ++i) {
  //   std::cout << i << " " << ref_data.input[i] << "\n";
  // }

  // Initialize application responsible for the convolution.
  VkFFTApplication app_forward{};
  // configuration.keepShaderCode = 1;
  resFFT = initializeVkFFT(&app_forward, configuration);
  if (resFFT != VKFFT_SUCCESS) {
    std::cout << "Failed to initialize forward application\n";
    return resFFT;
  }


  // Sample forward FFT command buffer allocation + execution performed on
  // kernel. FFT can also be appended to user defined command buffers.
  VkFFTLaunchParams launchParams = {};
  launchParams.inputBuffer = (void **)&inputBuffer;
  launchParams.outputBuffer = (void **)&inputBuffer;
  launchParams.kernel = (void **)&kernel;
  resFFT = performVulkanFFT(vkGPU, &app_forward, &launchParams, -1, 1);
  if (resFFT != VKFFT_SUCCESS) {
    std::cout << "Forward FFT failed " << resFFT << "\n";
    return resFFT;
  }

  // Transfer data from GPU using staging buffer.
  std::vector<floatT> buffer_output(n_batch * size);
  res = cudaMemcpy(buffer_output.data(), inputBuffer, inputBufferSize,
                   cudaMemcpyDeviceToHost);
  if (res != cudaSuccess)
    return VKFFT_ERROR_FAILED_TO_COPY;

  // Check result
  // for (int i = 0; i < size; ++i) {
  //   std::cout << i << " " << buffer_output[i] << " " << ref_data.output[i]
  //             << "\n";
  // }
  for (auto i = 0u; i < ref_data.output.size(); ++i) {
    buffer_output[i] = std::abs(buffer_output[i] - ref_data.output[i]);
  }

  auto sum2 = std::reduce(
      buffer_output.begin(), buffer_output.end(), (floatT)0,
      [](const floatT &sum, const floatT &x) { return sum + x * x; });
  auto max_abs =
      *(std::max_element(buffer_output.begin(), buffer_output.end()));

  std::cout << "Transform size: " << size << "x" << n_batch << " ";
  std::cout << "Linf=" << max_abs << " L2=" << std::sqrt(sum2 / n_batch / size)
            << std::endl;

  if (max_abs > 1e-9) {
    printDebugInformation(&app_forward, &app_forward.localFFTPlan->axes[0][0]);
    return VKFFT_ERROR_UNSUPPORTED_FFT_OMIT;
  }

  cudaFree(inputBuffer);
  cudaFree(buffer);
  cudaFree(kernel);
  deleteVkFFT(&app_forward);

  return resFFT;
}

VkFFTResult launchVkFFT(VkGPU *vkGPU) {
  // Sample Vulkan project GPU initialization.
  VkFFTResult resFFT = VKFFT_SUCCESS;

#if (VKFFT_BACKEND == 0)
  VkResult res = VK_SUCCESS;
  // create instance - a connection between the application and the Vulkan
  // library
  res = createInstance(vkGPU, sample_id);
  if (res != 0) {
    // printf("Instance creation failed, error code: %" PRIu64 "\n", res);
    return VKFFT_ERROR_FAILED_TO_CREATE_INSTANCE;
  }
  // set up the debugging messenger
  res = setupDebugMessenger(vkGPU);
  if (res != 0) {
    // printf("Debug messenger creation failed, error code: %" PRIu64 "\n",
    // res);
    return VKFFT_ERROR_FAILED_TO_SETUP_DEBUG_MESSENGER;
  }
  // check if there are GPUs that support Vulkan and select one
  res = findPhysicalDevice(vkGPU);
  if (res != 0) {
    // printf("Physical device not found, error code: %" PRIu64 "\n", res);
    return VKFFT_ERROR_FAILED_TO_FIND_PHYSICAL_DEVICE;
  }
  // create logical device representation
  res = createDevice(vkGPU, sample_id);
  if (res != 0) {
    // printf("Device creation failed, error code: %" PRIu64 "\n", res);
    return VKFFT_ERROR_FAILED_TO_CREATE_DEVICE;
  }
  // create fence for synchronization
  res = createFence(vkGPU);
  if (res != 0) {
    // printf("Fence creation failed, error code: %" PRIu64 "\n", res);
    return VKFFT_ERROR_FAILED_TO_CREATE_FENCE;
  }
  // create a place, command buffer memory is allocated from
  res = createCommandPool(vkGPU);
  if (res != 0) {
    // printf("Fence creation failed, error code: %" PRIu64 "\n", res);
    return VKFFT_ERROR_FAILED_TO_CREATE_COMMAND_POOL;
  }
  vkGetPhysicalDeviceProperties(vkGPU->physicalDevice,
                                &vkGPU->physicalDeviceProperties);
  vkGetPhysicalDeviceMemoryProperties(vkGPU->physicalDevice,
                                      &vkGPU->physicalDeviceMemoryProperties);

  glslang_initialize_process(); // compiler can be initialized before VkFFT
#elif (VKFFT_BACKEND == 1)
  CUresult res = CUDA_SUCCESS;
  cudaError_t res2 = cudaSuccess;
  res = cuInit(0);
  if (res != CUDA_SUCCESS)
    return VKFFT_ERROR_FAILED_TO_INITIALIZE;
  res2 = cudaSetDevice((int)vkGPU->device_id);
  if (res2 != cudaSuccess)
    return VKFFT_ERROR_FAILED_TO_SET_DEVICE_ID;
  res = cuDeviceGet(&vkGPU->device, (int)vkGPU->device_id);
  if (res != CUDA_SUCCESS)
    return VKFFT_ERROR_FAILED_TO_GET_DEVICE;
  res = cuCtxCreate(&vkGPU->context, 0, (int)vkGPU->device);
  if (res != CUDA_SUCCESS)
    return VKFFT_ERROR_FAILED_TO_CREATE_CONTEXT;
#elif (VKFFT_BACKEND == 2)
  hipError_t res = hipSuccess;
  res = hipInit(0);
  if (res != hipSuccess)
    return VKFFT_ERROR_FAILED_TO_INITIALIZE;
  res = hipSetDevice((int)vkGPU->device_id);
  if (res != hipSuccess)
    return VKFFT_ERROR_FAILED_TO_SET_DEVICE_ID;
  res = hipDeviceGet(&vkGPU->device, (int)vkGPU->device_id);
  if (res != hipSuccess)
    return VKFFT_ERROR_FAILED_TO_GET_DEVICE;
  res = hipCtxCreate(&vkGPU->context, 0, (int)vkGPU->device);
  if (res != hipSuccess)
    return VKFFT_ERROR_FAILED_TO_CREATE_CONTEXT;
#elif (VKFFT_BACKEND == 3)
  cl_int res = CL_SUCCESS;
  cl_uint numPlatforms;
  res = clGetPlatformIDs(0, 0, &numPlatforms);
  if (res != CL_SUCCESS)
    return VKFFT_ERROR_FAILED_TO_INITIALIZE;
  cl_platform_id *platforms =
      (cl_platform_id *)malloc(sizeof(cl_platform_id) * numPlatforms);
  if (!platforms)
    return VKFFT_ERROR_MALLOC_FAILED;
  res = clGetPlatformIDs(numPlatforms, platforms, 0);
  if (res != CL_SUCCESS)
    return VKFFT_ERROR_FAILED_TO_INITIALIZE;
  uint64_t k = 0;
  for (uint64_t j = 0; j < numPlatforms; j++) {
    cl_uint numDevices;
    res = clGetDeviceIDs(platforms[j], CL_DEVICE_TYPE_ALL, 0, 0, &numDevices);
    cl_device_id *deviceList =
        (cl_device_id *)malloc(sizeof(cl_device_id) * numDevices);
    if (!deviceList)
      return VKFFT_ERROR_MALLOC_FAILED;
    res = clGetDeviceIDs(platforms[j], CL_DEVICE_TYPE_ALL, numDevices,
                         deviceList, 0);
    if (res != CL_SUCCESS)
      return VKFFT_ERROR_FAILED_TO_GET_DEVICE;
    for (uint64_t i = 0; i < numDevices; i++) {
      if (k == vkGPU->device_id) {
        vkGPU->platform = platforms[j];
        vkGPU->device = deviceList[i];
        vkGPU->context =
            clCreateContext(NULL, 1, &vkGPU->device, NULL, NULL, &res);
        if (res != CL_SUCCESS)
          return VKFFT_ERROR_FAILED_TO_CREATE_CONTEXT;
        cl_command_queue commandQueue =
            clCreateCommandQueue(vkGPU->context, vkGPU->device, 0, &res);
        if (res != CL_SUCCESS)
          return VKFFT_ERROR_FAILED_TO_CREATE_COMMAND_QUEUE;
        vkGPU->commandQueue = commandQueue;
        k++;
      } else {
        k++;
      }
    }
    free(deviceList);
  }
  free(platforms);
#elif (VKFFT_BACKEND == 4)
  ze_result_t res = ZE_RESULT_SUCCESS;
  res = zeInit(0);
  if (res != ZE_RESULT_SUCCESS)
    return VKFFT_ERROR_FAILED_TO_INITIALIZE;
  uint32_t numDrivers = 0;
  res = zeDriverGet(&numDrivers, 0);
  if (res != ZE_RESULT_SUCCESS)
    return VKFFT_ERROR_FAILED_TO_INITIALIZE;
  ze_driver_handle_t *drivers =
      (ze_driver_handle_t *)malloc(numDrivers * sizeof(ze_driver_handle_t));
  if (!drivers)
    return VKFFT_ERROR_MALLOC_FAILED;
  res = zeDriverGet(&numDrivers, drivers);
  if (res != ZE_RESULT_SUCCESS)
    return VKFFT_ERROR_FAILED_TO_INITIALIZE;
  uint64_t k = 0;
  for (uint64_t j = 0; j < numDrivers; j++) {
    uint32_t numDevices = 0;
    res = zeDeviceGet(drivers[j], &numDevices, nullptr);
    if (res != ZE_RESULT_SUCCESS)
      return VKFFT_ERROR_FAILED_TO_GET_DEVICE;
    ze_device_handle_t *deviceList =
        (ze_device_handle_t *)malloc(numDevices * sizeof(ze_device_handle_t));
    if (!deviceList)
      return VKFFT_ERROR_MALLOC_FAILED;
    res = zeDeviceGet(drivers[j], &numDevices, deviceList);
    if (res != ZE_RESULT_SUCCESS)
      return VKFFT_ERROR_FAILED_TO_GET_DEVICE;
    for (uint64_t i = 0; i < numDevices; i++) {
      if (k == vkGPU->device_id) {
        vkGPU->driver = drivers[j];
        vkGPU->device = deviceList[i];
        ze_context_desc_t contextDescription = {};
        contextDescription.stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC;
        res = zeContextCreate(vkGPU->driver, &contextDescription,
                              &vkGPU->context);
        if (res != ZE_RESULT_SUCCESS)
          return VKFFT_ERROR_FAILED_TO_CREATE_CONTEXT;

        uint32_t queueGroupCount = 0;
        res = zeDeviceGetCommandQueueGroupProperties(vkGPU->device,
                                                     &queueGroupCount, 0);
        if (res != ZE_RESULT_SUCCESS)
          return VKFFT_ERROR_FAILED_TO_CREATE_COMMAND_QUEUE;

        ze_command_queue_group_properties_t *cmdqueueGroupProperties =
            (ze_command_queue_group_properties_t *)malloc(
                queueGroupCount * sizeof(ze_command_queue_group_properties_t));
        if (!cmdqueueGroupProperties)
          return VKFFT_ERROR_MALLOC_FAILED;
        res = zeDeviceGetCommandQueueGroupProperties(
            vkGPU->device, &queueGroupCount, cmdqueueGroupProperties);
        if (res != ZE_RESULT_SUCCESS)
          return VKFFT_ERROR_FAILED_TO_CREATE_COMMAND_QUEUE;

        uint32_t commandQueueID = -1;
        for (uint32_t i = 0; i < queueGroupCount; ++i) {
          if ((cmdqueueGroupProperties[i].flags &&
               ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) &&
              (cmdqueueGroupProperties[i].flags &&
               ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COPY)) {
            commandQueueID = i;
            break;
          }
        }
        if (commandQueueID == -1)
          return VKFFT_ERROR_FAILED_TO_CREATE_COMMAND_QUEUE;
        vkGPU->commandQueueID = commandQueueID;
        ze_command_queue_desc_t commandQueueDescription = {};
        commandQueueDescription.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
        commandQueueDescription.ordinal = commandQueueID;
        commandQueueDescription.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;
        commandQueueDescription.mode = ZE_COMMAND_QUEUE_MODE_DEFAULT;
        res = zeCommandQueueCreate(vkGPU->context, vkGPU->device,
                                   &commandQueueDescription,
                                   &vkGPU->commandQueue);
        if (res != ZE_RESULT_SUCCESS)
          return VKFFT_ERROR_FAILED_TO_CREATE_COMMAND_QUEUE;
        free(cmdqueueGroupProperties);
        k++;
      } else {
        k++;
      }
    }

    free(deviceList);
  }
  free(drivers);
#endif

#define CHECK_VKFFT_RESULT(INPUT)                                              \
  if (auto resFFT = INPUT; resFFT != VKFFT_SUCCESS)                            \
  return resFFT

  CHECK_VKFFT_RESULT(
      perform_batched_1d_r2c_convolution_VkFFT<double>(vkGPU, 192, 1));
  CHECK_VKFFT_RESULT(
      perform_batched_1d_r2c_convolution_VkFFT<double>(vkGPU, 192, 2));
  CHECK_VKFFT_RESULT(
      perform_batched_1d_r2c_convolution_VkFFT<double>(vkGPU, 192, 3));
  CHECK_VKFFT_RESULT(
      perform_batched_1d_r2c_convolution_VkFFT<double>(vkGPU, 192, 9));
  CHECK_VKFFT_RESULT(
      perform_batched_1d_r2c_convolution_VkFFT<double>(vkGPU, 192, 128));
  CHECK_VKFFT_RESULT(
      perform_batched_1d_r2c_convolution_VkFFT<double>(vkGPU, 256, 1));
  CHECK_VKFFT_RESULT(
      perform_batched_1d_r2c_convolution_VkFFT<double>(vkGPU, 256, 2));
  CHECK_VKFFT_RESULT(
      perform_batched_1d_r2c_convolution_VkFFT<double>(vkGPU, 256, 9));
  CHECK_VKFFT_RESULT(
      perform_batched_1d_r2c_convolution_VkFFT<double>(vkGPU, 256, 128));
  CHECK_VKFFT_RESULT(
      perform_batched_1d_r2c_convolution_VkFFT<double>(vkGPU, 384, 1));
  CHECK_VKFFT_RESULT(
      perform_batched_1d_r2c_convolution_VkFFT<double>(vkGPU, 384, 2));
  CHECK_VKFFT_RESULT(
      perform_batched_1d_r2c_convolution_VkFFT<double>(vkGPU, 384, 9));
  CHECK_VKFFT_RESULT(
      perform_batched_1d_r2c_convolution_VkFFT<double>(vkGPU, 384, 256));
  CHECK_VKFFT_RESULT(
      perform_batched_1d_r2c_convolution_VkFFT<double>(vkGPU, 1024, 1));
  CHECK_VKFFT_RESULT(
      perform_batched_1d_r2c_convolution_VkFFT<double>(vkGPU, 1024, 8));
  CHECK_VKFFT_RESULT(
      perform_batched_1d_r2c_convolution_VkFFT<double>(vkGPU, 1024, 9));
  CHECK_VKFFT_RESULT(
      perform_batched_1d_r2c_convolution_VkFFT<double>(vkGPU, 2048, 1));
  CHECK_VKFFT_RESULT(
      perform_batched_1d_r2c_convolution_VkFFT<double>(vkGPU, 2048, 2));
  CHECK_VKFFT_RESULT(
      perform_batched_1d_r2c_convolution_VkFFT<double>(vkGPU, 2048, 8));

#if (VKFFT_BACKEND == 0)
  vkDestroyFence(vkGPU->device, vkGPU->fence, NULL);
  vkDestroyCommandPool(vkGPU->device, vkGPU->commandPool, NULL);
  vkDestroyDevice(vkGPU->device, NULL);
  DestroyDebugUtilsMessengerEXT(vkGPU, NULL);
  vkDestroyInstance(vkGPU->instance, NULL);
  glslang_finalize_process(); // destroy compiler after use
#elif (VKFFT_BACKEND == 1)
  res = cuCtxDestroy(vkGPU->context);
#elif (VKFFT_BACKEND == 2)
  res = hipCtxDestroy(vkGPU->context);
#elif (VKFFT_BACKEND == 3)
  res = clReleaseCommandQueue(vkGPU->commandQueue);
  if (res != CL_SUCCESS)
    return VKFFT_ERROR_FAILED_TO_RELEASE_COMMAND_QUEUE;
  clReleaseContext(vkGPU->context);
#elif (VKFFT_BACKEND == 4)
  res = zeCommandQueueDestroy(vkGPU->commandQueue);
  if (res != ZE_RESULT_SUCCESS)
    return VKFFT_ERROR_FAILED_TO_RELEASE_COMMAND_QUEUE;
  res = zeContextDestroy(vkGPU->context);
#endif

  return resFFT;
}

int main(int argc, char *argv[]) {
  VkGPU vkGPU = {};
#if (VKFFT_BACKEND == 0)
  vkGPU.enableValidationLayers = 0;
#endif

  VkFFTResult resFFT = launchVkFFT(&vkGPU);
  if (resFFT != VKFFT_SUCCESS)
    return resFFT;

  return 0;
}
