#include <cuda_runtime.h>

#include <cstdint>

namespace {

__global__ void set_conditional(cudaGraphConditionalHandle handle,
                                const int32_t* predicate) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    cudaGraphSetConditional(handle, predicate[0] != 0);
  }
}

__global__ void set_conditional_value(cudaGraphConditionalHandle handle,
                                      const int32_t* value) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    cudaGraphSetConditional(handle, static_cast<unsigned int>(value[0]));
  }
}

}  // namespace

extern "C" int vp_launch_set_conditional(uint64_t handle,
                                          const int32_t* predicate,
                                          void* stream) {
  set_conditional<<<1, 1, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
      static_cast<cudaGraphConditionalHandle>(handle), predicate);
  return static_cast<int>(cudaGetLastError());
}

extern "C" int vp_launch_set_conditional_value(uint64_t handle,
                                                const int32_t* value,
                                                void* stream) {
  set_conditional_value<<<1, 1, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
      static_cast<cudaGraphConditionalHandle>(handle), value);
  return static_cast<int>(cudaGetLastError());
}

extern "C" int vp_conditional_graph_cuda_runtime_version() {
  int version = 0;
  const cudaError_t status = cudaRuntimeGetVersion(&version);
  return status == cudaSuccess ? version : -static_cast<int>(status);
}
