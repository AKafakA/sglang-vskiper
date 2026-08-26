export CUDA_HOME=$R/envs/sglang-serve-w2-r1/lib/python3.12/site-packages/nvidia/cu13
export PATH="$CUDA_HOME/bin:$GREAL/bin:$R/envs/sglang-serve-w2-r1/bin:/usr/bin:/bin"
export CC=$GREAL/bin/gcc CXX=$GREAL/bin/g++
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$GREAL/lib64"
export CPATH="$R/envs/sglang-serve-w2-r1/lib/python3.12/site-packages/flashinfer/data/cccl/libcudacxx/include"
export NVCC_PREPEND_FLAGS="-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK"
export SGLANG_FD_ACTIVE_PHASES=both
export SGLANG_FD_EXECUTION_MODE=full_graph
export SGLANG_FD_FULL_GRAPH_DEVICE_ROUTE_TAPE=1
export SGLANG_FD_FULL_GRAPH_LAYER_COUNTERS=1
export SGLANG_FD_FULL_GRAPH_MASKED_DECODE_ATTENTION=1
export SGLANG_FD_FULL_GRAPH_ROUTE_ACCOUNTING=1
export SGLANG_VP_ADASKIP_PROFILE=$R/serving/f7-adaskip/adaskip_fixed_profile.json
export SGLANG_VP_FULL_GRAPH_SKIPPER=adaskip
