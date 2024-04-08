using CUDA
using NDTensors
include("src/BlockSparseGPUTests.jl")

begin
  BlockSparseGPUTests.test_one_d_heisenberg(; N=100, conserve_qns=true)
  BlockSparseGPUTests.test_one_d_heisenberg(NDTensors.cu; N=5, conserve_qns=false)
end
