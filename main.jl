using CUDA
include("src/BlockSparseGPUTests.jl")

begin
    BlockSparseGPUTests.test_one_d_heisenberg(;N=100, conserve_qns=true)
    BlockSparseGPUTests.test_one_d_heisenberg(cu;N=100, conserve_qns=true)

end