using CUDA, Adapt
using ITensors, NDTensors
include("src/BlockSparseGPUTests.jl")
using .BlockSparseGPUTests
using .BlockSparseGPUTests: Model, TwoDHubbSmall, TwoDHubbMed, TwoDHubbLarge

adapt32(ten::ITensor) = adapt(Float32, ten)
adapt32(m::MPO) = adapt32.(m)
adapt32(m::MPS) = adapt32.(m)
model_size = "medium"
conserve_ky = false
conserve_sz = false
conserve_nf = false
conserve_nfparity = true
sweeps = 5
ψ, h = construct_psi_h(
      "two_d_hubbard_momentum"; conserve_sz, conserve_nf, conserve_ky, conserve_nfparity, nsweeps=sweeps, dev = cu
    );


