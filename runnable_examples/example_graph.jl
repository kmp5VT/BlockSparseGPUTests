include("$(@__DIR__)/../src/BlockSparseGPUTests.jl")
using HDF5, ITensors, TimerOutputs, NDTensors
using .BlockSparseGPUTests
using Plots

## Converts a block index into a tuple of their block sizes
## eg (QN() => 2, QN() => 3) -> (2,3)
function block_extents(ind::Index)
  return ntuple(i -> dim(ind.space[i]), nblocks(ind))
end

function combine_blockextents(blockextents::Vector)
  hold = blockextents[1]
  for i in 2:length(blockextents)
    be = blockextents[i]
    l = length(hold) * length(be)
    hold_new = reshape([i * j for i in hold, j in be], l)
    hold = hold_new
  end

  return hold
end

## So here I want to take a specific tensor and calculate
## the size of all blocks in the tensor. 
function get_tensor_block_sizes(A::ITensor)
  is = inds(A)
  ## count the number of blocks in each index of A
  blockextents = [block_extents(i) for i in is]

  ## for each index of A compute the product of the size of each block
  return combine_blockextents(blockextents)
end

##
function histogram_tensor_block_size(
  A::ITensor; label::String, nonzero::Bool=false, color::Symbol
)
  ## convert tuple to vector for histogram
  if !nonzero
    AllBlockDimensions = log10.(get_tensor_block_sizes(A))
    histogram(
      AllBlockDimensions;
      label=label,
      color=color,
      fillalpha=0.35,
      xlabel="Log10 Number of elements in block",
      ylabel="Number of instances of block"
    )
  else
    NZBlockDimensions = log10.([blockdim(tensor(A), i) for i in nzblocks(A)])
    histogram(
      NZBlockDimensions; label=label, color=color,fillalpha=0.35,
      xlabel="Log10 Number of elements in block",
      ylabel="Number of instances of block"
    )
  end
end


begin 
  for filename in ["EL1", "EL2", "S1", "S2", "S3"]
    size = "small"
    fid = h5open("$(@__DIR__)/hdf5/$(size)/sparse/$(filename).h5")
    for j in ["T1", "T2"]
      T = read(fid, j, ITensor)

      histogram_tensor_block_size(T; label="$(j) All", color=:black, nonzero=false)
      t = histogram!(; title="$(j) $(filename) $(size) all blocks")
        savefig("$(@__DIR__)/plots/$(size)/all_blocks/$(filename)_all_block_dimensions_$(j).pdf")

      t = histogram_tensor_block_size(T; label="$(j) NZ", color=:green, nonzero=true)
      t = histogram!(; title="$(j) $(filename) $(size) nonzero blocks")
        savefig("$(@__DIR__)/plots/$(size)/nonzero/$(filename)_nz_block_dimensions_$(j).pdf")
    end
    close(fid)
  end
end

## Here I want to 3d histogram plot block sizes of i, j and k in a tensor contraction
## So given a set of indices for i, compute the block sizes of each
function historgram_operational_intensity(indsI, indsJ, indsK, label::String)
  ext_ijk = collect.(combine_blockextents([block_extents(p) for p in q]) for q in (indsI, indsJ, indsK))

  d = prod(length.(ext_ijk))
  #ei,ej,ek = Vector{Int}.(undef, (d,d,d))
  op_int = Vector{Float64}(undef, d)
  num = 1
  for j in ext_ijk[2]
    for i in ext_ijk[1]
      for k in ext_ijk[3]
        # ei[num] = i
        # ej[num] = j
        # ek[num] = k

        op_int[num] = 2.0 * i * j * k /1e9
        num += 1
      end
    end
  end

  op_int = log10.(op_int)
  histogram(op_int, xlabel="GEMM intensity\nlog10(GFLOPS)", ylabel="number of instances", label="", title="$(label) block contraction cost",fillalpha=0.35)
  # histogram!(op_int, label=label, fillalpha=0.35)
end

function histogram_contract_inds(T1::ITensor, T2::ITensor, label::String)
  j = commoninds(T1, T2)
  i = noncommoninds(inds(T1), j)
  k = noncommoninds(j, inds(T2))

  historgram_operational_intensity(i,j,k, label)
end

begin
  size = "small"
  for i in ["EL1", "EL2", "S1", "S2", "S3"]
    fid = h5open("$(@__DIR__)/hdf5/$(size)/sparse/$(i).h5")
    T1 = read(fid, "T1", ITensor)
    T2 = read(fid, "T2", ITensor)
    close(fid)

    t = histogram_contract_inds(T1, T2, "El1")
    savefig("$(@__DIR__)/plots/$(size)/op_int/$(i)_block_op_int_hist.pdf")
  end
end

begin
  size = "small"
  fid = h5open("$(@__DIR__)/hdf5/$(size)/sparse/psi.h5")
  psi = read(fid, "psi", MPS)
  length(psi)
  c = linkind(psi,2)
  cdims = collect(block_extents(c))
  t = plot(cdims, label = "site 4")
  for i in [4,6,8,10]
    c = linkind(psi,i)
    cdims = collect(block_extents(c))
    t = plot!(cdims, label="site $(i)")
  end
end
t
plot!(xlabel="Block index", ylabel="Block dimension", title="Block distribution for different bond indices")
savefig("$(@__DIR__)/plots/small/bond_dim/small_bond_dims.pdf")

c = linkind(psi,8)
cdims = collect(block_extents(c))
using Distributions
f = fit(Distributions.Normal, cdims)

m(x) = (1/√(2 * π * f.σ^2)) * exp(-1 * (x - f.μ)^2/(2 * f.σ^2))

x = range(0, 1, length=43)
y = @. m(x)

m(20)
f
mean(cdims)