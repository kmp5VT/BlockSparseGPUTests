include("$(@__DIR__)/../src/BlockSparseGPUTests.jl")
using .BlockSparseGPUTests
using .BlockSparseGPUTests: Model, TwoDHubbSmall, TwoDHubbMed, TwoDHubbLarge
using ITensors, ITensorMPS, JLD2
function block_extents(ind::Index)
  return ntuple(i -> dim(ind.space[i]), nblocks(ind))
end

function block_extents(ind::Vector)
  return [dim(i) for i in ind]
end

## symmetry options
## symm_kysznf, 
## symm_sznf
## symm_nf
## symm_nfparity
## S2 corresponds to the largest tensor contraction and tensor
## network naming is the same as that found in BlockSparseGPUTests/notes/DMRG_contractions.pdf 
d = jldopen("$(@__DIR__)/saved_tensors_jld2/2d_momentum_hubbard/medium/symm_nfparity/S2.jld")
T1 = d["T1"]
T2 = d["T2"]
block_extents(inds(T1))
minimum.(ITensors.compute_contraction_labels( inds(T1), inds(T2)))
commoninds(T1, T2)
## Find the inner indices
innerinds = commoninds(T1, T2)
## Find the outer indices for each tensor
poutinds = noncommoninds(innerinds, inds(T1))
qoutinds = noncommoninds(innerinds, inds(T2))
## Use combiner to fuse all the outer indices and matricize each tensor
T1mat = BlockSparseGPUTests.replace_ITensor_data_with_ones(T1) * combiner(poutinds)
T2mat = BlockSparseGPUTests.replace_ITensor_data_with_ones(T2) * combiner(qoutinds)

function compute_all_blocks(T::ITensor)
  all_block_sizes = Tuple{Int64,Int64}[]
  ind(T, 1)
  be1 = block_extents.(ind(T, 1))
  be2 = block_extents.(ind(T, 2))
  for i in 1:length(be1), j in 1:length(be2)
    push!(all_block_sizes, (be1[i], be2[j]))
  end
  return all_block_sizes
end

function compute_nz_blocks(T::ITensor)
  nz_blocks = Vector{CartesianIndices{2,Tuple{UnitRange{Int64},UnitRange{Int64}}}}()
  for i in nzblocks(T)
    push!(nz_blocks, NDTensors.blockindices(T.tensor, i))
  end
  nz_block_sizes = size.(nz_blocks)
  return nz_block_sizes
end

T1all_block_sizes = compute_all_blocks(T1mat)
T2all_block_sizes = compute_all_blocks(T2mat)

T1nz_block_sizes = compute_nz_blocks(T1mat)
T2nz_block_sizes = compute_nz_blocks(T2mat)

###Use this to plot left and right hand tensors
heatplottensors = false
if (heatplottensors)
  using GLMakie, InteractiveViz

  fig = iheatmap(array(dense(T1mat)); cursor=true, colormap=:magma, legend=true)
  fig = iheatmap(array(dense(T2mat)); cursor=true, colormap=:magma, legend=true)
end