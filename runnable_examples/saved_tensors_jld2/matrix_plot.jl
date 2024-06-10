include("$(@__DIR__)/../../src/BlockSparseGPUTests.jl")
using .BlockSparseGPUTests
using .BlockSparseGPUTests: Model, TwoDHubbSmall, TwoDHubbMed, TwoDHubbLarge
using ITensors, ITensorMPS, JLD2
d = jldopen("$(@__DIR__)/2d_momentum_hubbard/medium/symm_nfparity/S2.jld")
p = d["T1"]
q = d["T2"]
commoninds(p, q)
innerinds = commoninds(p,q)
poutinds = noncommoninds(innerinds, inds(p))
qoutinds = noncommoninds(innerinds, inds(q))
pmat = BlockSparseGPUTests.replace_ITensor_data_with_random(p) * combiner(poutinds)
qmat = BlockSparseGPUTests.replace_ITensor_data_with_random(q) * combiner(qoutinds)

function block_extents(ind::Index)
    return ntuple(i -> dim(ind.space[i]), nblocks(ind))
end
  
function block_extents(ind::Vector)
    return [dim(i) for i in ind]
end
all_blocks = [block_extents(ind(pmat,1)), block_extents(ind(pmat,2))]
nz_blocks = Vector{CartesianIndices{2, Tuple{UnitRange{Int64}, UnitRange{Int64}}}}()


using Makie, GLMakie, ModernGL, InteractiveViz
fig = iheatmap(array(dense(qmat)); cursor=true, colormap=:magma, legend=true)