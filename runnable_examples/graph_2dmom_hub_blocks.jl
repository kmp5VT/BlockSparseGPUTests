include("$(@__DIR__)/../src/BlockSparseGPUTests.jl")
using JLD2, ITensors, TimerOutputs, NDTensors
using .BlockSparseGPUTests
using Plots

## Converts a block index into a tuple of their block sizes
## eg (QN() => 2, QN() => 3) -> (2,3)
function block_extents(ind::Index)
  return ntuple(i -> dim(ind.space[i]), nblocks(ind))
end

function block_extents(ind::Vector)
  return [dim(i) for i in ind]
end

## compute the psi's for each system
## All symmetries
p_symm_all, H = construct_psi_h(
  "two_d_hubbard_momentum";
  conserve_nfparity=true,
  conserve_sz=true,
  conserve_nf=true,
  conserve_ky=true,
  model=BlockSparseGPUTests.Model{BlockSparseGPUTests.TwoDHubbMed}(),
);

p_symm_noparity, H = construct_psi_h(
  "two_d_hubbard_momentum";
  conserve_nfparity=false,
  conserve_sz=true,
  conserve_nf=true,
  conserve_ky=true,
  model=BlockSparseGPUTests.Model{BlockSparseGPUTests.TwoDHubbMed}(),
);

p_symm_nfsz, H = construct_psi_h(
  "two_d_hubbard_momentum";
  conserve_nfparity=true,
  conserve_sz=true,
  conserve_nf=true,
  conserve_ky=false,
  model=BlockSparseGPUTests.Model{BlockSparseGPUTests.TwoDHubbMed}(),
);

p_symm_nf, H = construct_psi_h(
  "two_d_hubbard_momentum";
  conserve_nfparity=true,
  conserve_sz=false,
  conserve_nf=true,
  conserve_ky=false,
  model=BlockSparseGPUTests.Model{BlockSparseGPUTests.TwoDHubbMed}(),
);

p_symm_sz, H = construct_psi_h(
  "two_d_hubbard_momentum";
  conserve_nfparity=true,
  conserve_sz=true,
  conserve_nf=false,
  conserve_ky=false,
  model=BlockSparseGPUTests.Model{BlockSparseGPUTests.TwoDHubbMed}(),
);

p_symm_nfparity, H = construct_psi_h(
  "two_d_hubbard_momentum";
  conserve_nfparity=true,
  conserve_sz=false,
  conserve_nf=false,
  conserve_ky=false,
  model=BlockSparseGPUTests.Model{BlockSparseGPUTests.TwoDHubbMed}(),
);

wfns = [p_symm_all, p_symm_nfsz, p_symm_nf, p_symm_nfparity];
labels = ["Sz + Nf + Ky", "Sz + Nf", "Nf", "Nfparity"];
markers = [:cross, :star, :circle, :diamond]
t = plot()
for x in 1:4
  ## grab the middle bond
  c = linkind(wfns[x], 9)
  s = space(c)

  ## filter out only Sz = 0
  #get_spin(i) = abs(ITensors.val(i.first, "Sz")) == 0
  #sz = block_extents(filter(get_spin, s))

  ## organize all blocks from largest to smallest
  sz = sort(block_extents(s); rev=true)

  t = plot!(sz; label=labels[x], markershape=markers[x])
end

## add labels to plot
plot!(;
  xlabel="Sorted block index",
  ylabel="Block dimension",
  title="Sorted block distribution for different symmetries\n χ = 1600",
)
plot!(; xscale=:log10)
savefig("$(@__DIR__)/plots/medium/bond_dim/2d_momentum_1600.pdf")



######################################
## Plotting largest blocks vs maxdim
obs = BlockSparseGPUTests.BondDimObserver(0, Int64[])
ψ, h = construct_psi_h(
  "two_d_hubbard_momentum";
  conserve_ky = true,
  conserve_sz = true,
  conserve_nf = true,
  conserve_nfparity = true,
  model=BlockSparseGPUTests.Model{BlockSparseGPUTests.TwoDHubbMed}(),
  nsweeps=5,
  observer=obs
);
maxdims = [100, 200, 400, 800, 1600]
kysznf_dims = Int64[8,10,17,30,56]
sznf_dims = Int64[16,26,47,86,161]
nf_dims = Int64[32,61,116,224,431]
nfparity_dims = Int64[52,100,200,402,805]
obs.maxdims
using Plots
plot(maxdims, kysznf_dims, label="ky+sz+nf", markershape=:square)
plot!(maxdims, sznf_dims, label="sz+nf", markershape=:circle)
plot!(maxdims, nf_dims, label="nf", markershape=:cross)
plot!(maxdims, nfparity_dims, label="nfparity", markershape=:diamond)
plot!(title="Maxdim vs largest block", xlabel="Set maxdim", ylabel="Largest block size")
savefig("$(@__DIR__)/plots/2d-momentum-medium/block_analysis/maxdim_vs_blockdim.pdf")