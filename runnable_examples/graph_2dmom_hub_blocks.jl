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
