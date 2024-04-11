using ITensors
using ITensors.Strided
using Random

## default values for the 1d Heisenberg model
function default_vals(::Model{<:OneDHeis})
  return false, 5, [10, 20, 100, 100, 200], [1E-11], [0.0]
end

"""
  compute_1d_heisenberg(; N::Integer, kwargs...)

A function to contruct and optimize a 1D Heisenberg chain of `N`
sites (particles) using the DMRG optimization process provided by ITensors.
For a description of the ITensor DMRG algorithm see the [ITensor documentation](https://itensor.github.io/ITensors.jl/stable/DMRG.html)

Returns:
  -'energy::Number' - The eigenvalues of the optimized MPS
  -'psi' - The optimized MPS
  -'H' - The Hamiltonian MPO

Optional Keyword Arguments
  -'N::Integer' - The number of MPS sites
  -'conserve_qns::Bool' - Enable/disable Blocksparse algorithm by conserving quantum numbers
  -'conserve_pns::Bool' - TODO make this usable
  -'nsweeps' - Number of DMRG sweeps
  -'maxdim' - The maximum 
"""
function compute_1d_heisenberg(
  N::Integer;
  conserve_qns=nothing,
  conserve_pns=false,
  nsweeps=nothing,
  maxdim=nothing,
  cutoff=nothing,
  noise=nothing,
)
  defaults = default_vals(Model{OneDHeis}())

  conserve_qns = (isnothing(conserve_qns) ? defaults[1] : conserve_qns)
  nsweeps = (isnothing(nsweeps) ? defaults[2] : nsweeps)
  maxdim = (isnothing(maxdim) ? defaults[3] : maxdim)
  cutoff = (isnothing(cutoff) ? defaults[4] : cutoff)
  noise = (isnothing(noise) ? defaults[5] : noise)

  sites = siteinds("S=1", N; conserve_qns=conserve_qns)

  os = OpSum()
  for j in 1:(N - 1)
    os .+= 0.5, "S+", j, "S-", j + 1
    os .+= 0.5, "S-", j, "S+", j + 1
    os .+= "Sz", j, "Sz", j + 1
  end
  H = MPO(os, sites)

  state = [isodd(n) ? "Up" : "Dn" for n in 1:N]
  psi0 = randomMPS(sites, state, maxdim[1])

  # Run the DMRG algorithm, returning energy and optimized MPS
  energy, psi = dmrg(H, psi0; nsweeps, maxdim, cutoff, outputlevel=0)
  return (energy, psi, H)
end
