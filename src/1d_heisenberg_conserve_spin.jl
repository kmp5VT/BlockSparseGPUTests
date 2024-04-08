using ITensors
using ITensors.Strided
using Random

Random.seed!(1234)

function default_vals()
  return false, 5, [10,20,100,100,200], [1E-11], [0.0]
end

function compute_1d_heisenberg(N::Integer; conserve_qns=nothing, conserve_pns = false, nsweeps=nothing, maxdim=nothing, cutoff=nothing, noise=nothing)
  defaults = default_vals()
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
  energy, psi = dmrg(H, psi0; nsweeps, maxdim, cutoff)
  return (energy, psi, H)
end
