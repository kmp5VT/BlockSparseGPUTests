#using ITensors

function default_vals(::Model{<:TwoDHubb})
  return false,
  true, 6, 3, 4.0, 1.0, 10, [100, 200, 400, 800, 1600], [1E-6],
  [1E-6, 1E-7, 1E-8, 0.0]
end

## conserve_particles
"""
```julia
  compute_2d_hubbard(N::Integer; kwargs...)
```

A function to contruct and optimize a 1D Heisenberg chain of `N`
sites (particles) using the DMRG optimization process provided by ITensors.
For a description of the ITensor DMRG algorithm see the [ITensor documentation](https://itensor.github.io/ITensors.jl/stable/DMRG.html)

Returns:

  - 'energy::Number' - The eigenvalues of the optimized MPS
  - 'psi' - The optimized MPS
  - 'H' - The Hamiltonian MPO

Required Arguments
  
  - 'Nx::Integer' - The number of MPS sites in the x dimension; default 6
  - 'Ny::Integer' - The number of MPS sites in the y dimension; default 3

Optional Keyword Arguments

  - 'conserve_qns::Bool' - Enable/disable Blocksparse algorithm by conserving quantum numbers; false
  - 'yperiodic' - Is the y dimension periodic; true
  - 'U' - TODO ; 4.0
  - 't' - TODO ; 1.0
  - 'nsweeps' - Number of DMRG sweeps; 10
  - 'maxdim' - The maximum size of the bond dimension for each DMRG sweeps, overrides cutoff; [100, 200, 400, 800, 1600] 
  - 'cutoff' - The truncating parameter for the DMRG 2-site optimization; [1E-6]
  - 'noise' - The amount of random noise added to the DMRG update step to prevent swamping behavior in difficult optimizations [1E-6, 1E-7, 1E-8, 0.0]
"""
function compute_2d_hubbard(
  Nx=nothing,
  Ny=nothing;
  conserve_qns=nothing,
  yperiodic=nothing,
  U=nothing,
  t=nothing,
  nsweeps=nothing,
  maxdim=nothing,
  cutoff=nothing,
  noise=nothing,
)
  defaults = default_vals(Model{TwoDHubb}())
  conserve_qns = (isnothing(conserve_qns) ? defaults[1] : conserve_qns)
  yperiodic = (isnothing(yperiodic) ? defaults[2] : yperiodic)
  Nx = (isnothing(Nx) ? defaults[3] : Nx)
  Ny = (isnothing(Ny) ? defaults[4] : Ny)
  N = Nx * Ny

  U = (isnothing(U) ? defaults[5] : U)
  t = (isnothing(t) ? defaults[6] : t)

  nsweeps = isnothing(nsweeps) ? defaults[7] : nsweeps
  maxdim = isnothing(maxdim) ? defaults[8] : maxdim
  cutoff = isnothing(cutoff) ? defaults[9] : cutoff
  noise = isnothing(noise) ? defaults[10] : noise

  sites = siteinds("Electron", N; conserve_qns=conserve_qns)

  lattice = square_lattice(Nx, Ny; yperiodic=yperiodic)

  os = OpSum()
  for b in lattice
    os += -t, "Cdagup", b.s1, "Cup", b.s2
    os += -t, "Cdagup", b.s2, "Cup", b.s1
    os += -t, "Cdagdn", b.s1, "Cdn", b.s2
    os += -t, "Cdagdn", b.s2, "Cdn", b.s1
  end
  for n in 1:N
    os += U, "Nupdn", n
  end
  H = MPO(os, sites)

  # Half filling
  state = [isodd(n) ? "Up" : "Dn" for n in 1:N]

  # Initialize wavefunction to a random MPS
  # of bond-dimension 10 with same quantum
  # numbers as `state`
  psi0 = randomMPS(sites, state)

  energy, psi = dmrg(H, psi0; nsweeps, maxdim, cutoff, noise)

  return energy, psi, H
end

## TODO I haven't tested this
function compute_2d_hubbard_conserve_momentum(
  Nx=nothing,
  Ny=nothing;
  conserve_qns=nothing,
  yperiodic=nothing,
  U=nothing,
  t=nothing,
  nsweeps=nothing,
  maxdim=nothing,
  cutoff=nothing,
  noise=nothing,
  conserve_ky=true,
)
  defaults = default_vals(Model{TwoDHubb}())
  conserve_qns = (isnothing(conserve_qns) ? defaults[1] : conserve_qns)
  yperiodic = (isnothing(yperiodic) ? defaults[2] : yperiodic)
  Nx = (isnothing(Nx) ? defaults[3] : Nx)
  Ny = (isnothing(Ny) ? defaults[4] : Ny)
  N = Nx * Ny

  U = (isnothing(U) ? defaults[5] : U)
  t = (isnothing(t) ? defaults[6] : t)

  nsweeps = isnothing(nsweeps) ? defaults[7] : nsweeps
  maxdim = isnothing(maxdim) ? defaults[8] : maxdim
  cutoff = isnothing(cutoff) ? defaults[9] : cutoff
  noise = isnothing(noise) ? defaults[10] : noise

  sites = siteinds("ElecK", N; conserve_qns=conserve_qns, conserve_ky, modulus_ky=Ny)

  os = hubbard(; Nx, Ny, t, U, ky=true)
  H = MPO(os, sites)

  # Number of structural nonzero elements in a bulk
  # Hamiltonian MPO tensor
  @show nnz(H[end ÷ 2])
  @show nnzblocks(H[end ÷ 2])

  # Create starting state with checkerboard
  # pattern
  state = map(CartesianIndices((Ny, Nx))) do I
    return iseven(I[1]) ⊻ iseven(I[2]) ? "↓" : "↑"
  end

  psi0 = randomMPS(itensor_rng, sites, state; linkdims=2)
  energy, psi = dmrg(H, psi0; nsweeps, maxdim, cutoff, noise,  )
  return (energy, psi, H)
end
