using NDTensors
using TimerOutputs: TimerOutput, @timeit
## Model is a string with option of 
# one_d_heisenberg
# two_d_hubbard
"""
```julia
construct_psi_h(model::String; kwargs...)
```

A function to easily construct a wavefunction ψ and Hamiltonian given a model name.

Returns:

  - 'psi' - The optimized MPS
  - 'H' - The Hamiltonian MPO

Required Arguments
  
  - model::String - name of available models; options "one_d_heisenberg", "two_d_hubbard"
  
Optional Keyword Arguments

  - 'N::Integer' - The number of MPS sites (for 1d systems); 
  - 'Nx::Integer' - The number of sites in the x mode (for 2d systems);
  - 'Ny::Integer' - The number of sites in the y mode (for 2d systems);
  - 'conserve_qns::Bool' - Enable/disable Blocksparse algorithm by conserving quantum numbers; false
  - 'conserve_pns::Bool' - TODO make this usable; false
  - 'nsweeps' - Number of DMRG sweeps; 5
  - 'maxdim' - The maximum size of the bond dimension for each DMRG sweeps, overrides cutoff; [10, 20, 100, 100, 200]
  - 'cutoff' - The truncating parameter for the DMRG 2-site optimization; [1e-11]
  - 'noise' - The amount of random noise added to the DMRG update step to prevent swamping behavior in difficult optimizations [0.0]
  - 'U' - 
  - 't' -
  - 'yperiodic' -
"""
function construct_psi_h(
  name::String;
  N=nothing,
  Nx=nothing,
  Ny=nothing,
  conserve_qns=nothing,
  conserve_pn=nothing,
  nsweeps=nothing,
  cutoff=nothing,
  maxdim=nothing,
  noise=nothing,
  U=nothing,
  t=nothing,
  yperiodic=nothing,
  model=nothing,
)
  if name == "one_d_heisenberg"
    (E, ψ, H) = compute_1d_heisenberg(
      N;
      conserve_qns=conserve_qns,
      nsweeps=nsweeps,
      maxdim=maxdim,
      cutoff=cutoff,
      noise=noise,
    )
  elseif name == "two_d_hubbard"
    (E, ψ, H) = compute_2d_hubbard(
      Nx,
      Ny;
      conserve_qns=conserve_qns,
      U=U,
      t=t,
      nsweeps=nsweeps,
      maxdim=maxdim,
      cutoff=cutoff,
      noise=noise,
      yperiodic=yperiodic,
      model=model,
    )
  else
    error("$(model) is not a valid model name. Please refer to documentation.")
  end

  return ψ, H
end
