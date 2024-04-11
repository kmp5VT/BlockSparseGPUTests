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
  model::String;
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
)
  if model == "one_d_heisenberg"
    (E, ψ, H) = compute_1d_heisenberg(
      N;
      conserve_qns=conserve_qns,
      nsweeps=nsweeps,
      maxdim=maxdim,
      cutoff=cutoff,
      noise=noise,
    )
  elseif model == "two_d_hubbard"
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
    )
  else
    error("$(model) is not a valid model name. Please refer to documentation.")
  end

  return ψ, H
end

## Setting the parameters to nothing allows one to use the defuault values which are defined in the 
## file for the model, i.e. 1d_heisenbrg
"""
```julia
test_1d_heisenberg(dev::Function=NDTensors.cpu; kwargs...)
```

A function to easily construct a 1d Heisenberg wavefunction and Hamiltonian then run representative
contraction and decomposition timings on the model.
The representative tensors `LHS` and `twosite` can be adapted by setting the function `dev` short for device.
`dev`'s purpose could be thought of as a way to move these ITensors from CPU to a GPU device.
This function uses the `TimerOutputs` package to benchmark the code and to see the results either set `print_timer=true`
or pass a `timer` to the function. 

Returns:

  - 'twosite' - The twosite tensor
  - 'LHS' - The left hand side of the DMRG optimization problem.

Arguments

  - 'dev::Function' - Adaptor function which modifies `LHS` and `twosite` tensors; NDTensors.cpu

Optional Arguments

  - 'N::Integer' - Number of sites
  - 'twosite::ITensor' - The two-site tensor which is being used for benchmarking
  - 'LHS::ITensor' - The left hand side of the DMRG problem.
  - 'site::Integer' - The site in the MPS wavefunction which will be used for the two-site tensor
  - 'nrepeat_contract' - The number of times the representative contraction will be computed
  - 'print_sites' - option to print the indices of `twosite` and `LHS`
  - 'conserve_qns' - Compute with BlockSparse tensor storage by preserving the quantum number symmetry
  - 'nsweeps' - DMRG number of sweeps option
  - 'cutoff' - DMRG eigenvalue cutoff option
  - 'maxdim' - DMRG max bond dimension option
  - 'noise' - DMRG least squares noise option
  - 'timer::TimerOutput' - Performance timer
  - 'timer_string_contract::String' - Timer description string for twosite and LHS contraction testing, "dev: LHS * 2-site"
  - 'print_timer::Bool' - option to print the timer at the end of computation; false
"""
function test_1d_heisenberg(
  dev::Function=NDTensors.cpu;
  N=5,
  twosite=nothing,
  LHS=nothing,
  site=nothing,
  nrepeat_contract=10,
  print_sites=false,
  conserve_qns=nothing,
  nsweeps=nothing,
  cutoff=nothing,
  maxdim=nothing,
  noise=nothing,
  timer=TimerOutput(),
  timer_string_contract="$(dev): LHS * 2-site",
  print_timer=false,
)
  ψ = H = nothing
  if isnothing(site)
    site = trunc(Int, N / 2)
    println("Choosing site N / 2: $(site)")
  end

  if isnothing(twosite) || isnothing(LHS)
    println("Constructing the MPS and MPO on CPU")
    ψ, H = construct_psi_h(
      "one_d_heisenberg"; N=N, conserve_qns=conserve_qns, nsweeps, cutoff, maxdim, noise
    )

    if print_sites
      println("The wavefunction at site $(site) has this block structure")
      easyprint(ψ[site])
    end

    println("Moving to device using function: $(dev)")
    ψ, H = dev.((ψ, H))

    println("Running contraction testing")
    twosite, LHS = representative_contract_timing(ψ, H; N=site, nrepeat=1)
  else
    println("Moving LHS and twosite to device using $(dev)")
    twosite, LHS = dev.((twosite, LHS))
    representative_contract_timing(ψ, H; N=site, nrepeat=1, twosite=twosite, LHS=LHS)
  end
  representative_contract_timing(
    ψ,
    H;
    N=site,
    nrepeat=nrepeat_contract,
    twosite=twosite,
    LHS=LHS,
    timer=timer,
    timer_string=timer_string_contract,
  )

  ## TODO add in the SVD testing options, run with all 3 SVD options
  #println("Running SVD testing")
  # representative_svd_timing(ψ, H; N=site,nrepeat=nrepeat_svd)
  if print_timer
    @show timer
  end
  return LHS, twosite
end

"""
```julia
test_2d_hubbard(dev::Function; kwargs...)
```

A function to easily construct a 2d Hubbard wavefunction and Hamiltonian then run representative
contraction and decomposition timings on the model.
The representative tensors `LHS` and `twosite` can be adapted by setting the function `dev` short for device.
`dev`'s purpose could be thought of as a way to move these ITensors from CPU to a GPU device.
This function uses the `TimerOutputs` package to benchmark the code and to see the results either set `print_timer=true`
or pass a `timer` to the function. 

Returns:

  - 'twosite' - The twosite tensor
  - 'LHS' - The left hand side of the DMRG optimization problem.

Arguments

  - 'dev::Function' - Adaptor function which modifies `LHS` and `twosite` tensors; NDTensors.cpu

Optional Arguments

  - 'Nx::Integer' - Number of sites in the x mode
  - 'Ny::Integer' - Number of sites in the y mode
  - 'twosite::ITensor' - The two-site tensor which is being used for benchmarking
  - 'LHS::ITensor' - The left hand side of the DMRG problem.
  - 'site::Integer' - The site in the MPS wavefunction which will be used for the two-site tensor
  - 'nrepeat_contract' - The number of times the representative contraction will be computed
  - 'print_sites' - option to print the indices of `twosite` and `LHS`
  - 'conserve_qns' - Compute with BlockSparse tensor storage by preserving the quantum number symmetry
  - 'U' - U
  - 't' - t
  - 'nsweeps' - DMRG number of sweeps option
  - 'cutoff' - DMRG eigenvalue cutoff option
  - 'maxdim' - DMRG max bond dimension option
  - 'noise' - DMRG least squares noise option
  - 'yperiodic' - yperiodic
  - 'timer::TimerOutput' - Performance timer
  - 'timer_string_contract::String' - Timer description string for twosite and LHS contraction testing, "dev: LHS * 2-site"
  - 'print_timer::Bool' - option to print the timer at the end of computation; false
"""
function test_2d_hubbard(
  dev::Function=NDTensors.cpu;
  Nx=nothing,
  Ny=nothing,
  twosite=nothing,
  LHS=nothing,
  site=nothing,
  nrepeat_contract=10,
  print_sites=false,
  conserve_qns=nothing,
  U=nothing,
  t=nothing,
  nsweeps=nothing,
  maxdim=nothing,
  cutoff=nothing,
  noise=nothing,
  yperiodic=nothing,
  timer=TimerOutput(),
  timer_string_contract="$(dev): LHS * 2-site",
  print_timer::Bool=false,
)
  ψ = H = nothing
  if isnothing(twosite) || isnothing(LHS)
    println("Constructing the MPS and MPO on CPU")
    ψ, H = construct_psi_h(
      "two_d_hubbard";
      Nx=Nx,
      Ny=Ny,
      conserve_qns=conserve_qns,
      nsweeps=nsweeps,
      cutoff=cutoff,
      maxdim=maxdim,
      noise=noise,
      yperiodic=yperiodic,
      U=U,
      t=t,
    )

    if isnothing(site)
      site = trunc(Int, length(ψ) / 2)
      println("Choosing site (Nx * Ny) / 2: $(site)")
    end

    if print_sites
      println("The wavefunction at site $(site) has this block structure")
      easyprint(ψ[site])
    end

    println("Moving to device using function: $(dev)")
    ψ, H = dev.((ψ, H))

    println("Running contraction testing")
    twosite, LHS = representative_contract_timing(ψ, H; N=site, nrepeat=1)
  else
    site = 1
    println("Moving LHS and twosite to device using $(dev)")
    twosite, LHS = dev.((twosite, LHS))
    representative_contract_timing(ψ, H; N=site, nrepeat=1, twosite=twosite, LHS=LHS)
  end
  representative_contract_timing(
    ψ,
    H;
    N=site,
    nrepeat=nrepeat_contract,
    twosite=twosite,
    LHS=LHS,
    timer=timer,
    timer_string=timer_string_contract,
  )

  ## TODO working on best way to write SVD testing
  # println("Running SVD testing")

  if print_timer
    @show timer
  end
  return LHS, twosite
end
