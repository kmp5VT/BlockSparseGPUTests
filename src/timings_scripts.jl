using ITensors, NDTensors, TimerOutputs
using ITensors.ITensorMPS
# using BenchmarkTools: @btime

"""
```julia
  representative_contract_timing(ψ, H; kwargs...)
```

A function which takes an optimized wavefunction ψ and Hamiltonian and constructs a two-site tensor with sites `N` and `N+1`
and forms the LHS of the DMRG problem using the `1` to `N-1` previous sites.
Using the two-site tensor and the LHS performance timing is done on this DMRG representative contraction.
If the optional arguments `LHS` and `twosite` are provided the code will skip the process of 
constructing these objects and, therefore, ignore the argument `N`. 
This choice has been made to make it easier to rerun timings with different options.
At the moment this code uses the `TimerOutputs` package for benchmarking and users need to 
pass a `timer` or mark `verbose=true` to inspect the output.

Returns:

  - twosite - The optimized MPS
  - LHS - The eigenvalues of the optimized MPS
  
Required Arguments

  - ψ - An optimized MPS
  - H - An MPO Hamiltonian

Optional Keyword Arguments

  - 'N::Integer' - The site which will be used to construct the two-site tensor (N and N+1 are used) so please choose N ≤ length(ψ) - 1
  - 'nrepeat::Integer' - The number of times the contraction code will be repeated for benchmarking purposes; 10
  - 'twosite::ITensor' - The two-site tensor which is being contracted: nothing
  - 'LHS' - The left hand side of the DMRG problem; nothing
  - 'verbose::Bool' - Print information about what the code is doing; false
  - 'timer::TimerOutput' - TimerOutput object which is used for benchmarking
  - 'timer_string' - The description string for the timer; "LHS with two-site"
"""
function representative_contract_timing(
  ψ,
  H;
  N=nothing,
  nrepeat=10,
  twosite=nothing,
  LHS=nothing,
  verbose=false,
  timer=TimerOutput(),
  timer_string::String="LHS with two-site",
)
  if isnothing(twosite)
    N = (isnothing(N) ? Int(length(ψ) / 2) : N)
    ## Given a psi grab the middle and middle plus one and contract
    if N == length(ψ)
      error("You cannot choose the last site to run test")
    end

    if verbose
      println("Orthogonalizing ψ to site $(N)")
    end
    orthogonalize!(ψ, N)
    if verbose
      println("Constructing the two-site tensor for site $(N) and $(N + 1)")
    end
    twosite = ψ[N] * ψ[N + 1]
    if verbose
      println("Dimensions of two-site tensor")
    end
  else
    if verbose
      println("Using the provided two-site tensor")
    end
  end
  if verbose
    easyprint(twosite)
    println()
  end

  ## contract to form LHS
  if isnothing(LHS)
    if verbose
      println("Forming the LHS tensor to contract with two-site tensor")
    end
    LHS = dag(ψ'[1]) * H[1] * (ψ[1])
    for i in 2:(N - 1)
      LHS = ((LHS * dag(ψ'[i])) * H[i]) * (ψ[i])
    end
  else
    if verbose
      println("Using the provided LHS tensor")
    end
  end

  if verbose
    println("Dimension of LHS tensor")
    easyprint(LHS)
    println()
    println("Starting the timer for contracting LHS with two-site tensor")
  end

  ## benchtools maybe to force contraction
  for i in 1:nrepeat
    @timeit timer timer_string begin
      LHS * twosite
    end
  end
  if verbose
    println("Timing for the contractions:\n$(timer)")
  end

  return twosite, LHS
end

"""
```julia
representative_svd_timing(ψ, H; kwargs...)
```

A function which takes an optimized wavefunction ψ and constructs a two-site tensor with sites `N` and `N+1`.
Performance timing is then done on a DMRG representative decomposition using ITensors eigsolve code.
If the optional argument `twosite` is provided the code will skip the process of 
constructing `twosite` and, therefore, ignore the argument `N`. 
This choice has been made to make it easier to rerun timings with different options.
At the moment this code uses the `TimerOutputs` package for benchmarking and users need to 
pass a `timer` or mark `verbose=true` to inspect the output.

Returns:

  - twosite - The optimized MPS
  
Required Arguments

  - ψ - An optimized MPS

Optional Keyword Arguments

  - 'N::Integer' - The site which will be used to construct the two-site tensor (N and N+1 are used) so please choose N ≤ length(ψ) - 1
  - 'nrepeat::Integer' - The number of times the contraction code will be repeated for benchmarking purposes; 10
  - 'twosite::ITensor' - The two-site tensor which is being contracted: nothing
  - 'verbose::Bool' - Print information about what the code is doing; false
  - 'timer::TimerOutput' - TimerOutput object which is used for benchmarking
  - 'timer_string::String' - The description string for the timer; "LHS with two-site"
  - 'which_decomp::String' - choose what kind of decomposition is used. Options: eigen, qr, svd

To see the full factorize source and Keywords see [ITensor's documentation](https://itensor.github.io/ITensors.jl/stable/ITensorType.html#LinearAlgebra.factorize-Tuple{ITensor,%20Vararg{Any}})
"""
function representative_svd_timing(
  ψ,
  H;
  N=nothing,
  nrepeat=10,
  twosite=nothing,
  verbose=false,
  timer=TimerOutput(),
  timer_string="eigsolve",
  mindim=nothing,
  maxdim=nothing,
  cutoff=nothing,
  ortho=nothing,
  which_decomp=nothing,
  eigen_perturbation=nothing,
  svd_alg=nothing,
  use_absolute_cutoff=nothing,
  use_relative_cutoff=nothing,
  min_blockdim=nothing,
)
  N = (isnothing(N) ? Int(length(ψ) / 2) : N)
  if isnothing(twosite)
    ## Given a psi grab the middle and middle plus one and contract
    if N == length(ψ)
      error("You cannot choose the last site to run test")
    end

    if verbose
      println("Orthogonalizing ψ to site $(N)")
    end
    orthogonalize!(ψ, N)
    if verbose
      println("Constructing the two-site tensor for site $(N) and $(N + 1)")
    end
    twosite = ψ[N] * ψ[N + 1]
    if verbose
      println("Dimensions of two-site tensor")
    end
  else
    if verbose
      println("Using the provided two-site tensor")
    end
  end
  if verbose
    easyprint(twosite)
    println()
  end

  ## benchtools maybe to force contraction
  indssite = inds(ψ[N])
  ITensors.factorize(
    twosite,
    indssite;
    mindim,
    maxdim,
    cutoff,
    ortho,
    which_decomp,
    eigen_perturbation,
    svd_alg,
    tags=tags(linkind(ψ, N)),
    use_absolute_cutoff,
    use_relative_cutoff,
    min_blockdim,
  )

  for i in 1:nrepeat
    @timeit timer timer_string begin
      ITensors.factorize(
        twosite,
        indssite;
        mindim,
        maxdim,
        cutoff,
        ortho,
        which_decomp,
        eigen_perturbation,
        svd_alg,
        tags=tags(linkind(ψ, N)),
        use_absolute_cutoff,
        use_relative_cutoff,
        min_blockdim,
      )
    end
  end
  if verbose
    println("Timing for the contractions:\n$(timer)")
  end

  return nothing
end
