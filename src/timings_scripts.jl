using ITensors, NDTensors, TimerOutputs
using ITensors.ITensorMPS
# using BenchmarkTools: @btime

"""
    timing_contract(ψ, H; kwargs...)

Required Arguments:
  - T1 - An optimized MPS
  - H - An MPO Hamiltonian
  - 'nrepeat::Integer' - The number of times the contraction code will be repeated for benchmarking purposes; 10 
  - verbose::Bool - Print information about what the code is doing; false
  - timer::TimerOutput - TimerOutput object which is used for benchmarking
  - timer_string - The description string for the timer; "LHS with two-site"
"""
function timing_contract(
  T1::ITensor,
  T2::ITensor;
  nrepeat=10,
  verbose=false,
  timer=TimerOutput(),
  timer_string::String="LHS with two-site",
)
  ## Compile contract
  T1 * T2
  ## benchtools maybe to force contraction
  for i in 1:nrepeat
    @timeit timer timer_string begin
      T1 * T2
    end
  end

  if verbose
    print_timer(timer; sortby=:name, compact=false, allocations=false)
  end
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
  - TN - twosite tensor from site `N` and `N+1` which will be decomposed
  - N - Site of `ψ` of the twosite tensor

Optional Keyword Arguments

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
  TN,
  N;
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
