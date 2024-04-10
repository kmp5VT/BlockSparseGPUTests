using NDTensors
using TimerOutputs: TimerOutput, @timeit
## Model is a string with option of 
# one_d_heisenberg
# two_d_hubbard
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
function test_one_d_heisenberg(
  dev=NDTensors.cpu;
  twosite=nothing,
  LHS=nothing,
  site=nothing,
  nrepeat_contract=10,
  print_sites=false,
  N=5,
  conserve_qns=nothing,
  nsweeps=nothing,
  cutoff=nothing,
  maxdim=nothing,
  noise=nothing,
  timer=TimerOutput(),
  timer_string_contract="$(dev): LHS * 2-site",
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

  println("Running SVD testing")

  #@show timer
  return LHS, twosite
end

function test_two_d_hubbard( dev = NDTensors.cpu;
  twosite=nothing,
  LHS=nothing,
  site=nothing,
  nrepeat_contract=10,
  print_sites=false,
  Nx = nothing,
  Ny = nothing,
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
  )
  ψ = H = nothing
  if isnothing(twosite) || isnothing(LHS)
    println("Constructing the MPS and MPO on CPU")
    ψ, H = construct_psi_h(
      "two_d_hubbard"; Nx=Nx, Ny=Ny, conserve_qns=conserve_qns, nsweeps, cutoff, maxdim, noise
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
    site=1
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

  println("Running SVD testing")

  #@show timer
  return LHS, twosite
end