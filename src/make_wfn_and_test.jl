using NDTensors
## Model is a string with option of 
# one_d_heisenberg
# two_d_hubbard
function construct_psi_h(
  model::String;
  N=5,
  conserve_qns=nothing,
  conserve_pn=nothing,
  nsweeps=nothing,
  cutoff=nothing,
  maxdim=nothing,
  noise=nothing,
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
  else
    error("$(model) is not a valid model name. Please refer to documentation.")
  end

  return ψ, H
end

function test_one_d_heisenberg(
  dev=NDTensors.cpu;
  site=nothing,
  nrepeat_contract=10,
  print_sites=false,
  N=5,
  conserve_qns=nothing,
  nsweeps=nothing,
  cutoff=nothing,
  maxdim=nothing,
  noise=nothing,
)
  println("Making the wavefunctions")
  ψ, H = construct_psi_h(
    "one_d_heisenberg"; N=N, conserve_qns=conserve_qns, nsweeps, cutoff, maxdim, noise,
  )

  if isnothing(site)
    site = Int(length(ψ) / 2)
    println("Choosing site $(site) which is in the middle of ψ")
  end

  if print_sites
    println("The wavefunction at site $(site) has this block structure")
    easyprint(ψ[site])
  end

  println("Moving to device using function $(dev)")
  ψ, H = dev.((ψ, H))

  println("Running contraction testing")
  twosite, LHS, result = representative_contract_timing(ψ, H; N=site, nrepeat=0, time=false)
  # representative_contract_timing(
  #   ψ, H; N=site, nrepeat=nrepeat_contract, twosite=twosite, LHS=LHS
  # )

  println("Running SVD testing")
  return nothing
end
