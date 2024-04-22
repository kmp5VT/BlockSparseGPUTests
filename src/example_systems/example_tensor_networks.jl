function make_all_tensor_networks(ψ, H, site)
  EL11, EL12 = make_EL1_network(ψ, H, site)
  EL21, EL22 = make_EL2_network(EL11, EL12, H, site)

  ## Components of the two site tensor
  S11 = ψ[site]
  S12 = ψ[site + 1]

  ## Contract the twosite with the environment tensor
  S21, S22 = make_S2_network(EL21, ψ, S11, S12, site)

  S31, S32 = make_S3_network(ψ, H, S22, site)

  return (EL11, EL12), (EL21, EL22), (S11, S12), (S21, S22), (S31, S32)
end

function make_EL1_network(ψ, H, site)
  ## Create the LHS environment and grab the site -2 environment
  L = BlockSparseGPUTests.left_environment_tensor(ψ, H; j=site)[site - 2]
  ## Grab the next MPS from ψ*
  EL12 = dag(ψ[site - 1])'

  return L, EL12
end

function make_EL2_network(EL11, EL12, H, site)
  EL21 = EL11 * EL12
  EL22 = H[site - 1]
  return EL21, EL22
end

function make_S2_network(EL21, ψ, S11, S12, site)
  S21 = EL21 * ψ[site - 1]
  S22 = S11 * S12
  return S21, S22
end

function make_S3_network(ψ, H, S22, site)
  S31 = S22 * H[site] * H[site + 1]
  S32 = BlockSparseGPUTests.right_environment_tensor(ψ, H; j=site)[site + 2]
  return S31, S32
end
