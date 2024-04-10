using CUDA, Adapt
using ITensors, NDTensors
using TimerOutputs
include("src/BlockSparseGPUTests.jl")

adapt32(ten::ITensor) = adapt(Float32, ten)
begin
  timer = TimerOutput()
  LHS, twosite = BlockSparseGPUTests.test_one_d_heisenberg(; N=100, conserve_qns=false, timer=timer, timer_string_contract="cpuF64: Dense LHS * 2site",nrepeat_contract=1000);
  BlockSparseGPUTests.test_one_d_heisenberg(adapt32; twosite = twosite, LHS = LHS, nrepeat_contract=1000, conserve_qns=false, timer = timer, timer_string_contract="cpuF32: Dense LHS * 2site");
  BlockSparseGPUTests.test_one_d_heisenberg(NDTensors.cu; twosite = twosite, LHS = LHS, nrepeat_contract=1000, conserve_qns=false, timer = timer, timer_string_contract="cuF64: Dense LHS * 2site");
  BlockSparseGPUTests.test_one_d_heisenberg(cu; twosite = twosite, LHS = LHS, nrepeat_contract=1000, conserve_qns=false, timer = timer, timer_string_contract="cuF32: Dense LHS * 2site");
  println("Printing twosite indices")
  BlockSparseGPUTests.easyprint(twosite)
  println("Printing the indices for LHS")
  BlockSparseGPUTests.easyprint(LHS)
  println()
  
  LHS, twosite = BlockSparseGPUTests.test_one_d_heisenberg(; N=100, conserve_qns=true, timer=timer, timer_string_contract="cpuF64: BS LHS * 2site",nrepeat_contract=1000);
  BlockSparseGPUTests.test_one_d_heisenberg(adapt32; twosite = twosite,nrepeat_contract=1000, LHS = LHS, conserve_qns=true, timer = timer, timer_string_contract="cpuF32: BS LHS * 2site");
  BlockSparseGPUTests.test_one_d_heisenberg(NDTensors.cu; twosite = twosite,nrepeat_contract=1000, LHS = LHS, conserve_qns=true, timer = timer, timer_string_contract="cuF64: BS LHS * 2site");
  BlockSparseGPUTests.test_one_d_heisenberg(cu; twosite = twosite,nrepeat_contract=1000, LHS = LHS, conserve_qns=true, timer = timer, timer_string_contract="cuF32: BS LHS * 2site");
  println("Printing twosite indices")
  BlockSparseGPUTests.easyprint(twosite)
  println("Printing the indices for LHS")
  BlockSparseGPUTests.easyprint(LHS)
  println()
  @show timer
  return nothing
end

begin
timer = TimerOutput()
LHSD, twositeD = BlockSparseGPUTests.test_two_d_hubbard(; nrepeat_contract=1000, conserve_qns=false, timer=timer, timer_string_contract="CPUF64 LHS * 2site", nsweeps=6);
LHSS, twositeS = BlockSparseGPUTests.test_two_d_hubbard(; nrepeat_contract=1000, conserve_qns=true, timer=timer, timer_string_contract="CPUF64 BS LHS * 2site", nsweeps=6);

BlockSparseGPUTests.test_two_d_hubbard(adapt32; twosite=twositeD, LHS = LHSD, nrepeat_contract=1000, conserve_qns=false, timer=timer, timer_string_contract="CPUF32 LHS * 2site", nsweeps=1);
BlockSparseGPUTests.test_two_d_hubbard(NDtensors.cu; twosite=twositeD, LHS = LHSD, nrepeat_contract=1000, conserve_qns=false, timer=timer, timer_string_contract="CUF64 LHS * 2site", nsweeps=1);
BlockSparseGPUTests.test_two_d_hubbard(cu; twosite=twositeD, LHS = LHSD, nrepeat_contract=1000, conserve_qns=false, timer=timer, timer_string_contract="CUF32 LHS * 2site", nsweeps=1);

BlockSparseGPUTests.test_two_d_hubbard(adapt32; twosite=twositeS, LHS = LHSS, nrepeat_contract=1000, conserve_qns=true, timer=timer, timer_string_contract="CPUF32 BS LHS * 2site", nsweeps=1);
BlockSparseGPUTests.test_two_d_hubbard(NDtensors.cu; twosite=twositeS, LHS = LHSS, nrepeat_contract=1000, conserve_qns=true, timer=timer, timer_string_contract="CUF64 BS LHS * 2site", nsweeps=1);
BlockSparseGPUTests.test_two_d_hubbard(cu; twosite=twositeS, LHS = LHSS, nrepeat_contract=1000, conserve_qns=true, timer=timer, timer_string_contract="CUF32 BS LHS * 2site", nsweeps=1);

@show timer
return nothing
end

ψ, H = BlockSparseGPUTests.construct_psi_h(
      "two_d_hubbard"; conserve_qns=false, nsweeps=4)
using Adapt

LHS, twosite = BlockSparseGPUTests.test_one_d_heisenberg(; N=100, conserve_qns=false, timer=timer, timer_string_contract="cpu: Dense LHS * 2site",nrepeat_contract=1000);
BlockSparseGPUTests.test_one_d_heisenberg(adapt32; twosite = twosite, LHS = LHS, nrepeat_contract=1000, conserve_qns=false, timer = timer, timer_string_contract="cpuF32: Dense LHS * 2site");
timer