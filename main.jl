using CUDA
using NDTensors
using TimerOutputs
include("src/BlockSparseGPUTests.jl")

begin
  timer = TimerOutput()
  LHS, twosite = BlockSparseGPUTests.test_one_d_heisenberg(; N=100, conserve_qns=false, timer=timer, timer_string_contract="cpu: Dense LHS * 2site",nrepeat_contract=1000);
  BlockSparseGPUTests.test_one_d_heisenberg(NDTensors.cu; twosite = twosite, LHS = LHS, nrepeat_contract=1000, conserve_qns=false, timer = timer, timer_string_contract="cuF64: Dense LHS * 2site");
  BlockSparseGPUTests.test_one_d_heisenberg(cu; twosite = twosite, LHS = LHS, nrepeat_contract=1000, conserve_qns=false, timer = timer, timer_string_contract="cuF32: Dense LHS * 2site");
  println("Printing twosite indices")
  BlockSparseGPUTests.easyprint(twosite)
  println("Printing the indices for LHS")
  BlockSparseGPUTests.easyprint(LHS)
  println()
  
  LHS, twosite = BlockSparseGPUTests.test_one_d_heisenberg(; N=100, conserve_qns=true, timer=timer, timer_string_contract="cpu: BS LHS * 2site",nrepeat_contract=1000);
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

ψ,H = BlockSparseGPUTests.construct_psi_h("two_d_hubbard";nsweeps=4)

timer = TimerOutput()
LHSD, twositeD = BlockSparseGPUTests.test_two_d_hubbard(; nrepeat_contract=1000, conserve_qns=false, timer=timer, timer_string_contract="CPU Dense LHS * 2site", nsweeps=4);
LHSS, twositeS = BlockSparseGPUTests.test_two_d_hubbard(; nrepeat_contract=1000, conserve_qns=true, timer=timer, timer_string_contract="CPU Sparse LHS * 2site", nsweeps=4);
BlockSparseGPUTests.test_two_d_hubbard(NDtensors.cu; twosite=twositeD, LHS = LHSD, nrepeat_contract=1000, conserve_qns=true, timer=timer, timer_string_contract="CU F64 Sparse LHS * 2site", nsweeps=4);
BlockSparseGPUTests.test_two_d_hubbard(NDtensors.cu; twosite=twositeS, LHS = LHSS, nrepeat_contract=1000, conserve_qns=true, timer=timer, timer_string_contract="CU F64 Sparse LHS * 2site", nsweeps=4);
timer

Block
ITensors.eigsolve()