include("../src/BlockSparseGPUTests.jl")
using Documenter, .BlockSparseGPUTests

DocMeta.setdocmeta!(
  BlockSparseGPUTests, :DocTestSetup, :(using .BlockSparseGPUTests); recursive=true
)

sitename = "BlockSparseGPUTests.jl"

settings = Dict(
  :modules => [BlockSparseGPUTests],
  :pages => [
    "Introduction" => "index.md",
    "Documentation" => [
      "Computing optimized MPS wavefunctions" => "example_systems.md",
      "Performance representative tests" => "tests.md",
      "ITensor analysis" => "analysis.md",
    ],
    #   "Installing Julia and ITensor" => "getting_started/Installing.md",
    #   "Running ITensor and Julia Codes" => "getting_started/RunningCodes.md",
    #   "Enabling Debug Checks" => "getting_started/DebugChecks.md",
    #   "Next Steps" => "getting_started/NextSteps.md",
    # ],
    # "Tutorials" => [
    #   "DMRG" => "tutorials/DMRG.md",
    #   "Quantum Number Conserving DMRG" => "tutorials/QN_DMRG.md",
    #   "MPS Time Evolution" => "tutorials/MPSTimeEvolution.md",
    # ],
  ],
  :format => Documenter.HTML(; prettyurls=false),
  :doctest => true,
  :checkdocs => :none,
)
