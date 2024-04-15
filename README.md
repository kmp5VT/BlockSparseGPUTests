
# BlockSparseGPUTests.jl

BlockSparseGPUTests is a library for testing the performance of different ITensor based
tensor network algorithms.

The source code for ITensor can be found [on Github](https://github.com/kmp5VT/BlockSparseGPUTests).

Additional documentation for ITensor can be found on the ITensor website [itensor.org](https://itensor.org/).

Development of ITensor is supported by the Flatiron Institute, a division of the Simons Foundation.

## Installation

The ITensors package can be installed with git only, it is not a registered Julia package.
From the Julia REPL, type `]` to enter the Pkg REPL mode and run:

```
~ git clone https://github.com/kmp5VT/BlockSparseGPUTests
~ cd BlockSparseGPUTests
~ julia
```

```julia
julia> include("src/BlockSparseGPUTests")
```

## A small set of scripts to construct and time DMRG calculations
For testing the ITensors GPU and backend performance using realistic DMRG systems.

Direction/goal of the library:
  1. Representative small, medium and large tensor networks can be extracted from HDF5 files in `runnable_examples/hdf5`
  2. The file `runnable_examples/time_contractions.jl` has easy to follow instructions on how to 
  grab the HDF5 tensor networks and time the contractions of the networks.
  3. There exists an examples on how to adapt the tensor networks for testing with different percisions and GPU backends.
  4. The easyprint function can be used to extract easy to read and useful information about tensors, indices and index block sizes.
  5. In the future will create functions that allow users to pluck the representative out of a DMRG optimization at a specific sweep and 
  at a specific site.


<!-- So far there is the `one_d_heisenberg` model and the `two_d_hubbard` model with 
different levels of symmetry. 
The code boots up and constructs a DMRG MPO and MPS based on specifications of 
the model, the number of sites, the bond dimensions, etc...
There are two modes of testing are the contraction of the fully contracted LHS MPS/MPO chain with a contracted two-site tensor


```math
\sum_\chi \left( \langle \Psi_{1,...,j-1} | H_{1,...,j-1} | \Psi_{1,...,j-1} \rangle \right)^\chi _{\chi '}
\left( |\Psi_{j}\rangle |\Psi_{j+1} \rangle) \right)^{a\chi}_{b\chi ''}
```

and the SVD of the two-site tensor

```math
\left( |\Psi_{j}\rangle |\Psi_{j+1} \rangle) \right)^{a\chi}_{b\chi ''}
= (|\Psi_{j}\rangle)^{a\chi}_{P} (|\Psi_{j+1} \rangle)^{P}_{b\chi}
```

where $a$ and $b$ are site indices.\\
TODO: Add pictoral diagrams to graphically show what the decompositions/contractions look like -->

## Documentation


## Full Example Codes

