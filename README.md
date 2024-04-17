
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
  4. The `summarize_itensor` function can be used to extract easy to read and useful information about tensors, indices and index block sizes.
  5. In the future will create functions that allow users to pluck the representative out of a DMRG optimization at a specific sweep and 
  at a specific site.
  6. In the future, making an easy function to extract the block data from the ITensors and write them to a dense Array.


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

You can find the HDF5 tensor network files in `runnable_examples/hdf5/SIZE` where `SIZE= ["small", "medium", "large"]`. Each tensor network in these folders corresponds to a tensor diagram from the `notes/DMRG_contraction` PDF and are the filenames correspond to the labels in this document.  A single HDF5 file can be read in simply using 

```julia
julia> using HDF5, ITensors
julia> fid = h5open("/path/to/tensor/network")
julia> RHS_tensor = read(fid, "T1", ITensor)
julia> LHS_tensor = read(fid, "T2", ITensor)
julia> close(fid)
```

In all the files, the right hand side tensor from the tensor network diagram is stored as "T1" and the left hand side tensor from the tensor network diagram is stored as "T2".
The tensor network contraction can simply be constructed using the `*` operation, i.e.

```julia
julia> output = RHS_tensor * LHS_tensor
```

and ones favorite benchmark tooling can be used to profile the tensors, for example

```julia
julia> @time RHS_tensor * LHS_tensor
```

There also exists a simple script for running all networks from a single folder in `runnable_examples/example_timings.jl`

There is also a simple interface to inspect the information about a single tensor 

```julia
julia> fid = h5open("runnable_examples/hdf5/small/sparse/S1.h5")
julia> T1 = read(fid, "T1", ITensor)
julia> close(fid)
julia> summarize_itensor(T1)
Order-3 Tensor
        Index 1:
                (dim=4|id=845|tags="Electron,Site,n=3"|dir=Neither)
        Index 2:
                (dim=55|id=100|tags="Link,l=3"|dir=Neither)
        Index 3:
                (dim=16|id=451|tags="Link,l=2"|dir=Neither)

julia> summarize_itensor(T1; outputlevel=1)
Order-3 Tensor
        Index 1:
                (dim=4|id=845|tags="Electron,Site,n=3"|dir=Neither)
                        blockdims:[1, 1, 1, 1]
        Index 2:
                (dim=55|id=100|tags="Link,l=3"|dir=Neither)
                        blockdims:[1, 3, 2, 3, 7, 3, 1, 8, 7, 1, 3, 7, 3, 3, 2, 1]
        Index 3:
                (dim=16|id=451|tags="Link,l=2"|dir=Neither)
                        blockdims:[1, 2, 2, 1, 4, 1, 2, 2, 1]

julia> summarize_itensor(T1; outputlevel=2)
Order-3 Tensor
        Index 1:
                (dim=4|id=845|tags="Electron,Site,n=3"|dir=Neither)
                        blockdims:[1, 1, 1, 1]
        Index 2:
                (dim=55|id=100|tags="Link,l=3"|dir=Neither)
                        blockdims:[1, 3, 2, 3, 7, 3, 1, 8, 7, 1, 3, 7, 3, 3, 2, 1]
        Index 3:
                (dim=16|id=451|tags="Link,l=2"|dir=Neither)
                        blockdims:[1, 2, 2, 1, 4, 1, 2, 2, 1]
          length of block Block(4, 1, 1) is 3
          length of block Block(2, 2, 1) is 3
          ...
```

## Full Example Codes

It is also possible to run ones own models using 