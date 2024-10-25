# Information about folders

Each of the folders here represents a DMRG calculation with a certain number of "Symmetry Sectors". These sectors are
1. Ky
2. Sz
3. Nf
4. Nf parity

The combination of multiple symmetries makes for more blocks (equivalently this means that each block is smaller).
The order from least to most number of blocks is therefore
1. symm_nfparity
2. symm_nf
3. symm_sznf
4. symm_kysznf

Inside of each of these folders are 5 different contractions.The naming of these contractions follows the convention from the [notes folder](https://github.com/kmp5VT/BlockSparseGPUTests/blob/main/notes/DMRG_Contractions.pdf)
They are named
1. E1
2. E2
3. S1
4. S2
5. S3

Each of these 5 folders holds information about the tensor (non-matricized) contraction which can be easily read into a C file.
