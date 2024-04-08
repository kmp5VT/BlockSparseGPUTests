## A small set of scripts to construct and time DMRG calculations
For testing the ITensors GPU and backend performance using realistic DMRG systems.
So far there is the `one_d_heisenberg` model and the `two_d_hubbard` model with 
different levels of symmetry. 
The code boots up and constructs a DMRG MPO and MPS based on specifications of 
the model, the number of sites, the bond dimensions, etc...
There are two modes of testing are the contraction of the fully contracted LHS MPS/MPO chain with a contracted two-site tensor


$$
\sum_\chi \left( \langle \Psi_{1,...,j-1} | H_{1,...,j-1} | \Psi_{1,...,j-1} \rangle \right)^\chi _{\chi '}$$ 

$$\left( |\Psi_{j}\rangle |\Psi_{j+1} \rangle) \right)^{a\chi}_{b\chi ''}
$$

and the SVD of the two-site tensor

$$
\left( |\Psi_{j}\rangle |\Psi_{j+1} \rangle) \right)^{a\chi}_{b\chi ''}
$$

$$

= (|\Psi_{j}\rangle)^{a\chi}_{P} (|\Psi_{j+1} \rangle)^{P}_{b\chi}
$$

where $a$ and $b$ are site indices.
