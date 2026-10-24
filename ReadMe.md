# tangle

Tools for measuring how a curve winds around itself, mainly applied to protein (and RNA) backbones. The core is a fast Cython implementation of the writhe, with modules built on top for comparing conformations, smoothing chains and working with trajectories.

The repository was previously called Wiggle, and the package is still imported as `wiggle` for now.

## Install

```
git clone https://github.com/mckeownish/tangle.git
cd tangle
pip install -e .
```

The writhe extension is compiled with OpenMP and `-march=native`, so build it on the machine you run it on.

## Modules

### writhe
Cython implementation of the writhe between segments of a discrete curve. `find_Sigma_array` gives the pairwise writhe for every segment pair of one curve, and `find_Sigma_array_batch_parallel` does the same for a whole trajectory using OpenMP threads.

### fingerprint
Sub-chain writhe fingerprints and pyramids for comparing two conformations of the same chain.

The fingerprint F(i, j) is the writhe of the sub-chain running from segment i to segment j. Taking the difference between two structures and arranging it by window centre and window length gives a pyramid that shows where along the chain the winding changed, and over what length.

It also includes:

- iterative midpoint smoothing of the chain, to remove local structure (helices) before comparing
- a check that smoothing has not passed the chain through itself (each corner cut is tested against the rest of the chain, as in KMT reduction)
- a check for strand passages between two conformations, reporting which segments passed through each other

```python
from wiggle import fingerprint as fp

cmp = fp.compare('state_A.pdb', 'state_B.pdb', m=7)    # PDB paths or (N, 3) arrays, m = smoothing level
fp.plot_pyramid(cmp, ells=[10, 20])                    # pyramid of the change in F/l
fp.plot_cross_section(cmp, ell=10).show()              # both structures coloured by the l = 10 row (plotly)
fp.write_bfactor_pdb('B_l10.pdb', cmp.resid, cmp.Xb, cmp.cross_section(10))   # values in the B-factor column

F = fp.fingerprint(traj_xyz, m=7, n_threads=8)         # trajectories, shape (T, N, 3)
counts, where = fp.smoothing_passages(X, m_max=15)      # any non-zero count means smoothing broke the chain
hits = fp.strand_passages(X0, X1)                       # segment pairs crossing on the path from X0 to X1
```

`compare` works on the residues present in both structures and keeps the longest stretch without gaps. Use `chain_a` / `chain_b` to pick chains, `atom="C4'"` (or another atom name) for non-protein traces, and `max_step=None` to switch off the gap check for coarse-grained models.

Tests: `python -m pytest tests/test_fingerprint.py`

### writhe_evo
Helper functions for working with the writhe: visualisation, trajectory analysis and simple comparisons.

### CarbonaraDataTools
Input/output for Carbonara (SAXS-based protein modelling): extracting coordinates, secondary structure and sequences, writing PDB files, and inferring approximate CB positions ahead of all-atom reconstruction.

### CA_2_AA
Rebuilds an approximate all-atom structure (side chains and full backbone) from a C-alpha trace, written out as a PDB. The reconstruction uses MODELLER and includes energy minimisation of the backbone, so the C-alpha positions can shift slightly.
