"""
Sub-chain writhe fingerprints and pyramids.

For a curve with segments 0 … n-1 (segment k joins points k and k+1), the fingerprint is

    F[i, j] = sum_{i <= k < l <= j} W[k, l]

the writhe of the sub-chain running from segment i to segment j. Comparing two
conformations of the same chain through ΔF = F_B - F_A, arranged by window centre and
window length ℓ = j - i, gives the *pyramid*: where along the chain, and on what scale,
the winding differs.

Typical use
-----------
    from wiggle import fingerprint as fp

    cmp = fp.compare('state_A.pdb', 'state_B.pdb', m=7)      # or two (N, 3) arrays
    fp.plot_pyramid(cmp, ells=[10, 20])
    fp.plot_cross_section(cmp, ell=10).show()                # plotly, both structures coloured

Smoothing
---------
`smooth(X, m)` applies m passes of midpoint averaging (each pass removes one point). On an
m-smoothed curve, segment k sits at residue resid[0] + k + (m + 1) / 2.
Smoothing can itself pass the curve through itself; `smoothing_passages` checks for this.
"""

import os
import warnings
from dataclasses import dataclass, field

import numpy as np

from wiggle.writhe import find_Sigma_array, find_Sigma_array_batch_parallel

L_MIN = 3          # shortest window used in pyramids (ℓ = 1 is always 0, ℓ = 2 is a single pair)


# =============================================================================
# Loading
# =============================================================================

def load_ca(path, chain=None, atom='CA', model=1):
    """Residue numbers and coordinates (Å) of one atom per residue from a PDB file.

    chain : chain ID, or None for the first chain containing `atom`.
    atom  : atom name used for the trace ('CA' for proteins, e.g. "C4'" or 'P' for RNA).
    model : MODEL number to read (1-based); files without MODEL records are read whole.
    Alternate locations other than ' ' / 'A' are skipped; the first occurrence of a residue wins.
    """
    resid, xyz, seen = [], [], set()
    current_model, use_chain = 1, chain
    with open(path) as fh:
        for line in fh:
            rec = line[:6]
            if rec == 'MODEL ':
                current_model = int(line[10:14])
                continue
            if rec == 'ENDMDL' and current_model == model:
                break
            if rec not in ('ATOM  ', 'HETATM') or current_model != model:
                continue
            if line[12:16].strip() != atom or line[16] not in ' A':
                continue
            ch = line[21]
            if use_chain is None:
                use_chain = ch
            if ch != use_chain:
                continue
            key = (int(line[22:26]), line[26])
            if key in seen:
                continue
            seen.add(key)
            resid.append(key[0])
            xyz.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    if not xyz:
        raise ValueError(f'no {atom} atoms found in {path} (chain={chain}, model={model})')
    return np.array(resid), np.array(xyz)


def as_curve(x, chain=None, atom='CA'):
    """(resid, X) from a PDB path or an (N, 3) array (residues then numbered 1 … N)."""
    if isinstance(x, (str, bytes)) or hasattr(x, '__fspath__'):
        return load_ca(x, chain=chain, atom=atom)
    X = np.asarray(x, dtype=float)
    if X.ndim != 2 or X.shape[1] != 3:
        raise ValueError(f'expected an (N, 3) array, got shape {X.shape}')
    return np.arange(1, len(X) + 1), X


def common_ca(a, b, chain_a=None, chain_b=None, atom='CA', max_step=4.2, verbose=True):
    """Coordinates of the residues present in both curves, restricted to the longest
    gap-free stretch (consecutive residue numbers and no step > max_step Å in either).

    a, b : PDB paths or (N, 3) arrays. Arrays must already correspond residue by residue.
    Returns (resid, Xa, Xb).
    """
    ra, xa = as_curve(a, chain_a, atom)
    rb, xb = as_curve(b, chain_b, atom)
    ia, ib = {r: i for i, r in enumerate(ra)}, {r: i for i, r in enumerate(rb)}
    res = np.intersect1d(ra, rb)
    if len(res) < 2:
        raise ValueError('the two curves share fewer than 2 residues')
    Pa, Pb = xa[[ia[r] for r in res]], xb[[ib[r] for r in res]]
    ok = np.diff(res) == 1
    if max_step is not None:
        ok &= (np.linalg.norm(np.diff(Pa, axis=0), axis=1) < max_step) \
            & (np.linalg.norm(np.diff(Pb, axis=0), axis=1) < max_step)
    runs = np.split(np.arange(len(res)), np.flatnonzero(~ok) + 1)
    keep = max(runs, key=len)
    if verbose:
        print(f'{len(res)} common residues; using {res[keep[0]]}-{res[keep[-1]]} '
              f'({len(keep)} residues, longest gap-free run)')
    return res[keep], Pa[keep], Pb[keep]


# =============================================================================
# Smoothing
# =============================================================================

def smooth_step(X):
    """One pass of midpoint averaging along the chain axis (works for (N, 3) and (T, N, 3))."""
    return 0.5 * (X[..., :-1, :] + X[..., 1:, :])


def smooth(X, m):
    """m passes of midpoint averaging; the curve loses m points.
    Equivalent to a binomial(m) moving average of the points."""
    X = np.asarray(X, dtype=float)
    for _ in range(m):
        X = smooth_step(X)
    return X


# =============================================================================
# Writhe and fingerprints
# =============================================================================

def writhe_matrix(X, n_threads=1):
    """Symmetric segment-pair writhe, W[k, l] = Omega_kl / 2pi, so sum_{k<l} W = Wr.
    X : (N, 3) -> (N-1, N-1), or (T, N, 3) -> (T, N-1, N-1)."""
    X = np.ascontiguousarray(X, dtype=np.float32)
    if X.ndim == 2:
        S = np.asarray(find_Sigma_array(X), dtype=float)
    elif X.ndim == 3:
        S = np.asarray(find_Sigma_array_batch_parallel(X, n_threads), dtype=float)
    else:
        raise ValueError(f'expected (N, 3) or (T, N, 3), got shape {X.shape}')
    S = np.nan_to_num(S)
    return S + np.swapaxes(S, -1, -2)


def fingerprint_from_writhe(W):
    """F[..., i, j] = sum_{i <= k < l <= j} W[..., k, l] (upper triangle; zero below)."""
    U = np.triu(W, 1)
    C = np.flip(np.cumsum(np.flip(U, -2), -2), -2).cumsum(-1)    # sum over k >= i, l <= j
    return np.triu(C)


def fingerprint(X, m=0, n_threads=1):
    """Sub-chain writhe fingerprint of the m-smoothed curve(s). X : (N, 3) or (T, N, 3)."""
    return fingerprint_from_writhe(writhe_matrix(smooth(X, m), n_threads))


def global_writhe(F):
    """Total writhe from a fingerprint (its corner)."""
    return F[..., 0, -1]


# =============================================================================
# Pyramids
# =============================================================================

def pyramid(Fa, Fb=None, per_length=True, l_min=L_MIN):
    """Centre-length map of a fingerprint (Fb None) or of the difference Fb - Fa.

    Returns S with S[..., ℓ-1, k] = value for the window of length ℓ = j - i centred on
    segment k (i = k - ℓ//2, j = i + ℓ), NaN where the window leaves the chain or ℓ < l_min.
    per_length : divide by ℓ (winding per segment), as in the usual pyramid plots.
    Works on single fingerprints (n, n) and stacks (T, n, n).
    """
    D = Fa if Fb is None else Fb - Fa
    n = D.shape[-1]
    S = np.full(D.shape[:-2] + (n - 1, n), np.nan)
    k = np.arange(n)
    for ell in range(max(l_min, 1), n):
        i = k - ell // 2
        j = i + ell
        ok = (i >= 0) & (j <= n - 1)
        v = D[..., i[ok], j[ok]]
        S[..., ell - 1, k[ok]] = v / ell if per_length else v
    return S


def segment_to_residue(k, m, resid):
    """Residue number at segment k of the m-smoothed curve (fractional for odd/even m)."""
    return resid[0] + np.asarray(k) + (m + 1) / 2


def cross_section(S, ell, m, resid, on='original'):
    """Row ℓ of a pyramid mapped to per-point values.

    on='original': one value per residue of the original chain (interpolated from segment
                   positions; NaN where the window leaves the chain).
    on='smoothed': one value per point of the m-smoothed curve (mean of its two segments).
    """
    row = S[ell - 1]
    k = np.arange(len(row))
    ok = np.isfinite(row)
    if on == 'original':
        if not ok.any():
            return np.full(len(resid), np.nan)
        return np.interp(resid, segment_to_residue(k[ok], m, resid), row[ok], left=np.nan, right=np.nan)
    if on == 'smoothed':
        seg = np.where(ok, row, np.nan)
        return np.r_[seg[0], 0.5 * (seg[:-1] + seg[1:]), seg[-1]]
    raise ValueError("on must be 'original' or 'smoothed'")


# =============================================================================
# Topology checks
# =============================================================================

def seg_tri_hits(P0, P1, A, B, C, eps=1e-9):
    """(T, E) bool: does segment P0[e] -> P1[e] pierce triangle (A[t], B[t], C[t])?
    Möller-Trumbore ray/triangle test restricted to the segment."""
    e1, e2 = (B - A)[:, None], (C - A)[:, None]
    d = (P1 - P0)[None]
    h = np.cross(d, e2)
    a = (e1 * h).sum(-1)
    with np.errstate(divide='ignore', invalid='ignore'):
        f = 1 / np.where(np.abs(a) < eps, np.nan, a)
    s = P0[None] - A[:, None]
    u = f * (s * h).sum(-1)
    q = np.cross(s, e1)
    v = f * (d * q).sum(-1)
    t = f * (e2 * q).sum(-1)
    return (u > eps) & (v > eps) & (u + v < 1 - eps) & (t > eps) & (t < 1 - eps)


def smoothing_passages(X, m_max=15):
    """Does midpoint smoothing pass the curve through itself?

    Each smoothing pass cuts every corner X[k+1] off along the triangle
    (M[k], X[k+1], M[k+1]), M being the new midpoints. If another edge of the curve pierces
    that triangle, the cut drags the chain through it (KMT-style test: a vertex can only be
    removed safely if no other edge passes through its corner triangle).

    Returns
    -------
    counts : (m_max,) number of pierced corner triangles at each pass (0 = topology kept)
    where  : list of (n_hits, 2) arrays, per pass, of (corner, edge) positions in residue
             units of the original chain (index + 1 based), for locating the problem.
    """
    X = np.asarray(X, dtype=float)
    counts, where = [], []
    for m in range(m_max):
        if len(X) < 4:
            break
        M = smooth_step(X)
        hits = seg_tri_hits(X[:-1], X[1:], M[:-1], X[1:-1], M[1:])
        k = np.arange(len(M) - 1)[:, None]
        e = np.arange(len(X) - 1)[None]
        hits &= (e < k) | (e > k + 1)                  # ignore the corner's own edges
        counts.append(int(hits.sum()))
        where.append(np.argwhere(hits) + [m / 2 + 1, m / 2])
        X = M
    return np.array(counts), where


def strand_passages(X0, X1, n_steps=80, tol=0.5):
    """Segment pairs that pass through each other on the straight-line path X0 -> X1.

    At each of n_steps interpolation steps, the sign of (r34 x r12) . r13 (the writhe
    handedness of a segment pair) is tracked; a sign change while the two segments are
    closer than tol Å is reported as a passage.
    Returns a list of (i, j, lam, distance): segments i < j, fraction along the path, closest approach.
    """
    X0, X1 = np.asarray(X0, dtype=float), np.asarray(X1, dtype=float)
    I, J = np.triu_indices(len(X0) - 1, 2)
    out, prev = [], None
    for lam in np.linspace(0, 1, n_steps + 1):
        X = (1 - lam) * X0 + lam * X1
        r12, r34 = X[I + 1] - X[I], X[J + 1] - X[J]
        s = np.sign(np.einsum('ij,ij->i', np.cross(r34, r12), X[J] - X[I]))
        if prev is not None:
            s = np.where(s == 0, prev, s)                 # exactly coplanar: keep the last sign, count once
            for q in np.flatnonzero(s != prev):
                d = _segment_distance(X[I[q]], X[I[q] + 1], X[J[q]], X[J[q] + 1])
                if d < tol:
                    out.append((int(I[q]), int(J[q]), float(lam), float(d)))
        prev = s
    return out


def _segment_distance(p1, q1, p2, q2):
    """Closest distance between segments p1-q1 and p2-q2."""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f = d1 @ d1, d2 @ d2, d2 @ r
    c, b = d1 @ r, d1 @ d2
    den = a * e - b * b
    s = np.clip((b * f - c * e) / den, 0, 1) if den > 1e-12 else 0.0
    t = np.clip((b * s + f) / e, 0, 1) if e > 1e-12 else 0.0
    s = np.clip((b * t - c) / a, 0, 1) if a > 1e-12 else 0.0
    return float(np.linalg.norm(r + s * d1 - t * d2))


# =============================================================================
# One-call comparison
# =============================================================================

@dataclass
class Comparison:
    """Result of `compare`: two aligned-by-residue curves and their fingerprint pyramid."""
    resid: np.ndarray
    Xa: np.ndarray
    Xb: np.ndarray
    m: int
    Fa: np.ndarray
    Fb: np.ndarray
    S: np.ndarray
    per_length: bool
    names: tuple = ('A', 'B')
    smoothing_passages: dict = field(default_factory=dict)

    @property
    def global_writhe(self):
        return global_writhe(self.Fa), global_writhe(self.Fb)

    def peak(self):
        """(value, centre residue, ℓ) of the largest |entry| in the pyramid."""
        l_idx, k = np.unravel_index(np.nanargmax(np.abs(self.S)), self.S.shape)
        return float(self.S[l_idx, k]), float(segment_to_residue(k, self.m, self.resid)), int(l_idx + 1)

    def cross_section(self, ell, on='original'):
        return cross_section(self.S, ell, self.m, self.resid, on)


def compare(a, b, m=0, chain_a=None, chain_b=None, atom='CA', per_length=True,
            max_step=4.2, check_smoothing=True, names=None, verbose=True):
    """Fingerprint pyramid of B - A.

    a, b      : PDB paths or (N, 3) arrays of the same chain.
    m         : smoothing level applied to both curves before computing fingerprints.
    per_length: pyramid of ΔF/ℓ (True) or ΔF (False).
    max_step  : longest allowed step between consecutive atoms (Å) when picking the
                gap-free stretch; None to skip (e.g. coarse-grained or non-protein traces).
    check_smoothing : if m > 0, run `smoothing_passages` on both curves and warn when
                      smoothing passes either chain through itself.
    """
    resid, Xa, Xb = common_ca(a, b, chain_a, chain_b, atom, max_step, verbose)
    if names is None:
        names = tuple(os.path.basename(x) if isinstance(x, str) else lab for x, lab in ((a, 'A'), (b, 'B')))
    passages = {}
    if check_smoothing and m > 0:
        for lab, X in zip(names, (Xa, Xb)):
            counts, where = smoothing_passages(X, m)
            passages[lab] = (counts, where)
            if counts.sum():
                first = int(np.flatnonzero(counts)[0])
                warnings.warn(f'smoothing passes {lab} through itself ({counts.sum()} corner cuts '
                              f'pierced, first at pass {first + 1}); winding at m={m} is not a '
                              f'smooth deformation of the original chain there', stacklevel=2)
    Fa, Fb = fingerprint(Xa, m), fingerprint(Xb, m)
    S = pyramid(Fa, Fb, per_length)
    cmp = Comparison(resid, Xa, Xb, m, Fa, Fb, S, per_length, names, passages)
    if verbose:
        v, r, ell = cmp.peak()
        wa, wb = cmp.global_writhe
        print(f'm = {m}: Wr(A) = {wa:+.2f}, Wr(B) = {wb:+.2f}; peak {v:+.4f} at residue ~{r:.0f}, ℓ = {ell}')
    return cmp


# =============================================================================
# Plotting (matplotlib for pyramids, plotly for 3D)
# =============================================================================

def plot_pyramid(cmp, ells=(), vmax=None, ax=None, cmap='seismic', title=None):
    """Pyramid of a Comparison: x = centre residue, y = window length ℓ. Returns the axes."""
    import matplotlib.pyplot as plt
    S, m, resid = cmp.S, cmp.m, cmp.resid
    n = S.shape[1]
    vmax = np.nanmax(np.abs(S)) if vmax is None else vmax
    if ax is None:
        _, ax = plt.subplots(figsize=(7.4, 5.2), dpi=150)
    im = ax.imshow(S, origin='lower', aspect='auto', cmap=cmap, vmin=-vmax, vmax=vmax,
                   interpolation='nearest',
                   extent=[segment_to_residue(-0.5, m, resid), segment_to_residue(n - 0.5, m, resid),
                           0.5, S.shape[0] + 0.5])
    for ell in ells:
        ax.axhline(ell, c='k', ls='--', lw=0.8)
    ax.set_xlabel('centre residue')
    ax.set_ylabel(r'sub-chain length $\ell$')
    lab = r'$\Delta F/\ell$' if cmp.per_length else r'$\Delta F$'
    cax = ax.inset_axes([0, 1.04, 1, 0.04])
    cb = ax.figure.colorbar(im, cax=cax, orientation='horizontal')
    cb.set_label(f'{lab}   (smoothing $m={m}$)', labelpad=4)
    cax.xaxis.set_ticks_position('top')
    cax.xaxis.set_label_position('top')
    if title:
        ax.text(0.01, 0.97, title, transform=ax.transAxes, va='top', fontsize=11)
    return ax


def _kabsch_onto(A, B):
    """Centre A and rigidly rotate B onto it."""
    A0, B0 = A - A.mean(0), B - B.mean(0)
    U, _, Vt = np.linalg.svd(B0.T @ A0)
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    return A0, B0 @ U @ D @ Vt


def _plotly_scale(cmap_name, n=128):
    import matplotlib.pyplot as plt
    cm = plt.get_cmap(cmap_name)
    return [[x, 'rgb({:.0f},{:.0f},{:.0f})'.format(*(255 * np.array(cm(x)[:3])))] for x in np.linspace(0, 1, n)]


def plot_cross_section(cmp, ell, on='original', vmax=None, cmap='seismic', width=1200, height=550):
    """Plotly figure: both structures side by side, coloured by the pyramid row at ℓ.
    on='original' colours the real chains; on='smoothed' draws the m-smoothed curves.
    Grey where the window leaves the chain. Returns the figure (call .show() or .write_html())."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    col = cmp.cross_section(ell, on)
    if on == 'original':
        A, B = _kabsch_onto(cmp.Xa, cmp.Xb)
        pos = cmp.resid.astype(float)
    else:
        A, B = _kabsch_onto(smooth(cmp.Xa, cmp.m), smooth(cmp.Xb, cmp.m))
        pos = cmp.resid[0] + np.arange(len(A)) + cmp.m / 2
    vmax = np.nanmax(np.abs(col)) if vmax is None else vmax
    cs = _plotly_scale(cmap)
    grey = ~np.isfinite(col)
    labels = [f'res {p:.0f}: ' + ('window off chain' if g else f'{v:+.4f}') for p, v, g in zip(pos, col, grey)]
    hidden = dict(visible=False, showgrid=False, showticklabels=False, showbackground=False, title='')
    fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'scene'}, {'type': 'scene'}]], subplot_titles=list(cmp.names))
    for c, P in enumerate([A, B], start=1):
        fig.add_trace(go.Scatter3d(x=P[:, 0], y=P[:, 1], z=P[:, 2], mode='lines',
                                   line=dict(width=4, color='rgb(190,190,190)'), hoverinfo='skip', showlegend=False),
                      row=1, col=c)
        Q = np.where(grey[:, None], np.nan, P)
        fig.add_trace(go.Scatter3d(x=Q[:, 0], y=Q[:, 1], z=Q[:, 2], mode='lines+markers',
                                   line=dict(width=8, color=np.nan_to_num(col), colorscale=cs, cmin=-vmax, cmax=vmax),
                                   marker=dict(size=3, color=np.nan_to_num(col), colorscale=cs, cmin=-vmax, cmax=vmax,
                                               colorbar=dict(title='ΔF/ℓ' if cmp.per_length else 'ΔF') if c == 2 else None),
                                   text=labels, hoverinfo='text', showlegend=False), row=1, col=c)
        fig.add_trace(go.Scatter3d(x=[P[0, 0]], y=[P[0, 1]], z=[P[0, 2]], mode='markers',
                                   marker=dict(size=7, color='black'), hoverinfo='skip', showlegend=False), row=1, col=c)
    fig.update_scenes(aspectmode='data', xaxis=hidden, yaxis=hidden, zaxis=hidden)
    fig.update_layout(height=height, width=width, template='plotly_white',
                      title=f'ℓ = {ell}, smoothing m = {cmp.m}  (black dot = first residue; grey = window off the chain)')
    return fig


# =============================================================================
# Export
# =============================================================================

def write_bfactor_pdb(path, resid, X, values, atom='CA', resname='UNK', chain='A', bonds=True):
    """Write a single-structure PDB of the trace with `values` in the B-factor column
    (NaN -> 0). With bonds=True, CONECT records join consecutive points so VMD/PyMOL draw
    the chain even when points are further apart than their automatic bond cutoff."""
    values = np.nan_to_num(np.asarray(values, dtype=float))
    name = f' {atom:<3s}' if len(atom) < 4 else atom
    with open(path, 'w') as fo:
        for a, (r, (x, y, z), b) in enumerate(zip(resid, X, values), start=1):
            fo.write(f'ATOM  {a:5d} {name} {resname:>3s} {chain}{int(r):4d}    '
                     f'{x:8.3f}{y:8.3f}{z:8.3f}  1.00{b:6.2f}           C\n')
        if bonds:
            for a in range(1, len(resid)):
                fo.write(f'CONECT{a:5d}{a + 1:5d}\n')
        fo.write('END\n')
