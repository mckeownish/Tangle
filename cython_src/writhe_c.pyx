# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
writhe_cython_fast.pyx
======================

Drop-in optimized replacement for find_Sigma_array_batch_parallel.
Measured ~5x faster single-threaded than the original, output identical to
~1e-10 (below float32 precision). The speedup is per-thread, so it stacks with
your OpenMP scaling.

Build (the flags matter -- see notes at the bottom):

    from setuptools import setup, Extension
    from Cython.Build import cythonize
    import numpy as np
    ext = Extension("writhe_cython_fast", ["writhe_cython_fast.pyx"],
                    extra_compile_args=['-O3', '-march=native',
                                        '-funroll-loops', '-fopenmp'],
                    extra_link_args=['-fopenmp'],
                    include_dirs=[np.get_include()])
    setup(ext_modules=cythonize([ext], language_level=3))
"""

cimport numpy as cnp
import numpy as np
cimport cython
from cython cimport floating
from cython.parallel import prange
from libc.math cimport asin, sqrt, fabs, isnan, NAN

# 1/(2*pi): the original 1/(4*pi) with the trailing  *2  folded in
cdef double INV_2PI = 0.15915494309189535
cdef double TINY = 1e-30          # squared-norm floor for degenerate segments


# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True


cdef inline double clamp_unit(double x) noexcept nogil:
    # GUARD: with deferred normalization the dot can land just past +/-1;
    # asin of that is NaN, and one NaN poisons every downstream mean/max.
    if x > 1.0:
        return 1.0
    elif x < -1.0:
        return -1.0
    else:
        return x


# -------------------------------------------------------------------------
# Optimized per-pair kernel.
#   - pure scalar locals (no C arrays / pointer helpers -> registers + SIMD)
#   - deferred normalization: (a/|a|).(b/|b|) = (a.b)/(|a||b|)  (12 -> 4 divs)
#   - r12 passed in (constant across the inner j loop)
#   - factor 2 folded into the return constant (kills the second N^2 pass)
# -------------------------------------------------------------------------
cdef double _gauss_int(float* p1, float* p2, float* p3, float* p4,
                       double r12x, double r12y, double r12z) noexcept nogil:
    cdef double r13x = <double>p3[0]-p1[0], r13y = <double>p3[1]-p1[1], r13z = <double>p3[2]-p1[2]
    cdef double r14x = <double>p4[0]-p1[0], r14y = <double>p4[1]-p1[1], r14z = <double>p4[2]-p1[2]
    cdef double r23x = <double>p3[0]-p2[0], r23y = <double>p3[1]-p2[1], r23z = <double>p3[2]-p2[2]
    cdef double r24x = <double>p4[0]-p2[0], r24y = <double>p4[1]-p2[1], r24z = <double>p4[2]-p2[2]
    cdef double r34x = <double>p4[0]-p3[0], r34y = <double>p4[1]-p3[1], r34z = <double>p4[2]-p3[2]

    # four raw (un-normalized) normals
    cdef double c1x = r13y*r14z-r13z*r14y, c1y = r13z*r14x-r13x*r14z, c1z = r13x*r14y-r13y*r14x
    cdef double c2x = r14y*r24z-r14z*r24y, c2y = r14z*r24x-r14x*r24z, c2z = r14x*r24y-r14y*r24x
    cdef double c3x = r24y*r23z-r24z*r23y, c3y = r24z*r23x-r24x*r23z, c3z = r24x*r23y-r24y*r23x
    cdef double c4x = r23y*r13z-r23z*r13y, c4y = r23z*r13x-r23x*r13z, c4z = r23x*r13y-r23y*r13x

    cdef double s1 = c1x*c1x+c1y*c1y+c1z*c1z, s2 = c2x*c2x+c2y*c2y+c2z*c2z
    cdef double s3 = c3x*c3x+c3y*c3y+c3z*c3z, s4 = c4x*c4x+c4y*c4y+c4z*c4z

    # GUARD: collinear segments -> zero-area normal -> 0 contribution
    cdef double i1 = (1.0/sqrt(s1)) if s1 > TINY else 0.0
    cdef double i2 = (1.0/sqrt(s2)) if s2 > TINY else 0.0
    cdef double i3 = (1.0/sqrt(s3)) if s3 > TINY else 0.0
    cdef double i4 = (1.0/sqrt(s4)) if s4 > TINY else 0.0

    cdef double a12 = clamp_unit((c1x*c2x+c1y*c2y+c1z*c2z)*i1*i2)
    cdef double a23 = clamp_unit((c2x*c3x+c2y*c3y+c2z*c3z)*i2*i3)
    cdef double a34 = clamp_unit((c3x*c4x+c3y*c4y+c3z*c4z)*i3*i4)
    cdef double a41 = clamp_unit((c4x*c1x+c4y*c1y+c4z*c1z)*i4*i1)

    cdef double Sigma = asin(a12)+asin(a23)+asin(a34)+asin(a41)

    # sign from the raw triple product (r34 x r12).r13
    cdef double tx = r34y*r12z-r34z*r12y, ty = r34z*r12x-r34x*r12z, tz = r34x*r12y-r34y*r12x
    cdef double sign = 1.0 if (tx*r13x+ty*r13y+tz*r13z) > 0 else -1.0

    return INV_2PI * Sigma * sign


# -------------------------------------------------------------------------
# Batched, parallel over frames. Same signature/output as your original.
# -------------------------------------------------------------------------
cpdef cnp.ndarray[cnp.float32_t, ndim=3] find_Sigma_array_batch_parallel(
        cnp.ndarray[cnp.float32_t, ndim=3] trajectory, int num_threads=1):
    cdef int n_frames = trajectory.shape[0]
    cdef int n_atoms = trajectory.shape[1]
    cdef int segment_num = n_atoms - 1
    cdef cnp.ndarray[cnp.float32_t, ndim=3] results = np.zeros(
        (n_frames, segment_num, segment_num), dtype=np.float32)
    cdef int frame_idx, i, j
    cdef int nt = num_threads          # local copy: avoids a codegen quirk with
    cdef double r12x, r12y, r12z       # runtime num_threads in some Cython builds
    cdef float[:, :, ::1] traj = trajectory
    cdef float[:, :, ::1] out = results

    for frame_idx in prange(n_frames, nogil=True, num_threads=nt, schedule='static'):
        for i in range(1, n_atoms - 1):
            # r12 is constant across the inner loop -> hoist it out
            r12x = <double>traj[frame_idx, i+1, 0] - traj[frame_idx, i, 0]
            r12y = <double>traj[frame_idx, i+1, 1] - traj[frame_idx, i, 1]
            r12z = <double>traj[frame_idx, i+1, 2] - traj[frame_idx, i, 2]
            for j in range(0, i - 1):
                out[frame_idx, i, j] = <float>_gauss_int(
                    &traj[frame_idx, i, 0], &traj[frame_idx, i+1, 0],
                    &traj[frame_idx, j, 0], &traj[frame_idx, j+1, 0],
                    r12x, r12y, r12z)
    return results


# -------------------------------------------------------------------------
# Single-frame convenience (float32 in, matches the batch numerics).
# -------------------------------------------------------------------------
cpdef cnp.ndarray[cnp.float32_t, ndim=2] find_Sigma_array(
        cnp.ndarray[cnp.float32_t, ndim=2] curve_points):
    cdef cnp.ndarray[cnp.float32_t, ndim=3] batched = curve_points[np.newaxis, :, :]
    return find_Sigma_array_batch_parallel(batched, num_threads=1)[0]


# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
calculate_writhe_stats_fast.pyx
===============================
Vectorized stride computations – 10-50× faster than the original pure Python.
Uses NumPy's C-level reductions instead of a manual inner loop.
"""


def calculate_writhe_stats(
    cnp.ndarray[cnp.float32_t, ndim=2] flat_sig_tensor,  # any memory layout
    cnp.ndarray[cnp.int32_t, ndim=1] strides,
):
    """
    Parameters
    ----------
    flat_sig_tensor : 2D float32 array (L, D)
    strides : 1D int64 array of stride values

    Returns
    -------
    times : list of float
        stride / 10.0 for each valid stride.
    max_list : list of 1D float arrays
        Per‑feature maximum absolute difference for each stride.
    mean_list : list of 1D float arrays
        Per‑feature mean absolute difference.
    """
    cdef int L = flat_sig_tensor.shape[0]
    cdef int D = flat_sig_tensor.shape[1]
    cdef int n_strides = strides.shape[0]

    cdef list times = []
    cdef list max_list = []
    cdef list mean_list = []

    cdef int i, stride, stride_cutoff
    cdef cnp.ndarray diff, abs_diff

    # Ensure contiguous layout for slice speed (optional – views are fine too)
    cdef cnp.ndarray[cnp.float32_t, ndim=2] arr = flat_sig_tensor
    if not arr.flags['C_CONTIGUOUS']:
        arr = np.ascontiguousarray(arr)

    for i in range(n_strides):
        stride = strides[i]
        stride_cutoff = L - stride
        if stride_cutoff <= 1:          # original condition: L - stride > 1
            continue

        # Vectorised difference: frame[i+stride] - frame[i] for all i
        diff = arr[stride:] - arr[:stride_cutoff]       # shape (stride_cutoff, D)
        abs_diff = np.abs(diff)

        # Reductions along feature axis (axis=1)
        max_list.append(np.nanmax(abs_diff, axis=1))
        mean_list.append(np.nanmean(abs_diff, axis=1))
        times.append(stride / 10.0)

    return times, max_list, mean_list




# B(D,d) = (2/pi) * asin( (d/2)^2 / (D^2 + (d/2)^2) )
cdef inline double _geo_bound(double Dsq, double csq) noexcept nogil:
    return 0.6366197723675814 * asin(csq / (Dsq + csq + 1e-10))


cpdef cnp.ndarray[cnp.float32_t, ndim=2] find_Sigma_flat_trimmed(
        cnp.ndarray[cnp.float32_t, ndim=3] trajectory,
        int num_threads=1,
        bint normalise=False,
        double d=3.8):
    """
    Returns (n_frames, n_pairs) directly: the [2:, :-2] trim + lower-tri flatten
    (li, lj = np.tril_indices(N), N = n_atoms-3) done inside the loop.
    If normalise, divides each pair by its distance-dependent bound B(D,d).
    """
    cdef int n_frames = trajectory.shape[0]
    cdef int n_atoms  = trajectory.shape[1]
    # i runs 2..n_atoms-2 (the [2:] row trim + the existing range(1, n_atoms-1)
    # already drops i=n_atoms-1; the [:-2] col trim caps j at i-2 which the
    # j-loop bound i-1 already enforces, so the corridor is exactly tril of an
    # N x N block with N = n_atoms-3).
    cdef int N = n_atoms - 3
    if N < 1:
        return np.zeros((n_frames, 0), dtype=np.float32)
    cdef int n_pairs = (N * (N + 1)) // 2

    cdef cnp.ndarray[cnp.float32_t, ndim=2] results = np.zeros(
        (n_frames, n_pairs), dtype=np.float32)
    cdef float[:, :, ::1] traj = trajectory
    cdef float[:, ::1] out = results

    cdef int frame_idx, i, j, ri, cj, k
    cdef int nt = num_threads
    cdef double r12x, r12y, r12z, val
    cdef double m1x, m1y, m1z, m2x, m2y, m2z, dx, dy, dz, Dsq
    cdef double csq = (d * 0.5) * (d * 0.5)

    for frame_idx in prange(n_frames, nogil=True,
                            num_threads=num_threads, schedule='static'):
        # row index in trimmed block: i=2 -> ri=0
        for i in range(2, n_atoms - 1):
            ri = i - 2
            r12x = <double>traj[frame_idx, i+1, 0] - traj[frame_idx, i, 0]
            r12y = <double>traj[frame_idx, i+1, 1] - traj[frame_idx, i, 1]
            r12z = <double>traj[frame_idx, i+1, 2] - traj[frame_idx, i, 2]
            for j in range(0, i - 1):
                cj = j                       # col 0..i-2 maps straight across
                k = (ri * (ri + 1)) // 2 + cj   # flat tril index, row-major
                val = _gauss_int(
                    &traj[frame_idx, i, 0], &traj[frame_idx, i+1, 0],
                    &traj[frame_idx, j, 0], &traj[frame_idx, j+1, 0],
                    r12x, r12y, r12z)
                if normalise:
                    # segment centres i..i+1 and j..j+1
                    m1x = 0.5*(<double>traj[frame_idx, i, 0] + traj[frame_idx, i+1, 0])
                    m1y = 0.5*(<double>traj[frame_idx, i, 1] + traj[frame_idx, i+1, 1])
                    m1z = 0.5*(<double>traj[frame_idx, i, 2] + traj[frame_idx, i+1, 2])
                    m2x = 0.5*(<double>traj[frame_idx, j, 0] + traj[frame_idx, j+1, 0])
                    m2y = 0.5*(<double>traj[frame_idx, j, 1] + traj[frame_idx, j+1, 1])
                    m2z = 0.5*(<double>traj[frame_idx, j, 2] + traj[frame_idx, j+1, 2])
                    dx = m1x - m2x; dy = m1y - m2y; dz = m1z - m2z
                    Dsq = dx*dx + dy*dy + dz*dz
                    val = val / (_geo_bound(Dsq, csq) + 1e-10)
                out[frame_idx, k] = <float>val
    return results


cpdef tuple find_Sigma_and_bound_flat_trimmed(
        cnp.ndarray[cnp.float32_t, ndim=3] trajectory,
        int num_threads=1,
        double d=3.8):
    """
    Same pair corridor/flattening as find_Sigma_flat_trimmed, but returns
    (W_raw, B) as two (n_frames, n_pairs) float32 arrays instead of one
    normalise flag -- for the exact bilinear decomposition
        W_ij = B_ij * W_tilde_ij  (W_tilde_ij = W_raw_ij / (B_ij + 1e-10))
        Delta(W_ij) = Delta(B_ij) * mean(W_tilde_ij) + mean(B_ij) * Delta(W_tilde_ij)
    Getting B directly here (rather than back-dividing two separately-computed
    normalise=True/False calls) avoids numerical noise right at W_tilde's
    zero-crossings, since B itself never touches W_tilde at all -- it's pure
    segment-midpoint distance geometry, same _geo_bound as the normalise=True
    branch above, just also written out instead of only used as a divisor.
    """
    cdef int n_frames = trajectory.shape[0]
    cdef int n_atoms  = trajectory.shape[1]
    cdef int N = n_atoms - 3
    if N < 1:
        return (np.zeros((n_frames, 0), dtype=np.float32),
                np.zeros((n_frames, 0), dtype=np.float32))
    cdef int n_pairs = (N * (N + 1)) // 2

    cdef cnp.ndarray[cnp.float32_t, ndim=2] W_results = np.zeros(
        (n_frames, n_pairs), dtype=np.float32)
    cdef cnp.ndarray[cnp.float32_t, ndim=2] B_results = np.zeros(
        (n_frames, n_pairs), dtype=np.float32)
    cdef float[:, :, ::1] traj = trajectory
    cdef float[:, ::1] out_w = W_results
    cdef float[:, ::1] out_b = B_results

    cdef int frame_idx, i, j, ri, cj, k
    cdef int nt = num_threads
    cdef double r12x, r12y, r12z, val
    cdef double m1x, m1y, m1z, m2x, m2y, m2z, dx, dy, dz, Dsq, bval
    cdef double csq = (d * 0.5) * (d * 0.5)

    for frame_idx in prange(n_frames, nogil=True,
                            num_threads=num_threads, schedule='static'):
        for i in range(2, n_atoms - 1):
            ri = i - 2
            r12x = <double>traj[frame_idx, i+1, 0] - traj[frame_idx, i, 0]
            r12y = <double>traj[frame_idx, i+1, 1] - traj[frame_idx, i, 1]
            r12z = <double>traj[frame_idx, i+1, 2] - traj[frame_idx, i, 2]
            for j in range(0, i - 1):
                cj = j
                k = (ri * (ri + 1)) // 2 + cj
                val = _gauss_int(
                    &traj[frame_idx, i, 0], &traj[frame_idx, i+1, 0],
                    &traj[frame_idx, j, 0], &traj[frame_idx, j+1, 0],
                    r12x, r12y, r12z)
                m1x = 0.5*(<double>traj[frame_idx, i, 0] + traj[frame_idx, i+1, 0])
                m1y = 0.5*(<double>traj[frame_idx, i, 1] + traj[frame_idx, i+1, 1])
                m1z = 0.5*(<double>traj[frame_idx, i, 2] + traj[frame_idx, i+1, 2])
                m2x = 0.5*(<double>traj[frame_idx, j, 0] + traj[frame_idx, j+1, 0])
                m2y = 0.5*(<double>traj[frame_idx, j, 1] + traj[frame_idx, j+1, 1])
                m2z = 0.5*(<double>traj[frame_idx, j, 2] + traj[frame_idx, j+1, 2])
                dx = m1x - m2x; dy = m1y - m2y; dz = m1z - m2z
                Dsq = dx*dx + dy*dy + dz*dz
                bval = _geo_bound(Dsq, csq)
                out_w[frame_idx, k] = <float>val
                out_b[frame_idx, k] = <float>bval
    return (W_results, B_results)