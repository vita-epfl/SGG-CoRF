# cython: infer_types=True
cimport cython
from libc.math cimport exp, fabs, sqrt, fmin, fmax
import numpy as np

cdef inline float clip(float v, float minv, float maxv) nogil:
    return fmax(minv, fmin(maxv, v))

@cython.boundscheck(False)
@cython.wraparound(False)
def scalar_values_3d(float[:, :, :] field, float[:] x, float[:] y, float default=-1, float[:] scale=None):
    values_np = np.full((field.shape[0], x.shape[0],), default, dtype=np.float32)
    cdef float[:,:] values = values_np
    cdef float cv, cx, cy, csigma, csigma2
    cdef long minx, miny, maxx, maxy

    for i in range(values.shape[1]):
        if scale is not None:
          csigma = scale[i]
        else:
          csigma = 1.0
        cx = x[i]
        cy = y[i]
        minx = (<long>clip(cx - 0.5*csigma, 0, field.shape[1] - 1))
        maxx = (<long>clip(cx + 0.5*csigma, minx + 1, field.shape[1]))
        miny = (<long>clip(cy - 0.5*csigma, 0, field.shape[0] - 1))
        maxy = (<long>clip(cy + 0.5*csigma, miny + 1, field.shape[0]))

        values[:, i] = np.amax(field[:, miny:maxy, minx:maxx], axis=(1,2))

    return values_np
