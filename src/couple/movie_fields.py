"""Shared mesh-free drawing and accepted-state movie timing."""
import numpy as np
from scipy.spatial import Delaunay, cKDTree

YEAR_SECONDS = 365 * 24 * 3600


def frame_indices(clock, seconds, fps):
    if seconds <= 0 or not np.isfinite(seconds) or fps < 1 or len(clock) < 2:
        raise ValueError('Need positive playback duration/fps and at least two states')
    n = max(2, round(seconds * fps), len(clock))
    starts = np.rint((clock[:, 1]-clock[0, 1])/(clock[-1, 1]-clock[0, 1])*(n-1)).astype(int)
    starts = np.clip(starts, np.arange(len(clock)), n-len(clock)+np.arange(len(clock)))
    for k in range(len(starts)-2, -1, -1):
        starts[k] = min(starts[k], starts[k+1]-1)
    indices = np.searchsorted(starts, np.arange(n), side='right')-1
    return np.r_[np.zeros(fps, dtype=int), indices, np.full(fps, len(clock)-1, dtype=int)]


class SpatialGrid:
    """Linear spatial interpolation for display only; mask outside sample hull."""
    def __init__(self, points, domain, resolution=120):
        self.x = np.linspace(domain[0], domain[1], resolution)
        self.y = np.linspace(domain[2], domain[3], resolution)
        self.xx, self.yy = np.meshgrid(self.x, self.y)
        query = np.c_[self.xx.ravel(), self.yy.ravel()]
        tri = Delaunay(points)
        simplex = tri.find_simplex(query)
        self.inside = simplex >= 0
        safe = np.maximum(simplex, 0)
        transform = tri.transform[safe]
        b = np.einsum('ijk,ik->ij', transform[:, :2], query-transform[:, 2])
        self.weights = np.c_[b, 1-b.sum(axis=1)]
        self.vertices = tri.simplices[safe]
        self.nearest = cKDTree(points).query(query)[1]

    def sample(self, values, categorical=False):
        if categorical:
            result = np.asarray(values)[self.nearest].astype(float)
        else:
            values = np.asarray(values)[self.vertices]
            result = np.sum(values*self.weights, axis=1)
        result[~self.inside] = np.nan
        return np.ma.masked_invalid(result.reshape(self.xx.shape))


def velocity_vectors(gauss):
    """Physical solid velocity, liquid velocity, and segregation flux, cm/year."""
    solid = gauss[:, :2] * YEAR_SECONDS * 100
    flux = gauss[:, 2:4] * YEAR_SECONDS * 100
    wet = gauss[:, 4] > 0
    liquid = np.full_like(solid, np.nan)
    liquid[wet] = solid[wet] + flux[wet]/gauss[wet, 4, None]
    return solid, liquid, flux


def reference_speed(vectors):
    lengths = np.linalg.norm(vectors, axis=1)
    finite = lengths[np.isfinite(lengths)]
    maximum = float(finite.max()) if len(finite) else 0.
    if maximum <= 0:
        return 1.
    power = 10.**np.floor(np.log10(maximum))
    return next(value*power for value in (1, 2, 5, 10) if value*power >= maximum)


class VelocityDrawing:
    """Uncolored quivers at sampled Gauss positions; constant-width streamlines."""
    names = ('Solid velocity $v_s$', 'Liquid velocity $v_l$', 'Darcy segregation flux $q$')
    colors = ('#146082', '#b34827', '#69469d')

    def __init__(self, fig, axes, points, domain, references, max_arrows=400):
        self.fig, self.axes, self.points, self.domain, self.references = fig, axes, points, domain, references
        # Edge exports may carry two traces at the same position. Average those
        # only for streamline interpolation; quivers retain their actual samples.
        unique, inverse = np.unique(points, axis=0, return_inverse=True)
        self.inverse, self.counts = inverse, np.bincount(inverse)
        self.grid = SpatialGrid(unique, domain, resolution=90)
        side = max(2, int(np.sqrt(max_arrows)))
        xx, yy = np.meshgrid(np.linspace(points[:, 0].min(), points[:, 0].max(), side),
                             np.linspace(points[:, 1].min(), points[:, 1].max(), side))
        self.arrows = np.unique(cKDTree(points).query(np.c_[xx.ravel(), yy.ravel()])[1])

    def draw(self, gauss):
        span = max(self.domain[1]-self.domain[0], self.domain[3]-self.domain[2])
        for row, (vectors, label, color, reference) in enumerate(zip(velocity_vectors(gauss), self.names, self.colors, self.references)):
            left, right = self.axes[row]
            for ax in (left, right):
                ax.clear(); ax.set_xlim(self.domain[:2]); ax.set_ylim(self.domain[2:])
                ax.set_aspect('equal'); ax.set_xlabel('x (km)'); ax.set_ylabel('y (km)'); ax.grid(False)
            left.set_title(label+' · quiver', fontsize=13, pad=14)
            right.set_title(label+' · streamlines', fontsize=13, pad=14)
            selected = self.arrows
            moving = np.isfinite(vectors[selected]).all(axis=1) & (np.linalg.norm(vectors[selected], axis=1)>0)
            selected = selected[moving]
            if len(selected):
                arrows = left.quiver(*self.points[selected].T, *vectors[selected].T, color=color,
                                     angles='xy', scale_units='xy', scale=reference/(.065*span), width=.0032, pivot='mid')
                left.quiverkey(arrows, .70, 1.015, reference, f'{reference:.3g} cm/year', labelpos='E', fontproperties={'size':9})
            else:
                left.text(.5,.5,'No liquid present' if row==1 else 'Zero flow',ha='center',transform=left.transAxes)
            # A triangle touching an undefined liquid point remains masked.
            components = []
            for axis in (0, 1):
                component = np.bincount(self.inverse, weights=vectors[:, axis], minlength=len(self.counts))/self.counts
                components.append(self.grid.sample(component))
            u, v = components
            if np.any(np.ma.filled(u*u+v*v, 0)>0):
                right.streamplot(self.grid.x, self.grid.y, u, v, color=color, linewidth=.85, density=1.05,
                                 arrowsize=1.0, broken_streamlines=True)
            else:
                right.text(.5,.5,'No liquid present' if row==1 else 'Zero flow',ha='center',transform=right.transAxes)
