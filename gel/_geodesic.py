import heapq

import numpy as np
import scipy.ndimage as nd
import scipy.spatial.distance as distance


def neighbors(coords, shape, connectivity=1):
    """Return the neighbors of a set of coordinates.

    Parameters
    ----------
    coords : array-like
        The coordinates for which we want neighbor coordinates.
    shape : array-like
        The shape of the array including coords. Used to check for borders.
    connectivity : int (optional), min=1, max=len(shape)
        The connectivity of the neighborhood.

    Returns
    -------
    neighbors : np.ndarray (num_neighbors x len(shape))
        The coordinates of neighboring array elements.
    """
    coords = np.atleast_2d(coords)
    shape = np.asarray(shape)
    ndim = len(shape)
    n_elem = 3 ** ndim
    footprint = nd.generate_binary_structure(ndim, connectivity)
    footprint.ravel()[n_elem / 2] = 0
    neighbors = coords + (np.asarray(footprint.nonzero()).T - np.ones(ndim))
    not_border = True - ((neighbors < 0).any(axis=1) + 
                         (neighbors >= shape).any(axis=1))
    return neighbors[not_border].astype(int)

def geodesic_expansion(labels, image, mode='viscosity', connectivity=1):
    """Expand the location of labels into a geodesic space defined by image.

    Parameters
    ----------
    labels : numpy array (integer type)
        The initial location of the labels (typically sparse)
    image : numpy array, dimensions = labels.ndim or labels.ndim + 1
        The space along which labels must expand. It can either be
        single-channel (same dimension as `labels`) or multi-channel (one more
        dimension than `labels`; the last dimension is assumed to be the
        channels).
    mode : string (optional)
        Whether to treat the image values as a `viscosity` (default) or a 
        `feature`. In the first case, the cost of expanding from pixel `x` to
        pixel `y` is `image[y]`. In the second, it is `d(image[x], image[y])`.
    connectivity : int (optional)
        The connectivity defining neighboring pixels. It is an int between 1
        and `labels.ndim`, inclusive. (default: 1)

    Returns
    -------
    labels : numpy array (integer type)
        The resulting label field
    """
    label_locations = labels.nonzero()
    initial_labels = labels[label_locations]
    distance_heap = [(0, coord, label) for coord, label in 
         zip(np.transpose(label_locations), initial_labels)]
    if mode == 'viscosity':
        def dist(img, src, dst): return img[dst]
    else:
        def dist(img, src, dst): return distance.euclidean(img[src], img[dst])
    labels_out = np.zeros_like(labels)
    while len(distance_heap) > 0:
        nearest = heapq.heappop(distance_heap)
        d, loc, lab = nearest
        if labels_out[loc] == 0:
            labels_out[loc] = lab
            for n in neighbors(loc, labels_out.shape, connectivity):
                if labels_out[n] == 0:
                    next_d = d + dist(image, loc, n)
                    heapq.heappush(distance_heap, (next_d, n, lab))
    return labels_out
