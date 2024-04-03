import numpy as np
from scipy.spatial import distance

def spatial_intensity_dist(
        c1, 
        c2, 
        max_distance: int | float,
        mklp_weight_factor: float,
        sir_weight_factor: float) -> float:
    
    """Modified version of sqeuclidian distance
    Square Euclidian distance is applied to spatial coordinates
    x and y.
    while an 'intensity' distance is computed with MLKP and
    SIR intensities
    Finally values are combined by weighted addition
    """

    # unwrapping
    (x1, y1, mlkp1, sir1), (x2, y2, mlkp2, sir2) = c1, c2
    
    # In case we have a None None point:
    if np.isnan([x1, y1, x2, y2]).any():
        return max_distance*2 # connection is invalidated
    
    # spatial coordinates: euclidean
    spatial_e = distance.euclidean([x1, y1], [x2, y2])
    
    mkpl_penalty = (
        3
        * mklp_weight_factor
        * np.abs(mlkp1 - mlkp2) / (mlkp1 + mlkp2)
    )
    sir_penalty = (
        3
        * sir_weight_factor
        * np.abs(sir1 - sir2) / (sir1 + sir2)
    )
    penalty = (
        1
        + mkpl_penalty
        + sir_penalty
    )
    return (spatial_e * penalty)**2 

