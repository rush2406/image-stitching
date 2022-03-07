from typing import List
import numpy as np
from images import Image
from matching.pair_match import PairMatch

def build_homographies(pair_match) -> None:
    """
    The homographies are saved in the images themselves
    """
    pair_match.compute_homography()

    pair_match.image_a.H = np.eye(3)
    pair_match.image_b.H = pair_match.H