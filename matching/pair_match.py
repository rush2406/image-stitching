from typing import List, Optional
import cv2
import numpy as np
import os
from images import Image


class PairMatch:
    def __init__(self, image_a: Image, image_b: Image, matches: Optional[List] = None):
        """
        Create a new PairMatch object.

        Parameters
        ----------
        image_a : Image
            First image of the pair.
        image_b : Image
            Second image of the pair.
        matches : Optional[List], optional
            List of matches between image_a and image_b, by default None
        """
        self.image_a = image_a
        self.image_b = image_b
        self.matches = matches
        self.H = None
        self.status = None
        self.matchpoints_a = None
        self.matchpoints_b = None

    def contains(self, image: Image) -> bool:
        """
        Check if the given image is contained in the pair match.

        Parameters
        ----------
        image : Image
            Image to check.

        Returns
        -------
        bool
            True if the given image is contained in the pair match, False otherwise.
        """
        return self.image_a == image or self.image_b == image

    def compute_homography(
        self, ransac_reproj_thresh: float = 4, ransac_max_iter: int = 500
    ) -> None:
        """
        Compute the homography between the two images of the pair.

        Parameters
        ----------
        ransac_reproj_thresh : float, optional
            Reprojection threshold used in the RANSAC algorithm, by default 5
        ransac_max_iter : int, optional
            Number of maximum iterations for the RANSAC algorithm, by default 500
        """
        self.matchpoints_a = np.float32(
            [self.image_a.keypoints[match.queryIdx].pt for match in self.matches]
        )
        self.matchpoints_b = np.float32(
            [self.image_b.keypoints[match.trainIdx].pt for match in self.matches]
        )
        
        self.H, self.status = cv2.findHomography(
        self.matchpoints_b,
        self.matchpoints_a,
        cv2.RANSAC,
        ransac_reproj_thresh,
        maxIters=ransac_max_iter,
    )