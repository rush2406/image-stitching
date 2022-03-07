from typing import List, Tuple

import cv2
import numpy as np
import imutils
import matplotlib.pyplot as plt

from images import Image
from rendering.utils import get_new_parameters, single_weights_matrix

def brute_force_blend(images,pair_match,wt1=0.5,wt2=0.5) :
    """
    Brute force
    """
    width = pair_match.image_a.image.shape[1] + pair_match.image_b.image.shape[1]
    height = pair_match.image_a.image.shape[0] + pair_match.image_b.image.shape[0]

    result = cv2.warpPerspective(pair_match.image_b.image, pair_match.image_b.H, (width,height))
    mask_left = np.zeros((height,width,3))
    mask_right = np.zeros((height,width,3))
    mask_overlap = np.zeros((height,width,3))

    result_left = np.zeros((height,width,3)).astype('uint8')
    result_left[0:pair_match.image_a.image.shape[0], 0:pair_match.image_a.image.shape[1]] = pair_match.image_a.image

    temp= np.zeros((height,width,3))
    temp[0:pair_match.image_a.image.shape[0], 0:pair_match.image_a.image.shape[1]]=1

    roverlap = result*temp
    temp[~roverlap.astype(bool)]=0
    loverlap = result_left*temp

    left = result_left*~temp.astype(bool)
    right = result*~temp.astype(bool)

    ans = (left + (wt1* loverlap + wt2 * roverlap) + right).astype('uint8')
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    ans = cv2.dilate(ans,kernel,iterations = 1)

    gray = cv2.cvtColor(ans, cv2.COLOR_RGB2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]

    # Finds contours from the binary image
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # get the maximum contour area
    c = max(cnts, key=cv2.contourArea)

    # get a bbox from the contour area
    (x, y, w, h) = cv2.boundingRect(c)

    # crop the image to the bbox coordinates
    ans = ans[y:y + h, x:x + w]

    # plt.imshow(ans/255)
    # plt.show()
    
    return ans

def add_image(
    panorama: np.ndarray, image: Image, offset: np.ndarray, weights: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Add a new image to the panorama using the provided offset and weights.

    Parameters
    ----------
    panorama : np.ndarray
        Existing panorama.
    image : Image
        Image to add to the panorama.
    offset : np.ndarray
        Offset already applied to the panorama.
    weights : np.ndarray
        Weights matrix of the panorama.

    Returns
    -------
    panorama : np.ndarray
        Panorama with the new image.
    offset : np.ndarray
        New offset matrix.
    weights : np.ndarray
        New weights matrix.
    """

    H = offset @ image.H
    size, added_offset = get_new_parameters(panorama, image.image, H)

    new_image = cv2.warpPerspective(image.image, added_offset @ H, size)

    if panorama is None:
        panorama = np.zeros_like(new_image)
        weights = np.zeros_like(new_image)
    else:
        panorama = cv2.warpPerspective(panorama, added_offset, size)
        weights = cv2.warpPerspective(weights, added_offset, size)

    image_weights = single_weights_matrix(image.image.shape)
    image_weights = np.repeat(
        cv2.warpPerspective(image_weights, added_offset @ H, size)[:, :, np.newaxis], 3, axis=2
    )

    normalized_weights = np.zeros_like(weights)
    normalized_weights = np.divide(
        weights, (weights + image_weights), where=weights + image_weights != 0
    )

    panorama = np.where(
        np.logical_and(
            np.repeat(np.sum(panorama, axis=2)[:, :, np.newaxis], 3, axis=2) == 0,
            np.repeat(np.sum(new_image, axis=2)[:, :, np.newaxis], 3, axis=2) == 0,
        ),
        0,
        new_image * (1 - normalized_weights) + panorama * normalized_weights,
    ).astype(np.uint8)

    new_weights = (weights + image_weights) / (weights + image_weights).max()

    return panorama, added_offset @ offset, new_weights


def simple_blending(images,pair_match) -> np.ndarray:
    """
    Build a panorama for the given images using simple blending.

    Parameters
    ----------
    images : List[Image]
        Images to build the panorama for.

    Returns
    -------
    panorama : np.ndarray
        Panorama of the given images.
    """

    panorama = None
    weights = None
    offset = np.eye(3)
    for image in images:
        panorama, offset, weights = add_image(panorama, image, offset, weights)

    return panorama
