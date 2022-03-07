import cv2
from skimage import exposure
import numpy as np
from images import Image
from typing import List
import os
import matplotlib.pyplot as plt
import imageio
from matching import MultiImageMatches, PairMatch, build_homographies, find_connected_components
from rendering import multi_band_blending, set_gain_compensations, simple_blending,brute_force_blend

def hist_match(source, template):
    """
    Adjust the pixel values such that its histogram
    matches that of a target image

    Arguments:
    -----------
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        template: np.ndarray
            Template image; can have different dimensions to source
    Returns:
    -----------
        matched: np.ndarray
            The transformed output image
    """

    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)


#left image
image1 = Image('/home/rushali/CV702_Assignment/Project/image pairs_02_02.png')
#right image
image2 = Image('/home/rushali/CV702_Assignment/Project/image pairs_02_01.png')

image1.image = hist_match(image1.image,image2.image).astype('uint8')

images = [image1,image2]

method = 'sift'

print('("Computing features...")')


for image in images:
	image.compute_features(method)

matcher = MultiImageMatches(images)
pair_match = matcher.get_pair_matches()


build_homographies(pair_match)

#print('("Computing gain compensations...")')

# set_gain_compensations(
#         images,
#         pair_matches,
#         sigma_n=10,
#         sigma_g=0.1,
#     )

# for image in images:
#     image.image = (image.image * image.gain[np.newaxis, np.newaxis, :]).astype(np.uint8)

os.makedirs(os.path.join("./", "results"), exist_ok=True)

result_brute_force = brute_force_blend(images,pair_match)
cv2.imwrite(os.path.join("./", "results", f"pano_brute_force.jpg"), result_brute_force)

result_simple_blend = simple_blending(images,pair_match)
cv2.imwrite(os.path.join("./", "results", f"pano_simple_blend.jpg"), result_simple_blend)

result_multi_band_blend = multi_band_blending(images, num_bands=4, sigma=0.1)
cv2.imwrite(os.path.join("./", "results", f"pano_multi_band_blend.jpg"), result_multi_band_blend)