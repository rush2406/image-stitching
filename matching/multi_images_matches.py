from typing import List
import cv2
from images import Image
from matching.pair_match import PairMatch


class MultiImageMatches:
    def __init__(self, images: List[Image], ratio: float = 0.75):
        """
        Create a new MultiImageMatches object.

        Parameters
        ----------
        images : List[Image]
            images to compare
        ratio : float, optional
            ratio used for the Lowe's ratio test, by default 0.75
        """

        self.images = images
        self.matches = {image.path: {} for image in images}
        self.ratio = ratio

    def get_matches(self, image_a: Image, image_b: Image) -> List:
        """
        Get matches for the given images.

        Parameters
        ----------
        image_a : Image
            First image.
        image_b : Image
            Second image.

        Returns
        -------
        matches : List
            List of matches between the two images.
        """
        if image_b.path not in self.matches[image_a.path]:
            matches = self.compute_matches(image_a, image_b)
            self.matches[image_a.path][image_b.path] = matches

        return self.matches[image_a.path][image_b.path]

    def get_pair_matches(self, max_images: int = 6) -> List[PairMatch]:
        """
        Get the pair matches for the given images
        """
        return PairMatch(self.images[0], self.images[1], self.get_matches(self.images[0], self.images[1]))

    def compute_matches(self, image_a: Image, image_b: Image) -> List:
        """
        Compute matches between image_a and image_b.

        Parameters
        ----------
        image_a : Image
            First image.
        image_b : Image
            Second image.

        Returns
        -------
        matches : List
            Matches between image_a and image_b.
        """
        
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        matches = []

        rawMatches = matcher.knnMatch(image_a.features, image_b.features, 2)

        for m, n in rawMatches:
            # ensure the distance is within a certain ratio of each
            # other (i.e. Lowe's ratio test)
            if m.distance < n.distance * self.ratio:
                matches.append(m)

        return matches
