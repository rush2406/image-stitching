import cv2
import numpy as np
import imageio


class Image:
    def __init__(self, path: str, size: int = None):
        """
        Image constructor.

        Parameters
        ----------
        path : str
            path to the image
        size : int, optional
            maximum dimension to resize the image to, by default None
        """
        self.path = path
        self.image: np.ndarray = cv2.imread(path)
        if size is not None:
            h, w = self.image.shape[:2]
            if max(w, h) > size:
                if w > h:
                    self.image = cv2.resize(self.image, (size, int(h * size / w)))
                else:
                    self.image = cv2.resize(self.image, (int(w * size / h), size))

        self.keypoints = None
        self.features = None
        self.H: np.ndarray = np.eye(3)
        self.component_id: int = 0
        self.gain: np.ndarray = np.ones(3, dtype=np.float32)

    def compute_features(self,method):
        """
        Compute the features and the keypoints of the image using SIFT.
        """

        if method == 'sift':
            descriptor = cv2.SIFT_create()
        elif method == 'brisk':
            descriptor = cv2.BRISK_create()
        elif method == 'orb':
            descriptor = cv2.ORB_create()

        keypoints, features = descriptor.detectAndCompute(self.image, None)
        self.keypoints = keypoints
        self.features = features