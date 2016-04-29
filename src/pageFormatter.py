
"""
Format one page of score into appropriate size (with a constant staff gap)
"""


class pageFormatter:
    """
    Page Formatter Class
    """
    def __init__(self, image):
        """
        Initialize a page formatter
        :param image: input image
        :type image: cv2.image
        """
        self.image = image
        self.scale = 0.0