# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
"""Example Google style docstrings.

This module demonstrates documentation as specified by the `Google Python
Style Guide`_. Docstrings may extend over multiple lines. Sections are created
with a section header and a colon followed by a block of indented text.

Example:
    Examples can be given using either the ``Example`` or ``Examples``
    sections. Sections support any reStructuredText formatting, including
    literal blocks::

        $ python example_google.py

Section breaks are created by resuming unindented text. Section breaks
are also implicitly created anytime a new section starts.

Attributes:
    module_level_variable1 (int): Module level variables may be documented in
        either the ``Attributes`` section of the module docstring, or in an
        inline docstring immediately following the variable.

        Either form is acceptable, but the two should not be mixed. Choose
        one convention to document module level variables and be consistent
        with it.

Todo:
    * For module TODOs
    * You have to also use ``sphinx.ext.todo`` extension

.. _Google Python Style Guide:
   http://google.github.io/styleguide/pyguide.html

"""
#-----------------------------------
# GLOBAL FEATURE EXTRACTION
#-----------------------------------

import numpy as np
import mahotas
import cv2


class Global_feature_extraction:
    """Exceptions are documented in the same way as classes.

    The __init__ method may be documented in either the class level
    docstring, or as a docstring on the __init__ method itself.

    Either form is acceptable, but the two should not be mixed. Choose one
    convention to document the __init__ method and be consistent with it.

    Note:
        Do not include the `self` parameter in the ``Args`` section.

    Args:
        msg (str): Human readable string describing the exception.
        code (:obj:`int`, optional): Error code.

    Attributes:
        msg (str): Human readable string describing the exception.
        code (int): Exception error code.

    """
    bins = 8

    # feature-descriptor-1: Hu Moments
    def shape(self,image):
        """Example function with types documented in the docstring.

    `PEP 484`_ type annotations are supported. If attribute, parameter, and
    return types are annotated according to `PEP 484`_, they do not need to be
    included in the docstring:

    Args:
        param1 (int): The first parameter.
        param2 (str): The second parameter.

    Returns:
        bool: The return value. True for success, False otherwise.

    .. _PEP 484:
        https://www.python.org/dev/peps/pep-0484/

        """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        feature = cv2.HuMoments(cv2.moments(image)).flatten()
        return feature

    # feature-descriptor-2: Haralick Texture
    def texture(self,image):
        """Example function with types documented in the docstring.

        `PEP 484`_ type annotations are supported. If attribute, parameter, and
        return types are annotated according to `PEP 484`_, they do not need to be
        included in the docstring:
            
        Args:
            param1 (int): The first parameter.
            param2 (str): The second parameter.
            
            Returns:
                bool: The return value. True for success, False otherwise.
                
                .. _PEP 484:
                    https://www.python.org/dev/peps/pep-0484/
        """
        # convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # compute the haralick texture feature vector
        haralick = mahotas.features.haralick(gray).mean(axis=0)
        # return the result
        return haralick

    # feature-descriptor-3: Color Histogram
    def color(self,image, mask=None):
        """Example function with types documented in the docstring.

        `PEP 484`_ type annotations are supported. If attribute, parameter, and
        return types are annotated according to `PEP 484`_, they do not need to be
        included in the docstring:
            
        Args:
            param1 (int): The first parameter.
            param2 (str): The second parameter.
            
            Returns:
                bool: The return value. True for success, False otherwise.
                
                .. _PEP 484:
                    https://www.python.org/dev/peps/pep-0484/
        """
        # convert the image to HSV color-space
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # compute the color histogram
        hist  = cv2.calcHist([image], [0, 1, 2], None, [self.bins, self.bins, self.bins], [0, 256, 0, 256, 0, 256])
        # normalize the histogram
        cv2.normalize(hist, hist)
        # return the histogram
        return hist.flatten()

    def edge_detecetion(self,img):
        """Example function with types documented in the docstring.

        `PEP 484`_ type annotations are supported. If attribute, parameter, and
        return types are annotated according to `PEP 484`_, they do not need to be
        included in the docstring:
            
        Args:
            param1 (int): The first parameter.
            param2 (str): The second parameter.
            
            Returns:
                bool: The return value. True for success, False otherwise.
                
                .. _PEP 484:
                    https://www.python.org/dev/peps/pep-0484/
        """
        img = cv2.imread(img, 0)
        cv2.imwrite("canny.jpg", cv2.Canny(img, 200,300))
        
    def some_feature(self):
        """Example function with types documented in the docstring.

        `PEP 484`_ type annotations are supported. If attribute, parameter, and
        return types are annotated according to `PEP 484`_, they do not need to be
        included in the docstring:
            
        Args:
            param1 (int): The first parameter.
            param2 (str): The second parameter.
            
            Returns:
                bool: The return value. True for success, False otherwise.
                
                .. _PEP 484:
                    https://www.python.org/dev/peps/pep-0484/
        """
        """ mahdi : I have some features code in my laptop that can help us .
        so what if i add them in this place as method.
        myself : it's grate  :) 
        """
        pass
    
        