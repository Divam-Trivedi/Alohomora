#!/usr/bin/env python3

"""
RBE/CS549 Spring 2022: Computer Vision
Homework 0: Alohomora: Phase 1 Starter Code

Colab file can be found at:
	https://colab.research.google.com/drive/1FUByhYCYAfpl8J9VxMQ1DcfITpY8qgsF

Author(s): 
Prof. Nitin J. Sanket (nsanket@wpi.edu), Lening Li (lli4@wpi.edu), Gejji, Vaishnavi Vivek (vgejji@wpi.edu)
Robotics Engineering Department,
Worcester Polytechnic Institute

Code adapted from CMSC733 at the University of Maryland, College Park.
"""

# Code starts here:

import numpy as np
import cv2
import os
import DoG
import LM
import Gabor
import Half_Disk
import Texton
import Brightness
import Color

image_save_path = 'dtrivedi_hw0/Phase1/BSDS500'
def main(image_save_path = image_save_path):

	"""
	Generate Difference of Gaussian Filter Bank: (DoG)
	Display all the filters in this filter bank and save image as DoG.png,
	use command "cv2.imwrite(...)"
	"""
	scales = [1, 2]  # Two scales
	orientations = 16  # Sixteen orientations
	size = 15
	save_path = image_save_path + '/DoG.png'
	DoG_Filter_Bank = DoG.oriented_dog_filters(scales, orientations, size, save_path, display=False)
	"""
	Generate Leung-Malik Filter Bank: (LM)
	Display all the filters in this filter bank and save image as LM.png,
	use command "cv2.imwrite(...)"
	"""
	filt_type = "LML" # or "LMS"
	filters = LM.makeLMfilters(filt_type) 
	LM.visualize_filters(filters, image_save_path + f'/{filt_type}.png', display=False)
	"""
	Generate Gabor Filter Bank: (Gabor)
	Display all the filters in this filter bank and save image as Gabor.png,
	use command "cv2.imwrite(...)"
	"""
	size = 64
	num_filters = 40
	wavelengths = [10, 15, 20]
	orientations = np.linspace(0, np.pi, 8)
	sigmas = [10, 15]
	aspect_ratios = [0.5, 1.0]
	phases = [0, np.pi/2]
	gabor_filters = Gabor.generate_gabor_filters(num_filters, size, wavelengths, orientations, sigmas, aspect_ratios, phases)
	Gabor.visualize_filters(gabor_filters, image_save_path + '/Gabor.png', display=False)
	"""
	Generate Half-disk masks
	Display all the Half-disk masks and save image as HDMasks.png,
	use command "cv2.imwrite(...)"
	"""
	radius = [4, 10, 20]
	orientations = [0.0, 180, 22.5, 202.5, 45, 225, 67.5, 247.5, 90, 270, 112.5, 292.5, 135.0, 315, 157.5, 337.5] # in degrees
	half_disk_masks = Half_Disk.generate_half_disk_masks(radius, orientations)
	Half_Disk.visualize_filters(half_disk_masks, image_save_path + '/HDMasks.png', display=False)
	"""
	Generate Texton Map
	Filter image using oriented gaussian filter bank
	"""
	image_folder = image_save_path + '/Images'
	image_paths = [image_folder + os.path.sep + f'{f}.jpg' for f in range(1, 11)]
	filter_bank = DoG_Filter_Bank ## Use the filter bank as needed
	texton_maps = Texton.generate_texton_map(image_paths, filter_bank)
	"""
	Generate texture ID's using K-means clustering
	Display texton map and save image as TextonMap_ImageName.png,
	use command "cv2.imwrite('...)"
	"""
	save_image_path = image_save_path + "/Texton_Maps"
	texture_ids = Texton.generate_texture_ids(image_paths, texton_maps, 128, save_image_path)
	"""
	Generate Texton Gradient (Tg)
	Perform Chi-square calculation on Texton Map
	Display Tg and save image as Tg_ImageName.png,
	use command "cv2.imwrite(...)"
	"""
	save_image_path = image_save_path + "/Texton_Gradient"
	masks = half_disk_masks
	texton_gradients = Texton.generate_texton_gradient(image_paths, texton_maps, 128, masks, save_image_path)
	"""
	Generate Brightness Map
	Perform brightness binning 
	"""
	save_image_path = image_save_path + "/Brightness_Maps"
	brightness_maps = Brightness.generate_brightness_map(image_paths, 16, save_image_path)
	"""
	Generate Brightness Gradient (Bg)
	Perform Chi-square calculation on Brightness Map
	Display Bg and save image as Bg_ImageName.png,
	use command "cv2.imwrite(...)"
	"""
	save_image_path = image_save_path + "/Brightness_Gradient"
	masks = half_disk_masks
	brightness_gradients = Brightness.generate_brightness_gradient_with_masks(image_paths, brightness_maps, 16, masks, save_image_path)
	"""
	Generate Color Map
	Perform color binning or clustering
	"""
	save_image_path = image_save_path + "/Color_Maps"
	# color_maps = Color.generate_color_map(image_paths, 16, 'RGB', image_save_path)
	# color_maps = Color.generate_color_map(image_paths, 16, 'YCbCr', image_save_path)
	# color_maps = Color.generate_color_map(image_paths, 16, 'HSV', image_save_path)
	color_maps = Color.generate_color_map(image_paths, 16, 'Lab', save_image_path)
	"""
	Generate Color Gradient (Cg)
	Perform Chi-square calculation on Color Map
	Display Cg and save image as Cg_ImageName.png,
	use command "cv2.imwrite(...)"
	"""
	masks = half_disk_masks
	save_image_path = image_save_path + "/Color_Gradient"
	color_gradients = Color.generate_color_gradient_with_masks(image_paths, color_maps, 16, masks, save_image_path)
	"""
	Read Sobel Baseline
	use command "cv2.imread(...)"
	"""
	sobel_images_path = image_save_path + "/SobelBaseline"
	sobel_img = [sobel_images_path + os.path.sep + f'{f}.png' for f in range(1, 11)]
	sobel_images = [None] * len(sobel_img)
	for i in range(len(sobel_img)):
		sobel_images[i] = cv2.imread(sobel_img[i])
	"""
	Read Canny Baseline
	use command "cv2.imread(...)"
	"""
	canny_images_path = image_save_path + '/CannyBaseline'
	canny_img = [canny_images_path + os.path.sep + f'{f}.png' for f in range(1, 11)]
	canny_images = [None] * len(canny_img)
	for i in range(len(canny_img)):
		canny_images[i] = cv2.imread(canny_img[i])
	"""
	Combine responses to get pb-lite output
	Display PbLite and save image as PbLite_ImageName.png
	use command "cv2.imwrite(...)"
	"""
	def rgb2gray(rgb):
		return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

	Tg = texton_gradients
	Bg = brightness_gradients
	Cg = color_gradients
	Sg = sobel_images
	Cg = canny_images

	save_image_path = image_save_path + "/PbLite_Images"

	for i in range(10):
		w1 = 0.5
		w2 = 0.5
		a = (rgb2gray(Tg[i]) * Bg[i] * rgb2gray(Cg[i]))
		b = w1 * rgb2gray(Sg[i]) + w2 * rgb2gray(Cg[i])

		PbEdges = a * b

		cv2.imwrite(save_image_path + os.path.sep + f'{i+1}.png', PbEdges)
    
if __name__ == '__main__':
    main()