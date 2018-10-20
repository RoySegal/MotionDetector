import os
import sys
import string
import datetime
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import smtplib
import time
import httplib
import urllib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from RT_config import *

HORIZONTAL_DIFF = 5
VERTICAL_DIFF = 10
FPS = 1
DIR_PATH = r"C:\temp\motiondetection\Video"
TIME_DIFF = 60 # seconds

LAST_DIFF_TIME = 0

def min_sqaured(first_img_data, second_img_data):
	if len(first_img_data) != len(second_img_data):
		raise "Wrong data size"

	accumulate_pixels = 0
	for pixels in zip(first_img_data[0], second_img_data[0]):
		accumulate_pixels += (int(pixels[0]) - int(pixels[1]))**2
	return int(accumulate_pixels**0.5)

def compare_images(first_image_path, second_image_path):
	prev_image = None
	
def y_distance(x_value, y_value, pol):
	return abs(y_value-pol[0]*x_value-pol[1])

def calculate_fit(result_matrices, result_matrix, row, col):
	if len(result_matrices) < 2:
		return np.array([0, 0], dtype='float64')
	values = []
	for matrix in result_matrices:
		values.append(matrix[row][col])
	values.append(result_matrix[row][col])
	return np.polyfit(range(len(values)), values, deg=1)

def send_push():
	conn = httplib.HTTPSConnection("api.pushover.net:443")
	conn.request("POST", "/1/messages.json",
  				urllib.urlencode({
    			"token": TOKEN,
			    "user": USER,
    			"message": "Movement Occurred"
  				}), { "Content-type": "application/x-www-form-urlencoded" })
	conn.getresponse()

def rt_video_detection():
	prev_image = None
	result_matrices = []
	pol_arr = np.array([[[0,0]]*VERTICAL_DIFF]*HORIZONTAL_DIFF, dtype='float64')
	dist = 0
	cap = cv2.VideoCapture(0)
	if cap.isOpened() is False:
		raise "Unable to open video file"

	frame_counter = 0
	global LAST_DIFF_TIME
	while cap.isOpened() is True:
		ret, frame = cap.read()
		if ret is False:
			break

		if prev_image is None:
			prev_image = np.array(Image.fromarray(frame).convert(mode='L').resize((600,600), Image.ANTIALIAS))
			continue

		frame_counter += 1

		if frame_counter*FPS % cap.get(cv2.CAP_PROP_FPS) != 0:
			continue

		cur_image = np.array(Image.fromarray(frame).convert(mode='L').resize((600,600), Image.ANTIALIAS))
		horizontal = prev_image.shape[0]/HORIZONTAL_DIFF
		vertical = prev_image.shape[1]/VERTICAL_DIFF
		result_matrix = np.zeros((prev_image.shape[0]/horizontal, prev_image.shape[1]/vertical))
		for row in xrange(result_matrix.shape[0]):
			for col in xrange(result_matrix.shape[1]):
				result_matrix[row][col] = min_sqaured(prev_image[row*horizontal:(row+1)*horizontal, col*vertical:(col+1)*vertical].reshape(1, horizontal*vertical),
					cur_image[row*horizontal:(row+1)*horizontal, col*vertical:(col+1)*vertical].reshape(1, horizontal*vertical))
				if len(result_matrices) >= 10:
					dist = y_distance(col, result_matrix[row][col], pol_arr[row][col])
					#print 'row: ' + str(row) + ' ' + 'col: ' + str(col) + ' ' + str(y_distance(col, result_matrix[row][col], pol_arr[row][col]))
					if col == 9 and row == 9:
						pass #print image_path + '   ' + str(y_distance(col, result_matrix[row][col], pol_arr[row][col]))
					pol_arr[row][col] = calculate_fit(result_matrices, result_matrix, row, col)
				cur_time = time.time()
				if dist > 9000:
					print dist
					dist = 0
					# Save the image
					time_to_stamp = string.split(datetime.datetime.now().isoformat(),'.')[0].replace(':','_')
					#Image.fromarray(prev_image[row*horizontal:(row+1)*horizontal, col*vertical:(col+1)*vertical]).save(os.path.join(DIR_PATH, 'detection\\') + str(frame_counter/cap.get(cv2.CAP_PROP_FPS)) + '_' + time_to_stamp + '_dist_' + str(int(dist)) + '_0.jpg')
					#Image.fromarray(cur_image[row*horizontal:(row+1)*horizontal, col*vertical:(col+1)*vertical]).save(os.path.join(DIR_PATH, 'detection\\') + str(frame_counter/cap.get(cv2.CAP_PROP_FPS)) + '_' + time_to_stamp + '_dist_' + str(int(dist)) + '_1.jpg')
					Image.fromarray(cur_image).save(os.path.join(DIR_PATH, 'detection\\') + str(frame_counter/cap.get(cv2.CAP_PROP_FPS)) + '_' + time_to_stamp + '_dist_' + str(int(dist)) + '_2.jpg')
					if cur_time - LAST_DIFF_TIME > TIME_DIFF:
						send_push()
						print 'time passed: ' + str(cur_time - LAST_DIFF_TIME)
						LAST_DIFF_TIME = cur_time

		result_matrices.append(result_matrix)
		if len(result_matrices) > 10:
			result_matrices.pop(0)
		prev_image = cur_image


if __name__ == '__main__':
	rt_video_detection()
	
	#compare_images(sys.argv[1], sys.argv[2])