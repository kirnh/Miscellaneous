import torch
import os
import sys
import time
import datetime
import argparse
import cv2
from PIL import Image
from skimage.transform import resize

from deep_sort.tools import generate_detections as gd

from YOLOv3.models import *
from YOLOv3.utils.utils import *
# from YOLOv3.utils.datasets import *

class Detector():
	def __init__(self, images_dir):
		self.images_dir = images_dir
		try:
			sample_image = cv2.imread(os.path.join(images_dir, os.listdir(images_dir)[0]))
			self.image_shape = sample_image.shape
		except:
			print('No images found in {}'.format(images_dir))
		self.image_id = 1

	def setup_YOLO(self, config_path, weights_path, img_size=416):
		# Set up YOLOv3 model
		self.YOLOv3_model = Darknet(config_path, img_size=img_size)
		self.YOLOv3_model.load_weights(weights_path)
		# self.YOLOv3_model = self.YOLOv3_model.cuda()
		# Set to eval mode
		self.YOLOv3_model.eval()

	def setup_feature_extractor(self, weights_path, batch_size=1):
		self.reid_feature_extractor = gd.create_box_encoder(weights_path, batch_size=batch_size)

	def create_image_tensor(self, image_path):
		img = np.array(Image.open(image_path))
		h, w, _ = img.shape
		dim_diff = np.abs(h - w)
		# Upper (left) and lower (right) padding
		pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
		# Determine padding
		pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
		# Add padding
		input_img = np.pad(img, pad, 'constant', constant_values=127.5) / 255.
		# Resize and normalize
		input_img = resize(input_img, (416, 416, 3), mode='reflect')
		# Channels-first
		input_img = np.transpose(input_img, (2, 0, 1))
		# As pytorch tensor
		input_img = torch.from_numpy(np.array([input_img])).float()
		return input_img, img

	def get_detections(self, image_path, conf_thres=0.8, nms_thres=0.4, img_size=416):
		image_tensor, self.image = self.create_image_tensor(image_path)
		self.image = cv2.cvtColor(np.array(self.image), cv2.COLOR_RGB2BGR)
		with torch.no_grad():
			# print('---------------0--')
			# print(image_tensor.shape)
			ti = time.time()
			detections = self.YOLOv3_model(image_tensor)
			to = time.time()
			print('YOLO: '+str(to-ti)+'s')
			detections = non_max_suppression(detections, 80, conf_thres, nms_thres)
		# print('------------------------------------YOLO ran---')
		# The amount of padding that was added
		pad_x = max(self.image_shape[0] - self.image_shape[1], 0) * (img_size / max(self.image_shape))
		pad_y = max(self.image_shape[1] - self.image_shape[0], 0) * (img_size / max(self.image_shape))
		# Image height and width after padding is removed
		unpad_h = img_size - pad_y
		unpad_w = img_size - pad_x
		bbox_list = []
		for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections[0]:
			# Only add detections of people
			if int(cls_pred) != 0:
				continue
			# Rescale coordinates to original dimensions
			box_h = ((y2 - y1) / unpad_h) * self.image_shape[0]
			box_w = ((x2 - x1) / unpad_w) * self.image_shape[1]
			y1 = ((y1 - pad_y // 2) / unpad_h) * self.image_shape[0]
			x1 = ((x1 - pad_x // 2) / unpad_w) * self.image_shape[1]
			bbox = np.array([x1, y1, box_w, box_h])
			bbox_list.append(bbox) 
		ti = time.time()
		reid_features = self.reid_feature_extractor(self.image, bbox_list)
		to = time.time()
		print('Re-ID feature extraction: '+str(to-ti)+'s')
		return_object = [np.r_[self.image_id, -1, bbox, cls_conf, -1, -1, -1, reid_feature] for bbox, reid_feature in zip(bbox_list, reid_features)]
		return return_object

###########################################################################################################################################################

images_dir = '/home/kh/Miscellaneous/People_Tracking/images'

yolo_config = '/home/kh/Miscellaneous/People_Tracking/YOLOv3/config/yolov3.cfg'
yolo_weights = '/home/kh/Miscellaneous/People_Tracking/YOLOv3/weights/yolov3.weights'

reid_net_weights = '/home/kh/Miscellaneous/People_Tracking/deep_sort/resources/networks/mars-small128.pb'

detector = Detector(images_dir)
detector.setup_YOLO(yolo_config, yolo_weights)

detector.setup_feature_extractor(reid_net_weights)

t1 = time.time()
detections = detector.get_detections('/home/kh/Miscellaneous/People_Tracking/images/000038.jpg')
t2 = time.time()
detections = detector.get_detections('/home/kh/Miscellaneous/People_Tracking/images/000038.jpg')
t3 = time.time()
detections = detector.get_detections('/home/kh/Miscellaneous/People_Tracking/images/000038.jpg')
t4 = time.time()

print('First image: {}s'.format(t2-t1))
print('Second image: {}s'.format(t3-t2))
print('Third image: {}s'.format(t4-t3))


