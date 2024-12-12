import time

import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2
import pyautogui as pag
from math import sqrt
import platform


def draw_landmarks_on_image(rgb_image, detection_result):
	hand_landmarks_list = detection_result.hand_landmarks
	new_image = np.copy(rgb_image)

	# Loop through the detected hands to visualize.
	for idx in range(len(hand_landmarks_list)):
		hand_landmarks = hand_landmarks_list[idx]

		# Draw the hand landmarks.
		hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
		hand_landmarks_proto.landmark.extend([
			landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
		])
		solutions.drawing_utils.draw_landmarks(
			new_image,
			hand_landmarks_proto,
			solutions.hands.HAND_CONNECTIONS,
			solutions.drawing_styles.get_default_hand_landmarks_style(),
			solutions.drawing_styles.get_default_hand_connections_style())

	return new_image


def get_average_hand_location(detection_result):
	cur = detection_result.hand_landmarks[0]
	avg_x = SCREEN_WIDTH - ((sum([cur[idx].x for idx in range(len(cur))]) / len(cur)) * SCREEN_WIDTH)
	avg_y = (sum([cur[idx].y for idx in range(len(cur))]) / len(cur)) * SCREEN_HEIGHT
	return avg_x, avg_y

def is_pinch(detection_result):
	cur = detection_result.hand_landmarks[0]
	thumb = (cur[4].x, cur[4].y, cur[4].z)
	index = (cur[8].x, cur[8].y, cur[8].z)
	if sqrt((thumb[0] - index[0]) ** 2 + (thumb[1] - index[1]) ** 2 + (thumb[2] - index[2]) ** 2) < CLICK_THRESHOLD:
		return True
	return False

def get_result(result, output_image, timestamp_ms):
	global img, landmarks, prev_time
	img = output_image.numpy_view()
	prev_time = timestamp_ms
	if len(result.hand_landmarks) > 0:
		landmarks = result
	else:
		landmarks = None


BaseOptions = mp.tasks.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FPS, 60)

PLATFORM = platform.system()

SCREEN_WIDTH, SCREEN_HEIGHT = pag.size()
CLICK_THRESHOLD = 0.08
DRAG_THRESHOLD_SECS = 0.5
click = 0

# Create a hand landmarker instance with the image mode:
options = HandLandmarkerOptions(
	base_options=BaseOptions(model_asset_path=r"/Users/paul/Documents/Python/handControl/hand_landmarker.task",
	                         delegate=BaseOptions.Delegate.GPU),
	num_hands=1,
	running_mode=VisionRunningMode.LIVE_STREAM,
	result_callback=get_result)
with HandLandmarker.create_from_options(options) as landmarker:
	timestamp = time.time()
	prev_time = timestamp
	img = cam.read()[1]
	landmarks = None
	while True:
		ret, frame = cam.read()

		# Display the captured frame
		if frame is not None:
			mp_image = mp.Image(image_format=mp.ImageFormat.SRGBA, data=cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA))
			landmarker.detect_async(mp_image, int((time.time() - timestamp) * 1000))

		if landmarks is not None:
			land_img = draw_landmarks_on_image(cv2.cvtColor(img, cv2.COLOR_RGBA2RGB), landmarks)
			hand = get_average_hand_location(landmarks)
			pinch = is_pinch(landmarks)
			if pinch and click == 0:
				click = time.time()
			elif pinch and click > 0 and time.time() - click > DRAG_THRESHOLD_SECS:
				pag.mouseDown()
				click = -1
			elif not pinch and click > 0:
				pag.leftClick()
				click = 0
			elif not pinch and click == -1:
				pag.mouseUp()
				click = 0

			if click >= 0 or platform not in ["Windows", "Linux"]:
				pag.moveTo(hand[0], hand[1], _pause=False)
			else:
				pag.dragTo(hand[0], hand[1], button="left", _pause=False)
			cv2.imshow('Camera', land_img)
		else:
			cv2.imshow('Camera', img)

		# Press 'q' to exit the loop
		if cv2.waitKey(1) == ord('q'):
			break

	# Release the capture object
	cam.release()
	cv2.destroyAllWindows()
