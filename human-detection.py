# Imports
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import cv2
import time
import sched
import threading
import numpy as np
import serial
import pygame

# Configuration
MODEL_PATH = '/home/ritcheydominic/RLGL/pose_landmarker_lite.task'
GREEN_LIGHT_MP3_FILE = "/home/ritcheydominic/RLGL/KoreanRedLightGreenLight.mp3"
DEBUG = True
MAX_PLAYERS = 4

MIN_POSE_DETECTION_CONFIDENCE = 0.5
MIN_POSE_PRESENCE_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5
TARGET_FPS = 5
SELECTED_CAMERA = 0
CAMERA_FOV = 75 # But should be probably be 90 (or 120 for Raspberry Pi Camera Module)
CAMERA_FRAME_WIDTH = 640
CAMERA_FRAME_HEIGHT = 480

WINNING_CHEST_AREA = 35000
MOVEMENT_CHEST_AREA_CHANGE_THRESHOLD = 0.1
MOVEMENT_WINGSPAN_CHANGE_THRESHOLD = 0.2
MOVEMENT_CENTER_X_CHANGE_THRESHOLD = 0.18
MOVEMENT_VECTOR_NORM_THRESHOLD = 36
MOVEMENT_COUNT_THRESHOLD = 3

# Variables
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode
arduino = serial.Serial(port='/dev/ttyACM0', baudrate = 115200, timeout=None) # TODO: Timeout

# Functions
def write_to_serial(x):
    arduino.write(bytes(x, 'utf-8'))
    return x

def calculate_relative_change(initial, final):
    # print("Initial: {}\tFinal: {}\tResult: {}".format(initial, final, (final - initial) / initial))
    return np.absolute((final - initial) / initial)

def calculate_center_of_landmarks(landmarks):
    if len(landmarks) == 0:
        return None
    
    mean_x = sum([landmark.x for landmark in landmarks]) / len(landmarks)
    mean_y = sum([landmark.y for landmark in landmarks]) / len(landmarks)
    mean_z = sum([landmark.z for landmark in landmarks]) / len(landmarks)

    return (int(mean_x), int(mean_y), mean_z)


def calculate_angle_from_center(center, frame_width, fov=75):
    # Calculate the horizontal offset from the center of the frame
    offset = center - frame_width / 2
    # Calculate angle
    angle = (offset / frame_width) * fov
    return angle

# def calculate_movement(prev_keypoints, curr_keypoints):
#     if len(prev_keypoints) == 0 or len(curr_keypoints) == 0:
#         print("ERR 1")
#         return 0
#     if len(prev_keypoints) != len(curr_keypoints): # Cannot calculate movement if number of keypoints is different
#         print("ERR 2")
#         return 0
#     print("ERR 3")
#     ret = []
#     for i in range(len(curr_keypoints)): # Separates human
#         curr_keypoints_splitout = []
#         prev_keypoints_splitout = []
#         for j in range(len(curr_keypoints[i])): # Separates actual landmarks
#             curr_keypoints_splitout.append([curr_keypoints[i][j].x, curr_keypoints[i][j].y])
#             prev_keypoints_splitout.append([prev_keypoints[i][j].x, prev_keypoints[i][j].y])
#         # print(curr_keypoints_splitout)
#         ret.append(np.linalg.norm(np.array(curr_keypoints_splitout[i]) - np.array(prev_keypoints_splitout[i]), axis=0).max())
#     return ret
#     # return np.linalg.norm(prev_landmarks - curr_keypoints, axis=1).max()

def calculate_vector_norm(prev_keypoints, curr_keypoints):
    curr_keypoints_splitout = []
    prev_keypoints_splitout = []
    for j in range(len(curr_keypoints)): # Separates actual landmarks
        curr_keypoints_splitout.append([curr_keypoints[j].x, curr_keypoints[j].y])
        prev_keypoints_splitout.append([prev_keypoints[j].x, prev_keypoints[j].y])
    return np.linalg.norm(np.array(curr_keypoints_splitout) - np.array(prev_keypoints_splitout), axis=0).max()

def calculate_chest_area(pose_landmarks):
    left_shoulder_x = pose_landmarks[11].x
    left_shoulder_y = pose_landmarks[11].y
    right_shoulder_x = pose_landmarks[12].x
    right_shoulder_y = pose_landmarks[12].y
    left_hip_x = pose_landmarks[23].x
    left_hip_y = pose_landmarks[23].y
    right_hip_x = pose_landmarks[24].x
    right_hip_y = pose_landmarks[24].y
    x_values = np.array([left_shoulder_x, right_shoulder_x, left_hip_x, right_hip_x])
    y_values = np.array([left_shoulder_y, right_shoulder_y, left_hip_y, right_hip_y])
    x_min = x_values.min()
    x_max = x_values.max()
    y_min = y_values.min()
    y_max = y_values.max()
    x_length = x_max - x_min
    y_length = y_max - y_min
    return x_length * y_length

def calculate_wingspan(pose_landmarks):
    left_wrist_x = pose_landmarks[15].x
    left_wrist_y = pose_landmarks[15].y
    right_wrist_x = pose_landmarks[16].x
    right_wrist_y = pose_landmarks[16].y
    return np.sqrt(((left_wrist_x - right_wrist_x) ** 2) + ((left_wrist_y - right_wrist_y) ** 2))

def on_result_ready(result, output_image, timestamp_ms):
    # Call which detect function based on state
    global current_stage
    if current_stage == 0:
        detect_winners(result, output_image, timestamp_ms)
    elif current_stage == 2:
        detect_victims(result, output_image, timestamp_ms)

def detect_winners(result, output_image, timestamp_ms):
    return result

# def detect_victims(result, output_image, timestamp_ms):
#     # if DEBUG == True:
#     #     print('pose landmarker result: {}'.format(result))
#     #     annotated_frame = draw_landmarks_on_image(output_image.numpy_view(), result)
#     #     out.write(cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))
#     pose_landmarks_list = result.pose_landmarks
#     # world_landmarks_list = result.pose_world_landmarks

#     # Normalize landmark coordinates
#     for i in range(len(pose_landmarks_list)):
#         for j in range(len(pose_landmarks_list[i])):
#             pose_landmarks_list[i][j].x *= CAMERA_FRAME_WIDTH
#             pose_landmarks_list[i][j].y *= CAMERA_FRAME_HEIGHT

#     # TODO: Calculate movement
#     global prev_landmarks
#     movement_results = calculate_movement(prev_landmarks, pose_landmarks_list)
#     prev_landmarks = pose_landmarks_list
#     print(movement_results)

#     # Calculate angles (for targets that moved)
#     pose_angles = []
#     for idx in range(len(pose_landmarks_list)):
#         pose_landmarks = pose_landmarks_list[idx]
#         center_x, center_y, center_z = calculate_center_of_landmarks(pose_landmarks)
#         print("Center X: {}".format(center_x))
#         print("Chest Area: {}".format(calculate_chest_area(pose_landmarks)))
#         pose_angles.append(-1 * int(calculate_angle_from_center(center_x, CAMERA_FRAME_WIDTH, CAMERA_FOV)) + 90)

#     # Send to Arduino (if ready)
#     arduinoString = ""
#     for i in range(len(pose_angles)):
#         arduinoString += str(pose_angles[i])
#         arduinoString += ","
#     if arduinoString == "":
#         arduinoString = "!"
#     print("Text sent to Arduino: {}".format(write_to_serial(arduinoString)))

def detect_victims(result, output_image, timestamp_ms):
    pose_landmarks_list = result.pose_landmarks

    if len(pose_landmarks_list) == 0:
        return

    # Normalize landmark coordinates
    for i in range(len(pose_landmarks_list)):
        for j in range(len(pose_landmarks_list[i])):
            pose_landmarks_list[i][j].x *= CAMERA_FRAME_WIDTH
            pose_landmarks_list[i][j].y *= CAMERA_FRAME_HEIGHT

    global prev_landmarks
    global prev_chest_areas
    global prev_wingspans
    # Return if number of detected people differs
    if len(pose_landmarks_list) != len(prev_landmarks):
        prev_landmarks = pose_landmarks_list
        prev_chest_areas.clear()
        prev_wingspans.clear()
        for idx in range(len(pose_landmarks_list)):
            prev_chest_areas.append(calculate_chest_area(pose_landmarks_list[idx]))
            prev_wingspans.append(calculate_wingspan(pose_landmarks_list[idx]))
        return

    movement_victim_detections[len(pose_landmarks_list)] += 1

    # Calculate movements
    target_idxs = []
    for idx in range(len(pose_landmarks_list)):
        chest_area = calculate_chest_area(pose_landmarks_list[idx])
        chest_area_change = calculate_relative_change(prev_chest_areas[idx], chest_area)
        vector_norm = calculate_vector_norm(prev_landmarks[idx], pose_landmarks_list[idx]) / chest_area
        wingspan = calculate_wingspan(pose_landmarks_list[idx])
        wingspan_change = calculate_relative_change(prev_wingspans[idx], wingspan)
        prev_nose_x = prev_landmarks[idx][0].x
        nose_x = pose_landmarks_list[idx][0].x
        center_x_change = (calculate_relative_change(prev_nose_x, nose_x) / chest_area) * 100000
        # print("Human {}:\n\tVector Norm: {}\n\tChest Area: {} ({} change from {})\n\tWingspan: {} ({} change from {})".format(idx, vector_norm, chest_area, chest_area_change, prev_chest_areas[idx], wingspan, wingspan_change, prev_wingspans[idx]))
        print("Human {}:\n\tCenter X: {} ({} change from {})\n\tChest Area: {} ({} change from {})\n\tWingspan: {} ({} change from {})".format(idx, nose_x, center_x_change, prev_nose_x, chest_area, chest_area_change, prev_chest_areas[idx], wingspan, wingspan_change, prev_wingspans[idx]))
        prev_chest_areas[idx] = chest_area
        prev_wingspans[idx] = wingspan

        if center_x_change > MOVEMENT_CENTER_X_CHANGE_THRESHOLD or chest_area_change > MOVEMENT_CHEST_AREA_CHANGE_THRESHOLD or wingspan_change > MOVEMENT_WINGSPAN_CHANGE_THRESHOLD:
            print("Human {} moved!".format(idx))
            movement_victim_violations[len(pose_landmarks_list)][idx] += 1
            target_idxs.append(idx)
    prev_landmarks = pose_landmarks_list # Save results for next check

    # This portion below will need to change with timing code, but will remain like this for testing
    # Calculate angles (for targets that moved)
    # pose_angles = []
    # for idx in range(len(pose_landmarks_list)):
    #     pose_landmarks = pose_landmarks_list[idx]
    #     center_x, center_y, center_z = calculate_center_of_landmarks(pose_landmarks)
    #     print("Center X: {}".format(center_x))
    #     print("Chest Area: {}".format(calculate_chest_area(pose_landmarks)))
    #     pose_angles.append(-1 * int(calculate_angle_from_center(center_x, CAMERA_FRAME_WIDTH, CAMERA_FOV)) + 90)

    pose_angles = []
    for idx in target_idxs:
        pose_landmarks = pose_landmarks_list[idx]
        center_x, center_y, center_z = calculate_center_of_landmarks(pose_landmarks)
        # print("Center X: {}".format(center_x))
        angle = -1 * int(calculate_angle_from_center(center_x, CAMERA_FRAME_WIDTH, CAMERA_FOV)) + 90
        movement_victim_angles[len(pose_landmarks_list)][idx] = angle
        pose_angles.append(angle)

    print("Saved angles: {}".format(movement_victim_angles[len(pose_landmarks_list)]))

    # Send to Arduino (if ready)
    #arduinoString = ""
    #for i in range(len(pose_angles)):
    #    arduinoString += str(pose_angles[i])
    #    arduinoString += ","
    #if arduinoString == "":
    #    arduinoString = "!"
    #print("Text sent to Arduino: {}".format(write_to_serial(arduinoString)))

def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        # Draw the pose landmarks.
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
        annotated_image,
        pose_landmarks_proto,
        solutions.pose.POSE_CONNECTIONS,
        solutions.drawing_styles.get_default_pose_landmarks_style())
    return annotated_image

def on_timer_elapsed():
    global current_stage
    current_stage = (current_stage + 1) % 4
    if current_stage == 0:
        print("Now in stage 0 (green light)")
        # Do tasks and schedule next on_timer_elapsed
        try:
            # Load the MP3 file
            pygame.mixer.music.load(GREEN_LIGHT_MP3_FILE)

            # Play the MP3 file
            pygame.mixer.music.play()

        except pygame.error as e:
            print("Error playing MP3 file:", e)
        #s.enter(5, 1, on_timer_elapsed)
        threading.Timer(5, on_timer_elapsed).start()
    elif current_stage == 1:
        # Same
        print("Now in stage 1 (red light; not detecting motion)")
        write_to_serial("#")
        # Make this delay random
        threading.Timer(0.5, on_timer_elapsed).start()
        #s.enter(0.5, 1, on_timer_elapsed)
    elif current_stage == 2:
        print("Now in stage 2 (red light; detecting motion)")
        threading.Timer(5, on_timer_elapsed).start()
        #s.enter(3, 1, on_timer_elapsed)
    elif current_stage == 3:
        print("Now in stage 3 (red light; shooting)")
        # Same
        best_index = 0
        global movement_victim_detections
        global movement_victim_violations
        global movement_victim_angles
        if movement_victim_detections[1] > movement_victim_detections[2] and movement_victim_detections[1] > movement_victim_detections[3] and movement_victim_detections[1] > movement_victim_detections[4]:
            best_index = 1
        elif movement_victim_detections[2] > movement_victim_detections[1] and movement_victim_detections[2] > movement_victim_detections[3] and movement_victim_detections[2] > movement_victim_detections[4]:
            best_index = 2
        elif movement_victim_detections[3] > movement_victim_detections[1] and movement_victim_detections[3] > movement_victim_detections[2] and movement_victim_detections[3] > movement_victim_detections[4]:
            best_index = 3
        elif movement_victim_detections[4] > movement_victim_detections[1] and movement_victim_detections[4] > movement_victim_detections[2] and movement_victim_detections[4] > movement_victim_detections[3]:
            best_index = 4
        if best_index > 0:
            for idx in range(len(movement_victim_violations[best_index])):
                if movement_victim_violations[best_index][idx] < MOVEMENT_COUNT_THRESHOLD:
                    movement_victim_angles[best_index][idx] = -1
        while arduino.in_waiting:
            arduino.readline()
        arduinoString = ""
        if best_index > 0:
            for i in range(len(movement_victim_angles[best_index])):
                if movement_victim_angles[best_index][i] != -1:
                    arduinoString += str(movement_victim_angles[best_index][i])
                    arduinoString += ","
        if arduinoString == "":
            arduinoString = "!"
        print("Text sent to Arduino: {}".format(write_to_serial(arduinoString)))

        prev_landmarks.clear()
        prev_chest_areas.clear()
        prev_wingspans.clear()
        movement_victim_angles = {1: [-1], 2: [-1, -1], 3: [-1, -1, -1], 4: [-1, -1, -1, -1]}
        movement_victim_violations = {1: [0], 2: [0, 0], 3: [0, 0, 0], 4: [0, 0, 0, 0]}
        movement_victim_detections = {1: 0, 2: 0, 3: 0, 4: 0}

        print("Returned from Arduino: {}".format(arduino.readline()))
        threading.Timer(0.1, on_timer_elapsed).start()
        #s.enter(5, 1, on_timer_elapsed)
    #current_stage = (current_stage + 1) % 4
    # Run scheduler
    #s.run()

# Create camera
cap = cv2.VideoCapture(SELECTED_CAMERA)

# Create serial connection
# arduino = serial.Serial(port='/dev/ttyACM0', baudrate = 115200, timeout=0.1)

# Create video file output (for debug only)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 5.0, (CAMERA_FRAME_WIDTH, CAMERA_FRAME_HEIGHT))

pygame.init()
pygame.mixer.init()

# Create landmarker
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    num_poses=MAX_PLAYERS,
    min_pose_detection_confidence=MIN_POSE_DETECTION_CONFIDENCE,
    min_pose_presence_confidence=MIN_POSE_PRESENCE_CONFIDENCE,
    min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=on_result_ready)

prev_landmarks = []
prev_chest_areas = []
prev_wingspans = []

movement_victim_angles = {1: [-1], 2: [-1, -1], 3: [-1, -1, -1], 4: [-1, -1, -1, -1]}
movement_victim_violations = {1: [0], 2: [0, 0], 3: [0, 0, 0], 4: [0, 0, 0, 0]}
movement_victim_detections = {1: 0, 2: 0, 3: 0, 4: 0}

current_stage = -1
#s = sched.scheduler(time.monotonic, time.sleep)
on_timer_elapsed()

with PoseLandmarker.create_from_options(options) as landmarker:
    prev_time = time.time_ns()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Control frame rate
        curr_time = time.time_ns()
        elapsed_time = curr_time - prev_time
        if elapsed_time < 1.0 / TARGET_FPS:
            continue
        prev_time = curr_time

        # Detect humans in camera feed
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        landmarker.detect_async(mp_image, curr_time // 1_000_000)

        # out.write(frame) # This will write to video file, but file ends up corrupted

        # Quit if "Q" key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
out.release()
cv2.destroyAllWindows()
pygame.quit()

