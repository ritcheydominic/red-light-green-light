# Imports
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import cv2
import time
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
CAMERA_FOV = 75 # Should probably be 90 (or 120 for Raspberry Pi Camera Module)
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
# Writes "x" to serial port connected to Arduino
def write_to_serial(x):
    arduino.write(bytes(x, 'utf-8'))
    return x

# Gives absolute value of relative change between "initial" and "final"
def calculate_relative_change(initial, final):
    # print("Initial: {}\tFinal: {}\tResult: {}".format(initial, final, (final - initial) / initial))
    return np.absolute((final - initial) / initial)

# Gives average x (left-right), y (up-down), and z (depth) values for set of landmarks
def calculate_center_of_landmarks(landmarks):
    if len(landmarks) == 0:
        return None
    
    mean_x = sum([landmark.x for landmark in landmarks]) / len(landmarks)
    mean_y = sum([landmark.y for landmark in landmarks]) / len(landmarks)
    mean_z = sum([landmark.z for landmark in landmarks]) / len(landmarks)

    return (int(mean_x), int(mean_y), mean_z)

# Finds angle from center line to set of landmarks based on frame width and FOV
def calculate_angle_from_center(center, frame_width, fov=75):
    # Calculate the horizontal offset from the center of the frame
    offset = center - frame_width / 2
    # Calculate angle
    angle = (offset / frame_width) * fov
    return angle

# Determines area of chest (from rectangle formed by landmarks on shoulders and hips) in units of pixels^2
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

# Determines length between left and right hands in units of pixels
def calculate_wingspan(pose_landmarks):
    left_wrist_x = pose_landmarks[15].x
    left_wrist_y = pose_landmarks[15].y
    right_wrist_x = pose_landmarks[16].x
    right_wrist_y = pose_landmarks[16].y
    return np.sqrt(((left_wrist_x - right_wrist_x) ** 2) + ((left_wrist_y - right_wrist_y) ** 2))

# Decides between detecting winners and victims depending on game state
# Winner detection not implemented due to time constraints
def on_result_ready(result, output_image, timestamp_ms):
    # Call which detect function based on state
    global current_stage
    if current_stage == 0:
        detect_winners(result, output_image, timestamp_ms)
    elif current_stage == 2:
        detect_victims(result, output_image, timestamp_ms)

# Detect if player is too close to camera during green light, marking them as winner
# Not implemented in game due to time constraints
def detect_winners(result, output_image, timestamp_ms):
    return result

# Detect if player moved during red light, marking them to be shot with ball
def detect_victims(result, output_image, timestamp_ms):
    pose_landmarks_list = result.pose_landmarks

    # If no one was detected in image, return
    if len(pose_landmarks_list) == 0:
        return

    # Normalize landmark coordinates
    for i in range(len(pose_landmarks_list)):
        for j in range(len(pose_landmarks_list[i])):
            pose_landmarks_list[i][j].x *= CAMERA_FRAME_WIDTH
            pose_landmarks_list[i][j].y *= CAMERA_FRAME_HEIGHT

    # Stop movement detection, but save results if number of detected people differs from previous frame
    global prev_landmarks
    global prev_chest_areas
    global prev_wingspans
    if len(pose_landmarks_list) != len(prev_landmarks):
        prev_landmarks = pose_landmarks_list
        prev_chest_areas.clear()
        prev_wingspans.clear()
        for idx in range(len(pose_landmarks_list)):
            prev_chest_areas.append(calculate_chest_area(pose_landmarks_list[idx]))
            prev_wingspans.append(calculate_wingspan(pose_landmarks_list[idx]))
        return

    # Increment detections for people count detected
    movement_victim_detections[len(pose_landmarks_list)] += 1

    # Calculate movements
    target_idxs = []
    for idx in range(len(pose_landmarks_list)):
        # Change in chest area is used for detecting z-wise (depth) movement in image
        chest_area = calculate_chest_area(pose_landmarks_list[idx])
        chest_area_change = calculate_relative_change(prev_chest_areas[idx], chest_area)

        # Change in wingspan is used for detecting movement of players' arms
        wingspan = calculate_wingspan(pose_landmarks_list[idx])
        wingspan_change = calculate_relative_change(prev_wingspans[idx], wingspan)

        # Change in center x coordinate is used for detecting x-wise (left-right) movement in image
        # X coordinate of nose is used as center x coordinate
        prev_nose_x = prev_landmarks[idx][0].x
        nose_x = pose_landmarks_list[idx][0].x
        center_x_change = (calculate_relative_change(prev_nose_x, nose_x) / chest_area) * 100000
        print("Human {}:\n\tCenter X: {} ({} change from {})\n\tChest Area: {} ({} change from {})\n\tWingspan: {} ({} change from {})".format(idx, nose_x, center_x_change, prev_nose_x, chest_area, chest_area_change, prev_chest_areas[idx], wingspan, wingspan_change, prev_wingspans[idx]))
        
        # Save results for next check
        prev_chest_areas[idx] = chest_area
        prev_wingspans[idx] = wingspan

        # If any movement value is greater than assocaited threshold, mark player as victim
        if center_x_change > MOVEMENT_CENTER_X_CHANGE_THRESHOLD or chest_area_change > MOVEMENT_CHEST_AREA_CHANGE_THRESHOLD or wingspan_change > MOVEMENT_WINGSPAN_CHANGE_THRESHOLD:
            print("Human {} moved!".format(idx))
            movement_victim_violations[len(pose_landmarks_list)][idx] += 1
            target_idxs.append(idx)
    prev_landmarks = pose_landmarks_list # Save results for next check

    # Determine and save angles of players marked as victims
    for idx in target_idxs:
        pose_landmarks = pose_landmarks_list[idx]
        center_x, center_y, center_z = calculate_center_of_landmarks(pose_landmarks)
        angle = -1 * int(calculate_angle_from_center(center_x, CAMERA_FRAME_WIDTH, CAMERA_FOV)) + 90

        # Save player's angle in 2D array indexed by number of players and player index (to be used later with determining confidence in movement calculations and whether players will actually be shot with balls)
        movement_victim_angles[len(pose_landmarks_list)][idx] = angle

    print("Saved angles: {}".format(movement_victim_angles[len(pose_landmarks_list)]))

# Draws landmark information on frame
# Not used since saving video doesn't work in this script for some reason
def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected poses to visualize
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        # Draw the pose landmarks
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

# Transitions game between finite states
#
# Game States
#   0: Green light (reading frames from camera and detecting winners), lasts 5 seconds (timed in sync with MP3 file)
#       Because winner detection wasn't implemented in this version of the game, this state just acts as green light where players can move about freely
#   1: Red light (head rotates, frames read from camera are discarded), lasts 0.5 seconds
#   2: Red light (reading frames from camera and detecting motion to determine victims), lasts 5 seconds
#   3: Red light (shooting any victims that moved too much with balls), lasts until Arduino returns message indicating all victims, if any, have been shot with balls
def on_timer_elapsed():
    global current_stage
    current_stage = (current_stage + 1) % 4
    if current_stage == 0:
        print("Now in stage 0 (green light)")
        try:
            # Load the MP3 file
            pygame.mixer.music.load(GREEN_LIGHT_MP3_FILE)

            # Play the MP3 file
            pygame.mixer.music.play()

        except pygame.error as e:
            print("Error playing MP3 file:", e)
        threading.Timer(5, on_timer_elapsed).start()
    elif current_stage == 1:
        print("Now in stage 1 (red light; not detecting motion)")
        
        # Send message to Arduino to turn head
        write_to_serial("#")

        threading.Timer(0.5, on_timer_elapsed).start()
    elif current_stage == 2:
        print("Now in stage 2 (red light; detecting motion)")
        threading.Timer(5, on_timer_elapsed).start()
    elif current_stage == 3:
        print("Now in stage 3 (red light; shooting)")

        # Out of all frames collected, determine where highest confidence of overall measurement is (based on numebr of detections with specific player count in them)
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

        # If we are confident 1+ players are in image, then check confidence in each player's movement for set of measurements
        if best_index > 0:
            for idx in range(len(movement_victim_violations[best_index])):
                # Discard angle and don't shoot player with balls if we aren't confident they moved
                if movement_victim_violations[best_index][idx] < MOVEMENT_COUNT_THRESHOLD:
                    movement_victim_angles[best_index][idx] = -1
        
        # Clear serial buffer of any pending messages to be read
        while arduino.in_waiting:
            arduino.readline()
        
        # Construct comma-separated message to be sent to Arduino with players to be shot with balls
        arduinoString = ""
        if best_index > 0:
            for i in range(len(movement_victim_angles[best_index])):
                if movement_victim_angles[best_index][i] != -1:
                    arduinoString += str(movement_victim_angles[best_index][i])
                    arduinoString += ","
        # If no players are to be shot with balls, send an "!"
        if arduinoString == "":
            arduinoString = "!"
        print("Text sent to Arduino: {}".format(write_to_serial(arduinoString)))

        # Clear data from this round of game
        prev_landmarks.clear()
        prev_chest_areas.clear()
        prev_wingspans.clear()
        movement_victim_angles = {1: [-1], 2: [-1, -1], 3: [-1, -1, -1], 4: [-1, -1, -1, -1]}
        movement_victim_violations = {1: [0], 2: [0, 0], 3: [0, 0, 0], 4: [0, 0, 0, 0]}
        movement_victim_detections = {1: 0, 2: 0, 3: 0, 4: 0}

        print("Returned from Arduino: {}".format(arduino.readline()))
        threading.Timer(0.1, on_timer_elapsed).start()

# Create camera
cap = cv2.VideoCapture(SELECTED_CAMERA)

# Create video file output (for debug only)
# This doesn't work for some reason
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 5.0, (CAMERA_FRAME_WIDTH, CAMERA_FRAME_HEIGHT))

# Create pygame instance for playing audio
pygame.init()
pygame.mixer.init()

# Set landmarker options 
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    num_poses=MAX_PLAYERS,
    min_pose_detection_confidence=MIN_POSE_DETECTION_CONFIDENCE,
    min_pose_presence_confidence=MIN_POSE_PRESENCE_CONFIDENCE,
    min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=on_result_ready)

# Create arrays for storing landmark information from previous frames while detecting movement
prev_landmarks = []
prev_chest_areas = []
prev_wingspans = []

# Create arrays for storing confidence and angle information while detecting movement
movement_victim_angles = {1: [-1], 2: [-1, -1], 3: [-1, -1, -1], 4: [-1, -1, -1, -1]}
movement_victim_violations = {1: [0], 2: [0, 0], 3: [0, 0, 0], 4: [0, 0, 0, 0]}
movement_victim_detections = {1: 0, 2: 0, 3: 0, 4: 0}

# Initialize game state
current_stage = -1
on_timer_elapsed()

# Create landmarker model
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

        # This will write to video file, but file ends up corrupted
        # out.write(frame)

        # Quit if "Q" key is pressed, but also doesn't work
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

# Clean up stuff after quitting game
cap.release()
out.release()
cv2.destroyAllWindows()
pygame.quit()

