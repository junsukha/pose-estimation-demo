import calculateangle
import cv2 
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils # when visualing out poses
mp_pose = mp.solutions.pose

# for smoothing function
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

#Copied from the above cell

import cv2
import matplotlib.pyplot as plt
import numpy as np

filename = "./videos/JJ1.mp4"
cap = cv2.VideoCapture(filename)
# cap = cv2.VideoCapture("Steph Curry.mp4")
# Curl counter variables
# counter = 0
# stage = None

times = []
time = 0
left_elbow_angles = []
left_shoulder_angles = []
left_wrist_angles = []
left_hip_angles = []
left_knee_angles = []
right_elbow_angles = []
right_shoulder_angles = []
right_wrist_angles = []
right_hip_angles = []
right_knee_angles = []
## setup mediapipe instance
images = []
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose: #.Pose access pose estimation model, #min_tracking_confidence tracks state
    while cap.isOpened():
        
        ret, frame = cap.read() # frame is image from camera
        
        if not ret:
            cap.release()
            break
        
        frame_width = int(cap.get(3))
        
        # Recolor image
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False # save memory
        images.append(image)
        
        # Make detection
        results = pose.process(image) # image here is RGB
        
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark # hold landamrks. including x,y,z. Use this for calculating angles
            # print(landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX])
            # Filter out landmarks with low visibility
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

            # shoulder angle
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]

            # wrist angle
            left_index = [landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].x, landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].y]
            right_index = [landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].y]

            # hip angle
            left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]

            # knee angle
            left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
            
            # left_ = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]

            # hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            # shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            # elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]   
            
            left_elbow_angle = calculateangle.calculate_angle(left_shoulder, left_elbow, left_wrist)
            left_shoulder_angle = calculateangle.calculate_angle(left_hip, left_shoulder, left_elbow)
            left_wrist_angle = calculateangle.calculate_angle(left_elbow, left_wrist, left_index)
            left_hip_angle = calculateangle.calculate_angle(left_knee, left_hip, left_shoulder)
            left_knee_angle = calculateangle.calculate_angle(left_ankle, left_knee, left_hip)

            right_elbow_angle = calculateangle.calculate_angle(right_shoulder, right_elbow, right_wrist)
            right_shoulder_angle = calculateangle.calculate_angle(right_hip, right_shoulder, right_elbow)
            right_wrist_angle = calculateangle.calculate_angle(right_elbow, right_wrist, right_index)
            right_hip_angle = calculateangle.calculate_angle(right_knee, right_hip, right_shoulder)
            right_knee_angle = calculateangle.calculate_angle(right_ankle, right_knee, right_hip)
            
            # if time % 5 == 0:
            left_elbow_angles.append(left_elbow_angle)
            left_shoulder_angles.append(left_shoulder_angle)
            left_wrist_angles.append(left_wrist_angle)
            left_hip_angles.append(left_hip_angle)
            left_knee_angles.append(left_knee_angle)

            right_elbow_angles.append(right_elbow_angle)
            right_shoulder_angles.append(right_shoulder_angle)
            right_wrist_angles.append(right_wrist_angle)
            right_hip_angles.append(right_hip_angle)
            right_knee_angles.append(right_knee_angle)

            times.append(time)
            time+=1
            
            # Visualize
            cv2.putText(image, 
                        str(left_elbow),
                        tuple(np.multiply(left_elbow, [640, 480]).astype(int)), # controal [640, 480] to window size
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        except:
           pass
        
        ## draw only relevant points. Do not draw points on a face
        # points_on_face = [1,2,3,4,5,6,7,8,9,10]
        # for i, landmark in enumerate(results.pose_landmarks.landmark):
        #     if (landmark in points_on_face):             
        
        # filtered = [landmark for i, landmark in enumerate(results.pose_landmarks.landmark) if i not in points_on_face]
        # results.pose_landmarks.landmark = filtered
        # results.pose_landmarks
       
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)) # image here is BGR
        
        cv2.imshow('Mediapipe Feed', image)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    
    cap.release()
    cv2.destroyAllWindows()
    
plt.plot(times, left_elbow_angles, color='b', label = 'left_elbow')

print(len(left_elbow_angles))


## smooth a curve
cubic_interpolation_model = interp1d(times, left_elbow_angles, kind = "cubic")
times = np.array(times)
# X_ = np.linspace(times.min(), times.max(), 50)
Y_ = cubic_interpolation_model(times)

Y_ = savgol_filter(Y_, len(times), 50)
plt.plot(times,Y_, color='r', label='smooth')
# plt.plot(times, left_shoulder_angles, color='r', label='left_shoulder')
# plt.plot(times, left_wrist_angles, color='g', label = 'left wrist')
# plt.plot(times, left_hip_angles, color='y', label = 'left hip')
# plt.plot(times, left_knee_angles, color='m', label = 'left knee')

# plt.plot(times, right_elbow_angles, color='#87CEEB', label = 'right_elbow')
# plt.plot(times, right_shoulder_angles, color='#FFC0CB', label='right_shoulder')
# plt.plot(times, right_wrist_angles, color='#90EE90', label = 'right wrist')
# plt.plot(times, right_hip_angles, color='#FFF01F', label = 'right hip')
# plt.plot(times, right_knee_angles, color='#A020F0', label = 'right knee')
plt.legend()

plt.xlabel('time')
plt.ylabel('angle')
plt.show()

################ show video and graph ######################
cap = cv2.VideoCapture(filename)

try:
    frames = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    width  = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
except AttributeError:
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fig, ax = plt.subplots(1,1)
plt.ion()
plt.show()

#Setup a dummy path
# x = np.linspace(0,width,frames)
x = times
# y = x/2. + 100*np.sin(2.*np.pi*x/1200)
# y = Y_
y = -Y_
# need this to plot graph as the original. otherwise upside down
left_elbow_angles = (-np.array(left_elbow_angles)).tolist()

print(len(images))
print(len(times))
print(len(y))

# for i in enumerate(times): 
for i, image in enumerate(images):
    fig.clf()
    flag, frame = cap.read()


    plt.imshow(image)
    times = times/np.max(times)*width
    # plt.plot(x,y,'k-', lw=2)
    plt.plot(times, y, 'k-', lw=2)
    plt.plot(times, left_elbow_angles, 'g-', lw=2)
    plt.plot(times[i],y[i],'or')
    plt.pause(0.001)
    
    
    
    # # Make detection
    # results = pose.process(image) # image here is RGB

    # # Recolor back to BGR
    # image.flags.writeable = True
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break        
        
