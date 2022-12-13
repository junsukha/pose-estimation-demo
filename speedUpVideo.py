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

import seaborn as sns
import os
# filename = "./videos/StephCurry.mp4"
filenames = ["./videos/StephCurry.mp4"]


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
y_values =[]

metrics_y_values = {
    'right_elbow_angles_y': [],
    'right_shoulder_angles_y': [],
    'right_wrist_angles_y': [],
    'right_hip_angles_y': [],
    'right_knee_angles_y': [],
}

for filename in filenames:
    # clear list 
    images = []
    left_elbow_angles.clear()
    times = []
    time = 0
    time_for_seaborn= []    

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose: #.Pose access pose estimation model, #min_tracking_confidence tracks state
        cap = cv2.VideoCapture(filename)
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
                
                # Visualize left_elbow position on each frame
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

        # while loop is ended        
        cap.release()
        cv2.destroyAllWindows()    
    
    # plot angles vs times graph
    print(f'Before sampling: {len(times)}')
    plt.plot(times, left_elbow_angles, color='b', label = 'left_elbow')
    plt.savefig(f"./output-images/{filename[9:-4]}'s unsmooth-angle-vs-time.jpg")
    plt.close()
    
    ###### sampling #####
    # samples are indices that will be filtered out
    samples = np.random.choice(np.arange(len(times)), size=len(times) - 60, replace=False) # want to fix the number of frames at 60
    print(samples)
    sampled_left_elbow_angles = []
    for i, angle in enumerate(left_elbow_angles):
        if i not in samples:
            # print(i)
            sampled_left_elbow_angles.append(angle)            
    sampled_times = np.arange(60) # mostly 30 fps and most given videos are of 2secs
    print(sampled_left_elbow_angles)
    sampled_left_elbow_angles = np.array(sampled_left_elbow_angles)
    print(f'After sampling: {len(sampled_times)}')
    plt.plot(sampled_times, sampled_left_elbow_angles, color='b', label = 'sampled_left_elbow')
    plt.savefig(f"./output-images/{filename[9:-4]}'s sampled-unsmooth-angle-vs-time.jpg")
    
    ##### smooth a curve #####
    # cubic_interpolation_model = interp1d(times, left_elbow_angles, kind = "cubic")
    cubic_interpolation_model = interp1d(sampled_times, sampled_left_elbow_angles)

    # times = np.array(times)
    sampled_times = np.array(sampled_times)
    # X_ = np.linspace(times.min(), times.max(), 50)
    # Y_ = cubic_interpolation_model(times)
    # Y_ = cubic_interpolation_model(sampled_times)
    
    # apply smoothing filter   
    # Y_ = savgol_filter(Y_, len(times), 50)
    Y_ = sampled_left_elbow_angles
    Y_ = savgol_filter(Y_, window_length=30, polyorder=7)
    
    # for margin graph
    y = -Y_
    # add list y of each video 
    print(f'{filename} video frames: {len(y)}')
    y_values += list(y)
    ##### smooth a curve #####
    
    ## plot each angle
    # plot left angles
    # plt.plot(times,Y_, color='r', label='smooth')
    plt.plot(sampled_times,Y_, color='r', label='smooth')
    # plt.plot(times, left_shoulder_angles, color='r', label='left_shoulder')
    # plt.plot(times, left_wrist_angles, color='g', label = 'left wrist')
    # plt.plot(times, left_hip_angles, color='y', label = 'left hip')
    # plt.plot(times, left_knee_angles, color='m', label = 'left knee')

    # plot right angles
    # plt.plot(times, right_elbow_angles, color='#87CEEB', label = 'right_elbow')
    # plt.plot(times, right_shoulder_angles, color='#FFC0CB', label='right_shoulder')
    # plt.plot(times, right_wrist_angles, color='#90EE90', label = 'right wrist')
    # plt.plot(times, right_hip_angles, color='#FFF01F', label = 'right hip')
    # plt.plot(times, right_knee_angles, color='#A020F0', label = 'right knee')

    plt.xlabel('time')
    plt.ylabel('angle')
    plt.legend(title=f"{filename[9:-4]}'s angle vs time")
    plt.savefig(f"./output-images/{filename[9:-4]}'s sampled smoothed angle-vs-time.jpg")
    plt.show(block=False)
    plt.close()


    ################ show video and graph ######################
    # filename = './Steph Curry.mp4'
    cap = cv2.VideoCapture(filename) # need this because cap is released before

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
    plt.show(block=False)
    
    

    #Setup a dummy path
    # x = np.linspace(0,width,frames)
    x = times
    # y = x/2. + 100*np.sin(2.*np.pi*x/1200)
    # y = Y_
    y = -Y_


    # y is array. convert it to list    
    y_values += list(y)
        
    # need this to plot graph as the original. otherwise upside down
    left_elbow_angles = (-np.array(left_elbow_angles)).tolist()
    y = savgol_filter(left_elbow_angles, window_length=30, polyorder=7)

    print(len(images))
    print(len(times))
    print(len(y))

    # for i in enumerate(times): 
    for i, image in enumerate(images):
        fig.clear()
        flag, frame = cap.read()

        
        plt.imshow(image)
        # this line is to match graph width to image width
        times = times/np.max(times)*width
        # plt.plot(x,y,'k-', lw=2)
        plt.plot(times, y, 'k-', lw=2) # smooth one
        plt.plot(times, left_elbow_angles, 'g-', lw=2) # original one
        plt.plot(times[i-1],y[i-1],'or') # red dot for each angle of smooth one
        plt.savefig("./graph-video/images/graph-image" + str(i) + ".jpg")

        # plt.pause(1e-7)
        
        
        
        # # Make detection
        # results = pose.process(image) # image here is RGB

        # # Recolor back to BGR
        # image.flags.writeable = True
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break        
    ################ show video and graph ######################        

# compress images to a video
images_folder = "./graph-video/images"
video_name = './graph-video/video/video.avi'

images = [img for img in os.listdir(images_folder)]
images.sort(key=lambda f: int(f[11:-4]))

frame = cv2.imread(os.path.join(images_folder, images[0]))
height, width, layers = frame.shape
video = cv2.VideoWriter(video_name, 0, 10, (width,height))

for image in images:
    print(image)
    video.write(cv2.imread(os.path.join(images_folder, image)))

cv2.destroyAllWindows()
video.release()


# visualize margin   
sns.set()

print(f'y_values length: {len(y_values)}')
y_values = np.array(y_values).reshape(2, -1)
y_values = np.mean(y_values, axis=0)

print(f'y_values.shape: {y_values.shape}')

x_axis = np.arange(len(y_values))   
plt.plot(x_axis, y_values, 'b-', label='y_value')
plt.fill_between(x_axis, y_values-10, y_values+10, color = 'b', alpha=0.2)

plt.legend(title='margin')
plt.ioff()
# plt.savefig
plt.show(block=False)
plt.savefig("./output-images/margin.jpg")




# def show_margin(y_values):
#     sns.set()
    
#     print(f'y_values length: {len(y_values)}')
#     y_values = np.array(y_values).reshape(2, -1)
#     y_values = np.mean(y_values, axis=0)

#     print(f'y_values.shape: {y_values.shape}')

#     x_axis = np.arange(len(y_values))   
#     plt.plot(x_axis, y_values, 'b-', label='y_value')
#     plt.fill_between(x_axis, y_values-10, y_values+10, color = 'b', alpha=0.2)

#     plt.legend(title='test')
#     plt.show()


# show_margin(y_values)  