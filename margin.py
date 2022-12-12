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

# filename = "./videos/JJ1.mp4"
# filenames = ["./videos/JJ1.mp4", "./videos/StephCurry.mp4"]
path = './videos'
filenames = os.listdir(path)



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
for filename in filenames:
    filename = './videos/' + filename
    # clear list 
    images = []
    left_elbow_angles.clear()
    right_elbow_angles.clear()
    right_shoulder_angles.clear()
    right_wrist_angles.clear()
    right_hip_angles.clear()
    right_knee_angles.clear()
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
    # print(samples)
    tracked_metrics = {
        'right_elbow_angles': right_elbow_angles,
        'right_shoulder_angles': right_shoulder_angles,
        'right_wrist_angles': right_wrist_angles,
        'right_hip_angles': right_hip_angles,
        'right_knee_angles': right_knee_angles,
    }
    sampled_metrics = {
        'sampled_right_elbow_angles': [],
        'sampled_right_shoulder_angles': [],
        'sampled_right_wrist_angles': [],
        'sampled_right_hip_angles': [],
        'sampled_right_knee_angles': [],
    }
    metrics_y_values = {
        'sampled_right_elbow_angles_y': [],
        'sampled_right_shoulder_angles_y': [],
        'sampled_right_wrist_angles_y': [],
        'sampled_right_hip_angles_y': [],
        'sampled_right_knee_angles_y': [],
    }
    sampled_left_elbow_angles = []
    for metric in tracked_metrics:
        y_values = []
        for i, angle in enumerate(tracked_metrics[metric]):
            if i not in samples:
                # print(i)
                sampled_metrics['sampled_' + metric].append(angle)
        sampled_times = np.arange(60) # mostly 30 fps and most given videos are of 2secs
        # print(sampled_left_elbow_angles)
        sampled_metrics['sampled_' + metric] = np.array(sampled_metrics['sampled_' + metric])
        print(f'After sampling: {len(sampled_times)}')
        plt.plot(sampled_times, sampled_metrics['sampled_' + metric], color='b', label = 'sampled_' + metric)
        plt.savefig(f"./output-images/{filename[9:-4]}'s sampled-unsmooth-angle-vs-time.jpg")
    ###### sampling #####
    
    ##### smooth a curve #####
    # cubic_interpolation_model = interp1d(times, left_elbow_angles, kind = "cubic")
        cubic_interpolation_model = interp1d(sampled_times, sampled_metrics['sampled_' + metric])

        # times = np.array(times)
        sampled_times = np.array(sampled_times)
        # X_ = np.linspace(times.min(), times.max(), 50)
        # Y_ = cubic_interpolation_model(times)
        # Y_ = cubic_interpolation_model(sampled_times)
    
    # apply smoothing filter   
    # Y_ = savgol_filter(Y_, len(times), 50)
        Y_ = sampled_metrics['sampled_' + metric]
        Y_ = savgol_filter(Y_, window_length=30, polyorder=7)
    
    # for margin graph
        y = -Y_
    # add list y of each video 
        print(f'{filename} video frames: {len(y)}')
        metrics_y_values['sampled_' + metric + '_y'] += list(y)
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
        plt.savefig(f"./output-images/{filename[9:-4]}'s sampled smoothed " + metric[6:-7] + "angle-vs-time.jpg")
        plt.show(block=False)
        plt.close()
    

    


# visualize margin   
for metric in metrics_y_values:
    sns.set()
    y_values = np.negative(metrics_y_values[metric])
    print(f'y_values length: {len(y_values)}')
    y_values = np.array(y_values).reshape(len(filenames), -1)
    y_means = np.mean(y_values, axis=0)
    y_std = np.std(y_values, axis=0)
    y_diff = y_values[0] - y_values[1]

    # save data
    np.savetxt('./data/' + metric[8:-9], y_values)
    # with open(r'./data/' + metric[6:-7], 'w') as fp:
    #     fp.writelines(y_values)


    print("y_std.shape: {}".format(y_std.shape))
    print(f'y_values.shape: {y_means.shape}')
    print("y_values: {}".format(y_values))
    print("y_diff: {}".format(y_diff))


    x_axis = np.arange(len(y_means))   
    plt.plot(x_axis, y_means, 'b-', label='y_value')
    plt.fill_between(x_axis, y_means-y_std, y_means+y_std, color = 'b', alpha=0.2)
    plt.ylim([0, 180])

    plt.legend(title='margin')
    plt.ioff()
    # plt.savefig
    plt.show(block=False)
    plt.savefig("./output-images/" + metric[8:-9] + "-margin.jpg")
    plt.close()
