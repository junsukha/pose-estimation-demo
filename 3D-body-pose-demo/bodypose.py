import cv2 as cv
import mediapipe as mp
import numpy as np
import sys

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

frame_shape = [720, 1280]

pose_keypoints =  [16, 14, 12, 11, 13, 15, 24, 23, 25, 26, 27, 28]

def run_mp(input_stream1, input_stream2, P0, P1):
    cap0 = cv.VideoCapture(input_stream1)
    cap1 = cv.VideoCapture(input_stream2)
    caps = [cap0, cap1]
    
    for cap in caps:
        cap.set(3, frame_shape[1]) # width
        cap.set(4, frame_shape[0]) # height

    pose0 = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    pose1 = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    
    kpts_cam0 = []
    kpts_cam1 = []
    kpts = [kpts_cam0, kpts_cam1]
    kpts_3d = []
    
    while True:
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()
        
        if not ret0 or not ret1: break
        
        if frame0.shape[1] != 720: # if width is not 720
            frame0 = frame0[:,frame_shape[1]//2 - frame_shape[0]//2:frame_shape[1]//2 + frame_shape[0]//2]
            frame1 = frame1[:,frame_shape[1]//2 - frame_shape[0]//2:frame_shape[1]//2 + frame_shape[0]//2]
            
        frame0 = cv.cvtColor(frame0, cv.COLOR_BGR2RGB)
        frame1 = cv.cvtColor(frame1, cv.COLOR_BGR2RGB)
        
        frame0.flags.writeable = False
        frame1.flags.writeable = False
        
        results0 = pose0.process(frame0)
        results1 = pose1.process(frame1)
        results = [results0, results1]
        
        frame0.flags.writeable = True
        frame1.flags.writeable = True
        frame0 = cv.cvtColor(frame0, cv.COLOR_RGB2BGR)
        frame1 = cv.cvtColor(frame1, cv.COLOR_RGB2BGR)
        
        frame0_keypoints = []
        frame1_keypoints = []
        frames_keypoints = [frame0_keypoints, frame1_keypoints]
        
        for i, result in enumerate(results):  
            
            if result.pose_landmarks:
            
                for i, landmark in enumerate(result.pose_landmarks.landmark):
                    if i not in pose_keypoints: continue
                    px1_x = landmark.x * frame0.shape[1] # width
                    px1_y = landmark.y * frame0.shape[0]
                    px1_x = int(round(px1_x))
                    px1_y = int(round(px1_y))
                    cv.circle(frame0, (px1_x, px1_y), 3, (0,0,255), -1)
                    kpts = [px1_x, px1_y]
                    frames_keypoints[i].append(kpts)
            else:
                frames_keypoints[i] = [[-1,-1] * len(pose_keypoints)]
            
        
             # keep keypoints of each frame. kpts has all frame info
             # while frame_keypoints are reset for every frame
            kpts[i].append(frames_keypoints[i])
            
        
        # calculate 3D position
        frame_p3ds = []
        for uv1, uv2 in zip(frames_keypoints[0], frames_keypoints[1]):
            if uv1[0] == -1 or uv2[0] == -1:
                _p3d = [-1,-1,-1]
            else:
                # DLT calculates 3d position of keypoint
                # uv1 consists of x,y,z,visibility
                _p3d = DLT(P0, P1, uv1, uv2 )

            frame_p3ds.append(_p3d)
            
            
        frame_p3ds = np.array(frame_p3ds).reshape((12,3))
        
        cv.imshow('cam1', frame1)
        cv.imshow('cam2'. frame2)
        
        k = cv.waitkey(1)
        if k & 0xFF == 27: break #27 is ESC
        
    cv.destroyAllWindows()
    for cap in caps:
        cap.release()
        
    return np.array(kpts_cam0), np.array(kpts_cam1), np.array(kpts_3d)