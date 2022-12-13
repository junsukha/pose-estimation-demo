import os
import cv2
##### compress images to a video #####
right_elbow_angles = []
right_shoulder_angles = []
right_wrist_angles = []
right_hip_angles = []
right_knee_angles = []

tracked_metrics = {
        'right_elbow_angles': right_elbow_angles,
        'right_shoulder_angles': right_shoulder_angles,
        'right_wrist_angles': right_wrist_angles,
        'right_hip_angles': right_hip_angles,
        'right_knee_angles': right_knee_angles,
}

for metric in tracked_metrics:
    images_folder = "./graph-video/images/" + metric
    video_name = './graph-video/video/'+ metric + '_video.avi'

    image_names = [img for img in os.listdir(images_folder)]
    image_names.sort(key=lambda f: int(f[11:-4]))

    frame = cv2.imread(os.path.join(images_folder, image_names[0]))
    height, width, layers = frame.shape
    video = cv2.VideoWriter(video_name, 0, 10, (width,height))

    for image in image_names:
        print(image)
        video.write(cv2.imread(os.path.join(images_folder, image)))
        # video.write(cv2.imread(image_names))

    cv2.destroyAllWindows()
    video.release()  
##### compress images to a video #####