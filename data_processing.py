import glob
import cv2
import mediapipe as mp
import numpy as np


def set_fps(fps, src_path, trg_path):
    videos = glob.glob(src_path + r'/*.mp4')

    for i in videos:

        cap = cv2.VideoCapture(i)

        file_name = i.split("\\")[-1]

        fps_val = int(cap.get(cv2.CAP_PROP_FPS))
        print(f"fps of the video {file_name} : {fps_val}")

        assert cap.isOpened()

        fps_in = cap.get(cv2.CAP_PROP_FPS)
        fps_out = fps  # value of fps is passed here

        index_in = -1
        index_out = -1

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(f"{trg_path}/{file_name}", fourcc, fps_out, (width, height))

        while True:
            success = cap.grab()
            if not success:
                break
            index_in += 1

            out_due = int(index_in / fps_in * fps_out)
            while out_due > index_out:
                success, frame = cap.retrieve()
                if not success:
                    break
                index_out += 1

                # Process the frame (you can modify or analyze it here)
                cv2.imshow("Video", frame)
                # Write the frame to the output video
                out.write(frame)

        cap.release()
        out.release()
        cv2.destroyAllWindows()
    return trg_path


def create_arrays(vid_src, trg_dir):
    videos = glob.glob(vid_src + r'/*.mp4')
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic
    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)

    land = ['LEFT_EYE', 'NOSE', 'LEFT_EAR', 'LEFT_ELBOW', 'LEFT_EYE_INNER', 'LEFT_HIP', 'LEFT_EYE_OUTER', 'LEFT_SHOULDER',
            'LEFT_WRIST', 'MOUTH_LEFT', 'MOUTH_RIGHT', 'RIGHT_EAR', 'RIGHT_EYE', 'RIGHT_ELBOW', 'RIGHT_EYE_INNER',
            'RIGHT_EYE_OUTER', 'RIGHT_SHOULDER', 'RIGHT_WRIST', 'RIGHT_HIP', 'LEFT_PINKY', 'LEFT_INDEX', 'LEFT_THUMB',
            'RIGHT_PINKY', 'RIGHT_INDEX', 'RIGHT_THUMB'
            ]

    for i in videos:
        cap = cv2.VideoCapture(i)

        file_name = i.split("\\")[-1]
        file = file_name.split(".")[0]
        print(file_name)

        video_land = []
        frame_val = 0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print("frame count is", frame_count)

        #Initiated holistic model
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

            if not cap.isOpened():
                print(f"Error opening video file: {i}")
                continue

            while True:

                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, (640, 480))

                #Recolor Feed
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                image_height, image_width, _ = image.shape

                #Make detections
                results = holistic.process(image)

                pose_data = np.zeros(1)

                for landmark_name in land:
                    # Get the PoseLandmark enum for the current landmark name
                    landmark_enum = getattr(mp_holistic.PoseLandmark, landmark_name)

                    if results.pose_landmarks:
                        landmark = results.pose_landmarks.landmark[landmark_enum]
                        if landmark:
                            pose_x = landmark.x
                            pose_y = landmark.y
                            pose_z = landmark.z if hasattr(landmark, 'z') else 0.0
                            pose_data = np.concatenate([pose_data, np.array([pose_x, pose_y, pose_z])])
                    else:
                        pose_data = np.concatenate((pose_data, np.zeros(3)))

                # Remove the initial zeros used for initialization
                pose_data = pose_data[1:]
                lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() \
                    if results.left_hand_landmarks else np.zeros(21 * 3)
                rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() \
                    if results.right_hand_landmarks else np.zeros(21 * 3)

                frame_counter = (1/frame_count)*frame_val
                frame_val += 1
                value = np.array([frame_counter])
                pose_land = np.concatenate([pose_data, lh, rh, value])
                video_land.append(pose_land)

                #Recolor image back to BGR for rendering
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                #Left hand
                mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                              mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                                              mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2))

                #Right hand
                mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                              mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                                              mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2))

                #Pose detections
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

                cv2.imshow('Holistic Model Detections', image)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

        final_array = np.concatenate(video_land)
        np.savetxt(f'{trg_dir}/{file}.skels', final_array, delimiter=' ', fmt='%.5f', encoding="utf-8", newline=' ')

        cap.release()
        cv2.destroyAllWindows()

    return trg_dir


def concat_data(req_skels_name, src_skel_path, trg_concat_skel_path):
    # req_skels_name can be train/test/dev
    data = glob.glob(src_skel_path + r'/*.skels')
    req = []
    for temp in data:
        temp_data = np.loadtxt(temp)
        # print(temp_data.shape)
        req.append(temp_data)

    with open(f'{trg_concat_skel_path}/{req_skels_name}.skels', 'a') as file:
        for array in req:
            print(array.shape)
            np.savetxt(file, array, fmt='%.5f', newline=' ')
            file.write('\n')

    return trg_concat_skel_path


def create_files_file(file_name, vid_src, trg_path):
    #file_name is the name of the file you want to create it can be train/test/dev
    names = glob.glob(vid_src + r'/*.mp4')

    with open(f"{trg_path}/{file_name}.files", 'a') as file:
        for name in names:
            file_name = name.split("\\")[-1]
            file_name = file_name.split(".")[0]
            file_name = str(file_name)
            file.write(f"{file_name}/" + file_name + '\n')
    return trg_path


def create_text_gloss(file_type, vid_src, gloss_trg_path, text_trg_path):
    # the parameter file_type can be train/test/dev
    files = glob.glob(vid_src + '/*.mp4')

    bad_char = ['!', '*', '(', ')', '_', '<', '>', '{', '}', '&', '%', '#', ':', ';']

    with open(f'{gloss_trg_path}/{file_type}.gloss', 'a') as data_file:
        for file in files:
            file = file.split("\\")[-1]
            file_text = file.split(".")[0]
            for i in bad_char:
                file_text = file_text.replace(i, ' ')
            file_text = file_text.upper()
            data_file.write(file_text)
            data_file.write('\n')

    with open(f'{text_trg_path}/{file_type}.text', 'a') as data_file:
        for file in files:
            file = file.split("\\")[-1]
            file_text = file.split(".")[0]
            for i in bad_char:
                file_text = file_text.replace(i, ' ')
            file_text = file_text.lower()
            data_file.write(file_text)
            data_file.write('\n')

    return tuple(gloss_trg_path, text_trg_path)
