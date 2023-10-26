import numpy as np
import cv2
import torch

from dtw import dtw
from constants import PAD_TOKEN
# data_path = "data/A_Lot.skels"
# data = np.loadtxt(data_path)

# PAD_TOKEN = '<pad>'
# references = None
# skip_frames = 1
# sequence_ID = None


def plot_video(joints, file_path, video_name, references=None, skip_frames=1, sequence_ID=None):
    # Number of elements to pick
    # num_elements_to_pick = 202

    # Initialize an empty list to store the picked elements
    # req_list = []
    # ref_list = []
    #
    # num_set = int(len(data) / 202)
    #
    # print(num_set)
    #
    # # Loop to pick elements
    # for i in range(num_set):
    #     start_index = i * 202
    #     end_index = (i + 1) * 202
    #     subset = data[start_index:end_index]
    #     req_list.append(subset)
    #
    # if references is not None:
    #     ref_set = int(len(references) / 202)
    #     for i in range(ref_set):
    #         start_index = i * 202
    #         end_index = (i + 1) * 202
    #         subset_ref = references[start_index:end_index]
    #         ref_list.append(subset_ref)
    #     references = np.array(req_list)
    #
    # # Convert the list of picked elements back to a NumPy array
    # picked_elements_array = np.array(req_list)
    # # references = picked_elements_array

    video_file = file_path + "/{}.mp4".format(video_name.split(".")[0])
    # Define the landmark names and connections to draw the skeleton
    # Define the landmark names including hand landmarks
    landmark_names = [
         'NOSE', 'LEFT_EYE_INNER', 'LEFT_EYE', 'LEFT_EYE_OUTER', 'RIGHT_EYE_INNER', 'RIGHT_EYE',
    'RIGHT_EYE_OUTER', 'LEFT_EAR', 'RIGHT_EAR', 'MOUTH_LEFT', 'MOUTH_RIGHT', 'LEFT_SHOULDER', 'RIGHT_SHOULDER',
    'LEFT_ELBOW', 'RIGHT_ELBOW', 'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_PINKY', 'RIGHT_PINKY',
    'LEFT_INDEX', 'RIGHT_INDEX', 'LEFT_THUMB', 'RIGHT_THUMB', 'LEFT_HIP', 'RIGHT_HIP',
        'LEFT_WRIST', 'LEFT_THUMB_CMC', 'LEFT_THUMB_MCP', 'LEFT_THUMB_IP',
        'LEFT_THUMB_TIP', 'LEFT_INDEX_FINGER_MCP', 'LEFT_INDEX_FINGER_PIP', 'LEFT_INDEX_FINGER_DIP',
        'LEFT_INDEX_FINGER_TIP', 'LEFT_MIDDLE_FINGER_MCP', 'LEFT_MIDDLE_FINGER_PIP', 'LEFT_MIDDLE_FINGER_DIP',
        'LEFT_MIDDLE_FINGER_TIP', 'LEFT_RING_FINGER_MCP', 'LEFT_RING_FINGER_PIP', 'LEFT_RING_FINGER_DIP',
        'LEFT_RING_FINGER_TIP', 'LEFT_PINKY_MCP', 'LEFT_PINKY_PIP', 'LEFT_PINKY_DIP', 'LEFT_PINKY_TIP',
        'RIGHT_WRIST', 'RIGHT_THUMB_CMC', 'RIGHT_THUMB_MCP', 'RIGHT_THUMB_IP', 'RIGHT_THUMB_TIP',
        'RIGHT_INDEX_FINGER_MCP', 'RIGHT_INDEX_FINGER_PIP', 'RIGHT_INDEX_FINGER_DIP', 'RIGHT_INDEX_FINGER_TIP',
        'RIGHT_MIDDLE_FINGER_MCP', 'RIGHT_MIDDLE_FINGER_PIP', 'RIGHT_MIDDLE_FINGER_DIP', 'RIGHT_MIDDLE_FINGER_TIP',
        'RIGHT_RING_FINGER_MCP', 'RIGHT_RING_FINGER_PIP', 'RIGHT_RING_FINGER_DIP', 'RIGHT_RING_FINGER_TIP',
        'RIGHT_PINKY_MCP', 'RIGHT_PINKY_PIP', 'RIGHT_PINKY_DIP', 'RIGHT_PINKY_TIP'
    ]

    # Define connections between landmarks, including hand connections
    connections = [
        # Body connections
        ('LEFT_EYE', 'LEFT_EYE_INNER'), ('LEFT_EYE', 'LEFT_EYE_OUTER'), ('LEFT_SHOULDER', 'LEFT_ELBOW'),
        ('LEFT_ELBOW', 'LEFT_WRIST'), ('LEFT_WRIST', 'LEFT_PINKY'), ('LEFT_WRIST', 'LEFT_INDEX'),
        ('LEFT_WRIST', 'LEFT_THUMB'), ('NOSE', 'LEFT_EYE'), ('NOSE', 'RIGHT_EYE'), ('NOSE', 'MOUTH_LEFT'),
        ('NOSE', 'MOUTH_RIGHT'), ('LEFT_HIP', 'RIGHT_HIP'), ('LEFT_HIP', 'LEFT_SHOULDER'),
        ('RIGHT_HIP', 'RIGHT_SHOULDER'), ('LEFT_SHOULDER', 'RIGHT_SHOULDER'), ('RIGHT_SHOULDER', 'RIGHT_ELBOW'),
        ('RIGHT_ELBOW', 'RIGHT_WRIST'), ('RIGHT_WRIST', 'RIGHT_PINKY'), ('RIGHT_WRIST', 'RIGHT_INDEX'),
        ('RIGHT_WRIST', 'RIGHT_THUMB'),

        # Hand connections
        ('LEFT_THUMB_MCP', 'LEFT_THUMB_IP'), ('LEFT_THUMB_IP', 'LEFT_THUMB_TIP'),
        ('LEFT_THUMB_CMC', 'LEFT_THUMB_MCP'), ('LEFT_WRIST', 'LEFT_INDEX_FINGER_MCP'),
        ('LEFT_INDEX_FINGER_MCP', 'LEFT_INDEX_FINGER_PIP'), ('LEFT_INDEX_FINGER_PIP', 'LEFT_INDEX_FINGER_DIP'),
        ('LEFT_INDEX_FINGER_DIP', 'LEFT_INDEX_FINGER_TIP'), ('LEFT_MIDDLE_FINGER_MCP', 'LEFT_INDEX_FINGER_MCP'),
        ('LEFT_MIDDLE_FINGER_MCP', 'LEFT_MIDDLE_FINGER_PIP'), ('LEFT_MIDDLE_FINGER_PIP', 'LEFT_MIDDLE_FINGER_DIP'),
        ('LEFT_MIDDLE_FINGER_DIP', 'LEFT_MIDDLE_FINGER_TIP'), ('LEFT_MIDDLE_FINGER_MCP', 'LEFT_RING_FINGER_MCP'),
        ('LEFT_RING_FINGER_MCP', 'LEFT_RING_FINGER_PIP'), ('LEFT_RING_FINGER_PIP', 'LEFT_RING_FINGER_DIP'),
        ('LEFT_RING_FINGER_DIP', 'LEFT_RING_FINGER_TIP'), ('LEFT_RING_FINGER_MCP', 'LEFT_PINKY_MCP'),
        ('LEFT_PINKY_MCP', 'LEFT_PINKY_PIP'), ('LEFT_PINKY_PIP', 'LEFT_PINKY_DIP'),
        ('LEFT_PINKY_DIP', 'LEFT_PINKY_TIP'), ('LEFT_WRIST', 'LEFT_PINKY_MCP'), ('LEFT_THUMB_CMC', 'LEFT_WRIST'),

        ('RIGHT_THUMB_MCP', 'RIGHT_THUMB_IP'), ('RIGHT_THUMB_IP', 'RIGHT_THUMB_TIP'),
        ('RIGHT_THUMB_CMC', 'RIGHT_THUMB_MCP'), ('RIGHT_WRIST', 'RIGHT_INDEX_FINGER_MCP'),
        ('RIGHT_INDEX_FINGER_MCP', 'RIGHT_INDEX_FINGER_PIP'), ('RIGHT_INDEX_FINGER_PIP', 'RIGHT_INDEX_FINGER_DIP'),
        ('RIGHT_INDEX_FINGER_DIP', 'RIGHT_INDEX_FINGER_TIP'), ('RIGHT_MIDDLE_FINGER_MCP', 'RIGHT_INDEX_FINGER_MCP'),
        ('RIGHT_MIDDLE_FINGER_MCP', 'RIGHT_MIDDLE_FINGER_PIP'), ('RIGHT_MIDDLE_FINGER_PIP', 'RIGHT_MIDDLE_FINGER_DIP'),
        ('RIGHT_MIDDLE_FINGER_DIP', 'RIGHT_MIDDLE_FINGER_TIP'), ('RIGHT_MIDDLE_FINGER_MCP', 'RIGHT_RING_FINGER_MCP'),
        ('RIGHT_RING_FINGER_MCP', 'RIGHT_RING_FINGER_PIP'), ('RIGHT_RING_FINGER_PIP', 'RIGHT_RING_FINGER_DIP'),
        ('RIGHT_RING_FINGER_DIP', 'RIGHT_RING_FINGER_TIP'), ('RIGHT_RING_FINGER_MCP', 'RIGHT_PINKY_MCP'),
        ('RIGHT_PINKY_MCP', 'RIGHT_PINKY_PIP'), ('RIGHT_PINKY_PIP', 'RIGHT_PINKY_DIP'),
        ('RIGHT_PINKY_DIP', 'RIGHT_PINKY_TIP'), ('RIGHT_WRIST', 'RIGHT_PINKY_MCP'), ('RIGHT_THUMB_CMC', 'RIGHT_WRIST')
    ]

    width, height = 640, 640
    FPS = (10 // skip_frames)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    if references is None:
        out = cv2.VideoWriter(video_file, fourcc, float(FPS), (width, height))
    elif references is not None:
        out = cv2.VideoWriter(video_file, fourcc, float(FPS), (width*2, height))  # Long

    num_frames = 0

    for (j, frame_joints) in enumerate(joints):
        # Your numpy array containing pose coordinates [x1, y1, z1, x2, y2, z2, ...]
        pose_coordinates = frame_joints[:-1]
        pose_coordinates = pose_coordinates * 640

        # Reached padding
        if PAD_TOKEN in frame_joints:
            continue

        # Create a white image (you can adjust the width and height as needed)
        white_image = np.zeros((height, width, 3), dtype=np.uint8)
        white_image.fill(255)  # Fill the image with white color
        cv2.putText(white_image, "Predicted Sign Pose", (180, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255), 2)

        # Ensure the pose_coordinates array has a length of 3 times the number of landmarks
        num_landmarks = len(pose_coordinates) // 3
        assert len(pose_coordinates) % 3 == 0, "Invalid pose coordinates length"

        # Split pose_coordinates into individual landmark coordinates
        landmarks = [pose_coordinates[i:i + 3] for i in range(0, len(pose_coordinates), 3)]

        # Draw the detected pose landmarks and skeleton on the white image
        for i in range(num_landmarks):
            x, y, _ = pose_coordinates[i * 3:i * 3 + 3]
            if 0 < x <= width and 0 < y <= height:
                cv2.circle(white_image, (int(x), int(y)), 5, (0, 0, 255), -1)

        # Draw the skeleton connections
        for connection in connections:
            landmark_start, landmark_end = connection
            index_start = landmark_names.index(landmark_start)
            index_end = landmark_names.index(landmark_end)
            x1, y1 = int(landmarks[index_start][0]), int(landmarks[index_start][1])
            x2, y2 = int(landmarks[index_end][0]), int(landmarks[index_end][1])
            if 0 < x1 <= width and 0 < y1 <= height and 0 < x2 <= width and 0 < y2 <= height:
                # Draw the connection line
                cv2.line(white_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # If reference is provided, create and concatenate on the end
        if references is not None:
            # Extract the reference joints
            ref_joints = references[j]
            # Initialise frame of white
            ref_frame = np.ones((height, width, 3), np.uint8) * 255

            # Cut off the percent_tok and multiply each joint by 3 (as was reduced in training files)
            ref_joints = ref_joints[:-1]
            ref_joints = ref_joints * 640

            ref_num_landmarks = len(ref_joints) // 3
            assert len(ref_joints) % 3 == 0, "Invalid pose coordinates length"

            # Split pose_coordinates into individual landmark coordinates
            landmarks_ref = [ref_joints[i:i + 3] for i in range(0, len(pose_coordinates), 3)]

            # Draw these joints on the frame
            # Draw the detected pose landmarks and skeleton on the white image
            for i in range(ref_num_landmarks):
                x, y, _ = pose_coordinates[i * 3:i * 3 + 3]
                if 0 < x <= width and 0 < y <= height:
                    cv2.circle(ref_frame, (int(x), int(y)), 5, (0, 0, 255), -1)

            # Draw the skeleton connections
            for connection1 in connections:
                landmark_start, landmark_end = connection1
                index_start = landmark_names.index(landmark_start)
                index_end = landmark_names.index(landmark_end)
                x1, y1 = int(landmarks_ref[index_start][0]), int(landmarks_ref[index_start][1])
                x2, y2 = int(landmarks_ref[index_end][0]), int(landmarks_ref[index_end][1])
                if 0 < x1 <= width and 0 < y1 <= height and 0 < x2 <= width and 0 < y2 <= height:
                    # Draw the connection line
                    cv2.line(ref_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.putText(ref_frame, "Ground Truth Pose", (180, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 2)

            white_image = np.concatenate((white_image, ref_frame), axis=1)

            if sequence_ID is not None:
                sequence_ID_write = "Sequence ID: " + sequence_ID.split("/")[-1]
                cv2.putText(ref_frame, sequence_ID_write, (700, 635), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 0, 0), 2)

            cv2.line(white_image, (640, 0), (640, 640), (0, 0, 0), 2)

        # Display the image with landmarks and skeleton
        num_frames += 1
        # cv2.imshow("Image with Pose Landmarks and Skeleton", white_image)
        out.write(white_image)

        # Wait for a short time and listen for a key press (exit on ESC key)
        key = cv2.waitKey(10)
        if key == 27:  # ESC key
            break

    out.release()
    cv2.destroyAllWindows()


# Apply DTW to the produced sequence, so it can be visually compared to the reference sequence
def alter_DTW_timing(pred_seq,ref_seq):

    # Define a cost function
    euclidean_norm = lambda x, y: np.sum(np.abs(x - y))

    # Cut the reference down to the max count value
    _ , ref_max_idx = torch.max(ref_seq[:, -1], 0)
    if ref_max_idx == 0: ref_max_idx += 1
    # Cut down frames by counter
    ref_seq = ref_seq[:ref_max_idx,:].cpu().numpy()

    # Cut the hypothesis down to the max count value
    _, hyp_max_idx = torch.max(pred_seq[:, -1], 0)
    if hyp_max_idx == 0: hyp_max_idx += 1
    # Cut down frames by counter
    pred_seq = pred_seq[:hyp_max_idx,:].cpu().numpy()

    # Run DTW on the reference and predicted sequence
    d, cost_matrix, acc_cost_matrix, path = dtw(ref_seq[:,:-1], pred_seq[:,:-1], dist=euclidean_norm)

    # Normalise the dtw cost by sequence length
    d = d / acc_cost_matrix.shape[0]

    # Initialise new sequence
    new_pred_seq = np.zeros_like(ref_seq)
    # j tracks the position in the reference sequence
    j = 0
    skips = 0
    squeeze_frames = []
    for (i, pred_num) in enumerate(path[0]):

        if i == len(path[0]) - 1:
            break

        if path[1][i] == path[1][i + 1]:
            skips += 1

        # If a double coming up
        if path[0][i] == path[0][i + 1]:
            squeeze_frames.append(pred_seq[i - skips])
            j += 1
        # Just finished a double
        elif path[0][i] == path[0][i - 1]:
            new_pred_seq[pred_num] = avg_frames(squeeze_frames)
            squeeze_frames = []
        else:
            new_pred_seq[pred_num] = pred_seq[i - skips]

    return new_pred_seq, ref_seq, d

# Find the average of the given frames
def avg_frames(frames):
    frames_sum = np.zeros_like(frames[0])
    for frame in frames:
        frames_sum += frame

    avg_frame = frames_sum / len(frames)
    return avg_frame
