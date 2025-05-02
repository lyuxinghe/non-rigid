import os
import numpy as np
import plotly.graph_objects as go
import rpad.visualize_3d.plots as vpl


# path = "/home/jacinto/data/insertion/tax3d-yk-creator/demonstrations/04-21-wp-2/execute_data/"
# # dirs = os.listdir(path)

# tax3d_dirs = [
#     "0426_124803", "0426_124626", "0426_124847", "0426_125221", 
#     "0426_124449", "0426_124711", "0426_125410", "0426_125319", 
#     "0426_125028", "0426_125453" 

# ]

# taxpose_dirs = [
#     "0426_130608" "0426_130704" "0426_130740" "0426_130823", 
#     "0426_130901" 
# ]


# for dir in tax3d_dirs:
#     data_path = os.path.join(path, dir, "tax3dv2_debug")
#     action = np.load(os.path.join(data_path, "action_points.npy"), allow_pickle=True)
#     anchor = np.load(os.path.join(data_path, "anchor_points.npy"), allow_pickle=True)
#     batch = dict(np.load(os.path.join(data_path, "data_batch.npz"), allow_pickle=True))
#     pred = dict(np.load(os.path.join(data_path, "pred_dict.npz"), allow_pickle=True))

#     action = action[::10, :]
#     anchor = anchor[::10, :]
#     action_seg = np.zeros(action.shape[0])
#     anchor_seg = np.ones(anchor.shape[0])
    
#     fig = vpl.segmentation_fig(
#         np.concatenate([
#             action, anchor
#         ], axis=0),
#         np.concatenate([
#             action_seg, anchor_seg
#         ], axis=0).astype(int),
#     )
#     fig.show()
#     quit()

data_dir = "/home/jacinto/data/insertion/tax3d-yk-creator/demonstrations/04-21-wp-2"
train_path_new = os.path.join(data_dir, "execute_data", "train")
test_path_new = os.path.join(data_dir, "execute_data", "test")
train_path_old = os.path.join(data_dir, "learn_data", "train")
test_path_old = os.path.join(data_dir, "learn_data", "test")


def viz_data(path):
    for file in os.listdir(path):
        if file.endswith(".npz"):
            data = np.load(os.path.join(path, file), allow_pickle=True)
            clouds = data["clouds"]
            classes = data["classes"]

            action = clouds[classes == 0]
            anchor = clouds[classes == 1]
            action_seg = np.zeros(action.shape[0],)
            anchor_seg = np.ones(anchor.shape[0],)
            fig = vpl.segmentation_fig(
                np.concatenate([
                    action, anchor
                ], axis=0),
                np.concatenate([
                    action_seg, anchor_seg
                ], axis=0).astype(int),
            )
            fig.show()

viz_data(train_path_old)
#$ viz_data(test_path_old)