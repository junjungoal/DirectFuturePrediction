import os
import numpy as np
import pickle
import cv2

fps = 4.0
coeff = '0.0-1.0-0.0'
data = pickle.load(open("/data/jun/projects/reward_induced_mtl_rep/datasets/vizdoom/{}/episode_0.pkl".format(coeff), 'rb'))
frames = np.repeat(np.array(data['obs']), 3, axis=3)
path = '{}.avi'.format(coeff)

n, h, w, c = frames.shape

video = cv2.VideoWriter(path, 0, fps, (w, h))

for frame in frames:
    video.write(frame.astype(np.uint8))

cv2.destroyAllWindows()
video.release()

