import os
import numpy as np
import pickle
import cv2

fps = 5.0
coeff = '0.5-0.5-1.0'
data = pickle.load(open("/data/jun/projects/reward_induced_mtl_rep/datasets/vizdoom/{}/episode_0.pkl".format(coeff), 'rb'))
frames = np.repeat(np.array(data['obs']), 3, axis=1).transpose((0, 2, 3, 1))
path = './{}.avi'.format(coeff)

n, h, w, c = frames.shape
print(path)
print(n, h, w, c)

video = cv2.VideoWriter(path, 0, fps, (w, h), 1)
print(video)

for frame in frames:
    video.write(frame.astype(np.uint8))

video.release()
# cv2.destroyAllWindows()

