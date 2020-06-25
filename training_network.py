import os
import deeplabcut
from pathlib import Path
path_config = '/data/11012579/projects/ratt9-jessegroot-2020-04-17/config.yaml'
path_video = '/data/11012579/projects/ratt9-jessegroot-2020-04-17/videos/Day_4_LD_S_CNO_rat9_cut.mp4'
#deeplabcut.train_network(path_config, shuffle=1, displayiters=10, saveiters=100)
deeplabcut.analyze_videos(path_config, [path_video], videotype='.mpg', shuffle=1, trainingsetindex=0, gputouse=None, save_as_csv=False, destfolder=None, batchsize=None, cropping=None, get_nframesfrommetadata=True, TFGPUinference=True, dynamic=(False, 0.5, 10))



