import deeplabcut
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date

# different prints to explain program for user usage
def print_usage(label):
    if label == "error":
        print("\nPlease follow instuctions\n")
    elif label == "start":
        print ("\nWelcome to the program to do you work using DeepLabCut. \nPlease load your project or start a new.\n\n")
        print ("Type 1 to load project \nType 2 to start a new project")
        print ("\nType exit to exit the program\n")
        return input()
    elif label == "program":
        print('\nUsage:\n')
        print(' Argument 1: Name of project')
        print(' Argument 2: Name of researcher')
        return str(input())
    elif label == "project":
        print('\nUsage:\n')
        print(' Type 1 for: Extracting frames.')
        print(' Type 2 for: Labeling frames.')
        print(' Type 3 for: Chekcing labels (Optional)')
        print(' Type 4 for: Create training dataset')
        print(' Type 5 for: Train dataset with set labels')
        print(' Type 6 for: Evaluate network')
        print(' Type 7 for: Analyze vidoes')
        print(' Type 8 for: Plot trajectories')
        print(' Type 9 for: Make labeled video')
        print(' Type 10 for: Add more videos')
        print(' Type 11 for: Plot video results')
        print(' Type 0 to exit')

def create_dict_list(path, type):
    directory_list = list()
    for root, dirs, files in os.walk(path, topdown=False):
        if type == 0:
            for name in dirs:
                if len(os.path.join(root, name).split("/")) == 5 or len(os.path.join(root, name).split("/")) == 4:
                    directory_list.append(os.path.join(root, name).split("/")[-1])
        elif type == 1:
            for file in files:
                if file.endswith('.mpg') or file.endswith('.mp4'):
                    directory_list.append(file)

    for project in range(len(directory_list)):
        print ("Type " + str(project) + " for project: " + directory_list[project])

    command = int(input())
    while command not in range(len(directory_list)):
        print ("please choose correct number")
        command = int(input())
    return [directory_list[command]]

def main():

    plt.close('all')
    exit = False
    while exit == False:
        choice = print_usage("start")

        if choice == "exit":
            exit = True
            sys.exit(0)
        elif choice == "1":
            command = create_dict_list("/data/11012579/projects/", 0)
        elif choice == "2":
            command = print_usage("program")
            command = command.split(" ")
            map_with_videos = create_dict_list("/data/11012579/videos/",0)[0]
            command.append("/data/11012579/videos/"+map_with_videos+"/"+create_dict_list("/data/11012579/videos/"+map_with_videos+"/",1)[0])


        if len(command) == 1:
                exit = usingDeep('/data/11012579/projects/'+ command[0] + '/config.yaml', command[0])
        elif len(command) == 3:
            print ("Check project name: \nName of project:" + command[0] +
                    "\nYour name:" + command[1] + "\nPath to video:" + command[2] +
                    "\nto contineu type: check \nTo change type: redo")
            input_2 = "redo/check"
            while input_2 not in ["check", "redo"]:
                print ("Type check or redo!")
                input_2 = str(input())
                if input_2 == "check":
                    deeplabcut.create_new_project(command[0], command[1], [command[2]], working_directory='/data/11012579/projects', copy_videos= False)
                    path_config = '/data/11012579/projects/'+ command[0] + '-' + command[1] + '-' + str(date.today()) + '/config.yaml'
                    print ("\nGo to projects -> your project -> config.yaml\
                            \nEdit: numframes2pick to select how manny frames you want to add \
                            \nEdit: bodyparts and type the bodyparts you want to keep track of\
                            \nEdit: Skeleton to draw lines between the tracked bodyparts\
                            \nRun ipython home.py to continue")
                    exit = usingDeep(path_config, command[0])

        else:
            print_usage("error")

def usingDeep(path_config, project):
    exit = False
    while exit == False:
        video_path = path_config.split("/")
        video_path = '/' + video_path[1]  + '/' + video_path[2] + '/' + video_path[3] + '/' + video_path[4] + '/videos/'
        print_usage("project")
        action = input()
        while action not in [0,1,2,3,4,5,6,7,8,9,10,11,12]:
            try:
                action = int(action)
                if action not in [0,1,2,3,4,5,6,7,8,9,10,11,12]:
                    print ("please type number between 0 and 12")
                    action = input()
            except ValueError:
                print ("Please enter number")
                action = input()
            print("been here")
        if action == 0:
            return False
        elif action == 1:
            print("do you want to crop the video? yes/no")
            if input() == "yes" or "y":
                print("how many videos you you want to crop? (use number: 1,2,3 etc)")
                crops = int(input())
                print("only crop all the video's and than exit")
                for loop in range(0,crops):
                    deeplabcut.extract_frames(path_config, 'manual', 'kmeans', crop=True)
            deeplabcut.extract_frames(path_config, 'automatic', 'kmeans', crop=True)
        elif action == 2:
            deeplabcut.label_frames(path_config)
        elif action == 3:
            deeplabcut.check_labels(path_config)
        elif action == 4:
            deeplabcut.create_training_dataset(path_config)
        elif action == 5:
            with open("training_network.py") as fp:
                lines = fp.readlines()
                lines[3] = lines[3].split("=")
                lines[3] = lines[3][0] + "= '" + path_config + "'\n"

            with open("training_network.py", "w") as fp:
                for line in lines:
                    fp.writelines(line)

            print ("run: sbatch slurm.sh")
            return True
        elif action == 6:
            try:
                deeplabcut.evaluate_network(path_config, Shuffles=[1], trainingsetindex=0, plotting=None, show_errors=True, comparisonbodyparts='all', gputouse=None, rescale=False)
            except OSError as e:
                print ("file does not exist")
        elif action == 7:
            print("\nType video name in project/videos you want to analyze")
            video_path = video_path + create_dict_list(path_config[:-11]+"videos/", 1)[0]
            with open("training_network.py") as fp:
                lines = fp.readlines()
                lines[3] = lines[3].split("=")
                lines[3] = lines[3][0] + "= '" + path_config + "'\n"
                lines[4] = lines[4].split("=")
                lines[4] = lines[4][0] + "= '" + video_path + "'\n"

            with open("training_network.py", "w") as fp:
                for line in lines:
                    fp.writelines(line)
            print ("run: sbatch slurm.sh after changing the command in training_network.py")
            return True
        elif action == 8:
            print("\nChoose the video in project/videos you want to plot trajectories from")
            video_path = video_path + create_dict_list(path_config[:-11]+"videos/", 1)[0]
            print(video_path)
            deeplabcut.plot_trajectories(path_config, [video_path], filtered=True)
        elif action == 9:
            print("\nChoose the video in project/videos you want to make a labeled video from")
            video_path = video_path + create_dict_list(path_config[:-11]+"videos/", 1)[0]
            deeplabcut.create_labeled_video(path_config,[video_path], videotype='.mp4', draw_skeleton=True)
        elif action == 10:
            print("\nChoose where to upload the video from")
            video_path = '/data/11012579/videos/' + create_dict_list('/data/11012579/videos/', 0)[0]
            print("\nChoose which video to upload")
            video_path_list = [video_path + "/" + create_dict_list(video_path, 1)[0]]
            while True:
                print("\nDo you want to add more videos?\nType yes or no")
                if input() == 'yes':
                    video_path_list.append(video_path + "/" + create_dict_list(video_path, 1)[0])
                else:
                    deeplabcut.add_new_videos(path_config, video_path_list, copy_videos = False)
                    break
        elif action == 11:
            print("also here")
            Dlc_results2 = pd.read_hdf('/data/11012579/videos/vidDLC_resnet50_demo_project_grab2Feb7shuffle1_11500.h5')
            Dlc_results2.plot()
        else:
            print_usage("error")

        print("klaar")

if __name__ == "__main__":
    main()

# deeplabcut.add_new_videos('/data/11012579/projects/rat7-Jesse-2020-02-25/config.yaml', ['/data/11012579/videos/rat_7/Day_3_LD_S_VEH_rat7.mpg', '/data/11012579/videos/rat_7/Day_2_LD_L_CNO_rat7.mpg', ''], copy_videos = False)
# deeplabcut.add_new_videos('/data/11012579/projects/rat7-Jesse-2020-02-25/config.yaml', ['/data/11012579/videos/rat_7/Day_3_LD_S_VEH_rat7.mpg', '/data/11012579/videos/rat_7/Day_2_LD_L_CNO_rat7.mpg', '/data/11012579/videos/rat_7/Day_4_LD_S_CNO_rat7.mpg'], copy_videos = False)
