# extract IDT features

import numpy as np
import subprocess, os
import sys

"""
Wrapper library for the IDTF executable.
Implements methods to extract IDTFs.
Seperate methods for extracting IDTF and computing Fisher Vectors.


Example usage:

python computeIDTF.py VID_DIR video_list.txt output_directory

"""
# Path to the video repository
UCF101_DIR = "/home/zhenyang/Workspace/data/UCF101"

# Improved Dense Trajectories binary
dtBin = './DenseTrackStab'


def extract(video_file, output_file):
    """
    Extracts the IDTFs and stores them in outputBase file.
    """
    if not os.path.exists(video_file):
        print '%s does not exist!' % video_file
        return False

    if os.path.exists(output_file):
        print '%s IDT Features exist, skip!' % video_file
        return False

    command = '%s -f %s -o %s' % (dtBin, video_file, output_file, )
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, universal_newlines=True)
    while proc.poll() is None:
        line = proc.stdout.readline()
        print(line)
    return True


if __name__ == '__main__':
    # Useage: python computeIDTF.py VID_DIR video_list.txt output_directory
    VID_DIR = sys.argv[1]
    video_list = sys.argv[2]
    IDT_DIR = sys.argv[3]
    try:
        f = open(video_list, 'r')
        videos = f.readlines()
        f.close()
        videos = [video.rstrip() for video in videos]
        for i in range(0, len(videos)):
            output_file = os.path.join(IDT_DIR,os.path.splitext(videos[i])[0]+".bin")
            video_file = os.path.join(VID_DIR,videos[i])
            print "generating IDTF for %s" % (videos[i],)
            extract(video_file, output_file)
            print "completed."
    except IOError:
        sys.exit(0)
