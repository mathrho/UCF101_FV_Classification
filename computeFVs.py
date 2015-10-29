import argparse
import os, subprocess, ThreadPool
import IDT_feature
import computeFV
import classify_library

"""
Uses multi-threading to extract IDTFs and compute the Fisher Vectors (FVs) for
each of the videos in the input list (vid_list). The Fisher Vectors are output
in the fv_dir
"""

# Improved Dense Trajectories binary
dtBin = './DenseTrackStab'
# ...
COMPUTE_FV = 'python ./computeFVstream.py'


# This is is the function that each worker will compute.
def processVideo(vid,IDT_DIR,FV_DIR,gmm_list):
    """
    Extracts the IDTFs, constructs a Fisher Vector, and saves the Fisher Vector at FV_DIR
    output_file: the full path to the newly constructed fisher vector.
    gmm_list: file of the saved list of gmms
    """
    input_file = os.path.join(IDT_DIR, vid.split('.')[0]+'.bin')
    output_file = os.path.join(FV_DIR, vid.split('.')[0]+'.fv')

    if not os.path.exists(input_file):
        print '%s IDT Feature does not exist!' % vid
        return False

    if os.path.exists(output_file):
        print '%s Fisher Vector exists, skip!' % vid
        return False

    video_desc = IDT_feature.vid_descriptors(IDT_feature.read_IDTF_file(input_file))
    computeFV.create_fisher_vector(gmm_list, video_desc, output_file)
    return True


def processVideoFrames(vid,IDT_DIR,FV_DIR,gmm_list):
    """
    Extracts the IDTFs, constructs a Fisher Vector for each video frame, and saves the Fisher Vector at FV_DIR
    output_file: the full path to the newly constructed fisher vector.
    gmm_list: file of the saved list of gmms
    """
    # do nothing
    


#python computeFVs.py vid_dir vid_list fv_dir gmm_list
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("vid_dir", help="Directory of the video features", type=str)
    parser.add_argument("vid_list", help="List of input videos in .txt file", type=str)
    parser.add_argument("fv_dir", help="Output directory to save FVs (.fv files)", type=str)
    parser.add_argument("gmm_list", help="File of saved list of GMMs", type=str)
    args = parser.parse_args()

    IDT_DIR = args.vid_dir
    FV_DIR = args.fv_dir

    f = open(args.vid_list, 'r')
    input_videos = f.readlines()
    f.close()
    input_videos = [line.split()[0] for line in [video.rstrip() for video in input_videos]]
    
    ###Just to prevent overwriting already processed vids
    completed_vids = [filename.split('.')[0] for filename in os.listdir(FV_DIR) if filename.endswith('.npz')]
    overlap = [vid for vid in input_videos if vid.split('.')[0] in completed_vids]
    
    #Multi-threaded FV construction.
    numThreads = 10
    pool = ThreadPool.ThreadPool(numThreads)
    for vid in input_videos:
        if vid not in overlap:
            pool.add_task(processVideo,vid,IDT_DIR,FV_DIR,args.gmm_list)
    pool.wait_completion()
