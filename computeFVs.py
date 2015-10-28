import argparse
import os, subprocess, ThreadPool
import classify_library

"""
Uses multi-threading to extract IDTFs and compute the Fisher Vectors (FVs) for
each of the videos in the input list (vid_in). The Fisher Vectors are output
in the fv_dir
"""

# ...
COMPUTE_FV = 'python ./computeFVstream.py'


# This is is the function that each worker will compute.
def processVideo(vid,vid_dir,idt_dir,fv_dir,gmm_list):
    """
    gmm_list is the file of the saved list of GMMs
    """
    vid_file = os.path.join(vid_dir,vid)
    fv_file = os.path.join(fv_dir, vid.split('.')[0]+'.fv')
    extractFV(video_file, output_file, gmm_listf)


def extractFV(videoName, outputBase, gmm_list):
    """
    Extracts the IDTFs, constructs a Fisher Vector, and saves the Fisher Vector at outputBase
    outputBase: the full path to the newly constructed fisher vector.
    gmm_list: file of the saved list of gmms
    """
    subprocess.call('%s %s | %s %s %s' % (dtBin, resizedName, COMPUTE_FV, outputBase, gmm_list), shell=True)
    return True


#python computeFVs.py vid_dir vid_list fv_dir gmm_list
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("vid_dir", help="Directory of the video dataset", type=str)
    parser.add_argument("vid_list", help="list of input videos in .txt file", type=str)
    parser.add_argument("fv_dir", help="output directory to save FVs (.fv files)", type=str)
    parser.add_argument("gmm_list", help="File of saved list of GMMs", type=str)
    args = parser.parse_args()

    MAIN_DIR = args.vid_dir
    VID_DIR = os.path.join(MAIN_DIR,'videos')
    IDT_DIR = os.path.join(MAIN_DIR,'features','idt')
    FV_DIR = args.fv_dir
    vid_list = args.vid_list

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
