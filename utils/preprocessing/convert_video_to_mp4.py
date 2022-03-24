import subprocess
from pathlib import Path
from argparse import ArgumentParser

if __name__ == "__main__":

    print("Video format converter")

    parser = ArgumentParser(description="Convert all avi videos in Affwild2 to mp4")
    parser.add_argument("--video_dir", type=str, default="F:\\DeepLearningData\\Affwild2_ABAW2022\\videos\\batch1",
                        help="folder containing the avi videos, should be the batch1")

    args = parser.parse_args()
    video_dir = Path(args.video_dir)
    assert video_dir.exists(), "Video dir {} not found".format(str(video_dir))

    # collect the avi files
    avi_files = video_dir.glob("*.avi")

    # convert to mp4
    for avi in avi_files:

        fid = avi.stem
        outfile = video_dir / (fid + ".mp4")

        sub_string = ["ffmpeg", "-i", str(avi), "-c:v", "copy", "-c:a", "aac", str(outfile)] # need to reencode cause MP4 container cant have 16bit pcm
        print(sub_string)
        #run the subprocess
        out = subprocess.run(sub_string, stderr=subprocess.STDOUT, stdout=subprocess.PIPE)
        print(out.stdout.decode())

    print("Done")