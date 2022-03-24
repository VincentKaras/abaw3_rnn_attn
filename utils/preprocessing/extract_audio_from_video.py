import subprocess
from pathlib import Path
from argparse import ArgumentParser

if __name__ == "__main__":

    print("Audio ripper for Affwild2 videos")
    parser = ArgumentParser(description="Rip audio from Affwild2 videos")
    parser.add_argument("--video_dir", type=str, default="F:\\DeepLearningData\\Affwild2_ABAW2022\\videos")
    parser.add_argument("--audio_dir", type=str, default="F:\\DeepLearningData\\Affwild2_ABAW2022\\audio")
    parser.add_argument("--audio_format", type=str, default="wav", help="Output format for the audio")
    parser.add_argument("--audio_sr", type=int, default=16000, help="Output audio sample rate")

    args = parser.parse_args()

    video_dir = Path(args.video_dir)
    assert video_dir.exists(), "Video dir {} not found".format(str(video_dir))

    audio_dir = Path(args.audio_dir)
    if not audio_dir.exists():
        audio_dir.mkdir(exist_ok=True)

    # audio_properties
    audio_sr = args.audio_sr
    audio_codec = "pcm_s16le"

    # glob the video dir for mp4 files
    ext = "mp4"
    video_files = video_dir.glob("**/*.{}".format(ext))

    for file in video_files:

        #vf = str(file)
        fid = file.stem
        # audio_output
        audio_out = Path(audio_dir) / (fid + ".wav")
        subprocess_string = ["ffmpeg", "-i", str(file), "-vn", "-acodec", audio_codec, "-ar", str(audio_sr), str(audio_out)]
        print(subprocess_string)
        # run the subprocess
        out = subprocess.run(subprocess_string, stderr=subprocess.STDOUT, stdout=subprocess.PIPE)
        print(out.stdout.decode())

    print("Done")