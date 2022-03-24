import subprocess
from pathlib import Path
from argparse import ArgumentParser

if __name__ == "__main__":

    print("Audio mono converter for Affwild2 videos")
    parser = ArgumentParser(description="convert audio from Affwild2 videos")
    parser.add_argument("--output_dir", type=str, default="F:\\DeepLearningData\\Affwild2_ABAW2022\\videos", help="dir to write converted files")
    parser.add_argument("--audio_dir", type=str, default="F:\\DeepLearningData\\Affwild2_ABAW2022\\audio")
    parser.add_argument("--audio_format", type=str, default="wav", help="Output format for the audio")
    parser.add_argument("--audio_sr", type=int, default=16000, help="Output audio sample rate")

    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    
    if not out_dir.exists():
        out_dir.mkdir(parents=True)

    audio_dir = Path(args.audio_dir)
    if not audio_dir.exists():
        audio_dir.mkdir(exist_ok=True)

    # audio_properties
    audio_sr = args.audio_sr
    audio_codec = "pcm_s16le"

    # glob the audio dir for mp3 files
    ext = args.audio_format
    audio_files = audio_dir.glob("**/*.{}".format(ext))

    for file in audio_files:

        #vf = str(file)
        fid = file.stem
        # audio_output
        audio_out = out_dir / (fid + ".wav")
        subprocess_string = ["ffmpeg", "-i", str(file), "-ac", "1", str(audio_out)]
        print(subprocess_string)
        # run the subprocess
        out = subprocess.run(subprocess_string, stderr=subprocess.STDOUT, stdout=subprocess.PIPE)
        print(out.stdout.decode())

    print("Done")