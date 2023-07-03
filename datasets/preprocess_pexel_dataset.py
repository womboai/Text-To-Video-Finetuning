# This script will download the pexels dataset from s3, and preprocess it.
# 1. For each video, we convert it into n second clips
# 2. For each n second clip, we run BLIP on it to get the caption, and then we save the caption and the clip
# %% run pip install
# !pip install -r requirements.txt
# %% imports
import boto3
import requests
import time 
import torch
from PIL import Image
from lavis.models import load_model_and_preprocess
from scenedetect import detect, ContentDetector
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.editor import VideoFileClip
from typing import List 
import glob
from concurrent.futures import ProcessPoolExecutor


# %% load blip 

# setup device to use
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

# loads BLIP-2 pre-trained model
model, vis_processors, _ = load_model_and_preprocess(name="blip2_t5", model_type="pretrain_flant5xxl", is_eval=True, device=device)


# %% download pexels dataset

s3 = boto3.client('s3', region_name="us-east-1")
S3_BUCKET = 'dev-wombo-dream-data'
PEXELS_DIR = 'video_datasets/pexels/videos/'
TMP_DIR = './tmp'
VERTICAL_VID_DIR = "/home/vivek/Text-To-Video-Finetuning/datasets/vertical_videos"

# %% process clips

# Download the pexels dataset from s3
def download_pexels_dataset():
    while True:
        print("Downloading next batch of files")
        kwargs = {'Bucket': S3_BUCKET, 'Prefix': PEXELS_DIR}
        resp = s3.list_objects_v2(**kwargs)
        for obj in resp['Contents']:
            key = obj['Key']
            print("Downloading", key)
            local_filename = key.split('/')[-1]
            local_filename = f"{TMP_DIR}/{local_filename}"
            print(f"downloading {key} to {local_filename}")
            s3.download_file(S3_BUCKET, key, local_filename)
            print("Downloaded", key)

        try:
            kwargs['ContinuationToken'] = resp['NextContinuationToken']
        except KeyError:
            break


def get_caption_from_image(raw_image: Image.Image) -> str:
    # prepare the image
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    # get the caption
    caption = model.generate({"image": image, "prompt": "Describe this image in as much detail as possible:"})
    print(caption)
    return caption[0]


def split_video_into_clips(video_path, clip_length=1) -> List[str]:
    # Load video
    clip = VideoFileClip(video_path)
    
    # Resample video at 24 fps
    clip = clip.set_duration(clip.duration).set_fps(24)

    # Calculate number of subclips
    duration = clip.duration
    num_clips = int(duration // clip_length)
    
    output_paths = []

    # Extract subclips
    for i in range(num_clips):
        start_time = i * clip_length
        end_time = (i + 1) * clip_length
        video_name = video_path.split('/')[-1]
        # remove .mp4
        if video_name.endswith('.mp4'):
            video_name = video_name[:-4]
        
        subclip_path = f"{VERTICAL_VID_DIR}/{video_name}_{i}.mp4"
        output_paths.append(subclip_path)
        
        # Extract subclip
        subclip = clip.subclip(start_time, end_time)
        subclip.write_videofile(subclip_path, codec='libx264')
   
    return output_paths


def get_clips_and_captions_from_video(video_path: str):
    # Split video into clips
    t1 = time.time()
    clip_paths = split_video_into_clips(video_path)
    print(f"Split {video_path} into {len(clip_paths)} clips in {time.time() - t1} seconds")
    # Get captions for each clip
    captions = []
    for clip_path in clip_paths:
        # Load clip
        clip = VideoFileClip(clip_path)

        # Run BLIP on clip
        image = Image.fromarray(clip.get_frame(0)).convert("RGB")
        t1 = time.time()
        caption = get_caption_from_image(image)
        print(f"Got caption for {clip_path} in {time.time() - t1} seconds")
        # save the caption (replace .mp4 with .txt)
        caption_path = f"{clip_path[:-4]}.txt"
        with open(caption_path, 'w') as f:
            f.write(caption)

# for each video in /tmp/, split it into clips, and run BLIP on each clip
def process_video(video_path):
    t1 = time.time()
    print("Processing", video_path)
    get_clips_and_captions_from_video(video_path)
    print(f"Processed {video_path} in {time.time() - t1} seconds")


def process_videos():
    print("Processing videos in parallel")
    video_paths = glob.glob(f"{TMP_DIR}/*.mp4")
    with ProcessPoolExecutor() as executor:
        executor.map(process_video, video_paths)


process_videos()