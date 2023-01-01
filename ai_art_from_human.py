import os
import pprint
import random
from dotenv import load_dotenv
import dropbox
from dropbox.files import WriteMode
from dropbox.exceptions import ApiError, AuthError
import math
from PIL import Image
import io
import warnings
import time
import concurrent.futures
from functools import partial
import random
from PIL import Image
from stability_sdk import client
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation
import configparser


# Read local file `config.ini`.
config = configparser.ConfigParser()                                     
config.read('config.ini')

load_dotenv()

#Load secret constants.
DROPBOX_ID=os.environ['DROPBOX_ID']
DROPBOX_SECRET=os.environ['DROPBOX_SECRET']
DROPBOX_REFRESH_TOKEN=os.environ['DROPBOX_REFRESH_TOKEN']
SHARING_EMAIL=os.environ['SHARING_EMAIL']

#Load configuration constants
HUMAN_ART_PATH=config['DROPBOX']['HUMAN_ART_PATH']
AI_ART_PATH=config['DROPBOX']['AI_ART_PATH']
MAX_GENERATIONS = config['INFERENCE']['MAX_GENERATIONS']
PARAMETER_RANDOMNESS_SWING = config.getfloat('INFERENCE', 'PARAMETER_RANDOMNESS_SWING')
SAMPLE_SIZE = config.getint('SAMPLING','SAMPLE_SIZE')

TMP_DIR = './tmp/'
TMP_ART_DIR= './tmp-dreams/'

if not os.path.exists(TMP_DIR):
    os.makedirs(TMP_DIR)

if not os.path.exists(TMP_ART_DIR):
    os.makedirs(TMP_ART_DIR)

stability_api = client.StabilityInference(
    key=os.environ['STABILITY_KEY'], # API Key reference.
    verbose=True, # Print debug messages.
    engine="stable-diffusion-v1-5", # Set the engine to use for generation. For SD 2.0 use "stable-diffusion-v2-0".
    # Available engines: stable-diffusion-v1 stable-diffusion-v1-5 stable-diffusion-512-v2-0 stable-diffusion-768-v2-0 
    # stable-diffusion-512-v2-1 stable-diffusion-768-v2-1 stable-inpainting-v1-0 stable-inpainting-512-v2-0
)

# 1/ List files from DB API 
# 2/ Sample 21 (biased on recency)
# 3/ Download and crop each file 
# 4/ 'dream' 
# 5/ pushback to DB, in an aptly named folder 
# 6/ send email with link to DB folder. 


def art_from_art(img_path, title): 
    img = Image.open(img_path)
    # Set up our initial generation parameters.
    answers2 = stability_api.generate(
        prompt = "Abstract painting. using oil paint and acrylic, highly detailed, visibile brush strokes and paint grains and textures. sand or magazine cutouts are included. Evokes the see and nature, happy mood.",
        init_image=img, # Assign our previously generated img as our Initial Image for transformation.
        start_schedule=(0.7 * random.uniform(1-PARAMETER_RANDOMNESS_SWING, 1+PARAMETER_RANDOMNESS_SWING)), # Set the strength of our prompt in relation to our initial image.
        steps=50, # Amount of inference steps performed on image generation. Defaults to 30. 
        cfg_scale=(8.0 * random.uniform(1-PARAMETER_RANDOMNESS_SWING, 1+PARAMETER_RANDOMNESS_SWING)), # Influences how strongly your generation is guided to match your prompt.
                    # Setting this value higher increases the strength in which it tries to match your prompt.
                    # Defaults to 7.0 if not specified.
        width=512, # Generation width, defaults to 512 if not included.
        height=512, # Generation height, defaults to 512 if not included.
        sampler=generation.SAMPLER_K_DPMPP_2M # Choose which sampler we want to denoise our generation with.
                                                    # Defaults to k_dpmpp_2m if not specified. Clip Guidance only supports ancestral samplers.
                                                    # (Available Samplers: ddim, plms, k_euler, k_euler_ancestral, k_heun, k_dpm_2, k_dpm_2_ancestral, k_dpmpp_2s_ancestral, k_lms, k_dpmpp_2m)
    )

    # Set up our warning to print to the console if the adult content classifier is tripped.
    # If adult content classifier is not tripped, display generated image.
    for resp in answers2:
        for artifact in resp.artifacts:        
            if artifact.finish_reason == generation.FILTER:
                warnings.warn(
                    "Your request activated the API's safety filters and could not be processed."
                    "Please modify the prompt and try again.")
            if artifact.type == generation.ARTIFACT_IMAGE:
                global img2
                img2 = Image.open(io.BytesIO(artifact.binary))
                path = TMP_ART_DIR+title+"-img2img.png"
                img2.save(path)
                return path

def dream_from_file(dbx, file):
    tmp_path = './{}/{}'.format(TMP_DIR, file['name'])
    dbx.files_download_to_file(tmp_path, file['path_lower'])

    #Crop 
    im = Image.open(tmp_path)

    # Calculate the new width and height of the image
    width, height = im.size
    # aspect_ratio = width / height
    new_total_pixels = 1048576
    if width * height > new_total_pixels:
        scaling = math.sqrt((width * height) / new_total_pixels)        
    else:
        scaling = 1

    new_width = math.floor(width / scaling / 64) * 64
    new_height = math.floor(height / scaling / 64) * 64
    # Resize the image
    resized_im = im.resize((new_width, new_height), resample=Image.Resampling.BICUBIC)
    os.remove(tmp_path)
    # Save the resized image
    resized_im.save(tmp_path)

    return art_from_art(tmp_path, file['name'])

def retry_sharing_job(async_job_id):
    sharing_job = dbx.sharing_check_share_job_status(async_job_id)
    if sharing_job.is_complete():
        print("Async sharing job completed...")
        pass
    else:
        print("Async sharing job in progress")
        print("....waiting 3 seconds...")
        time.sleep(3)
        retry_sharing_job(async_job_id)



# 1/ List the files
with dropbox.Dropbox(oauth2_refresh_token=DROPBOX_REFRESH_TOKEN, app_key=DROPBOX_ID, app_secret=DROPBOX_SECRET) as dbx:

    entries = dbx.files_list_folder(HUMAN_ART_PATH, recursive=True).entries
    
    folder_list = [x.name for x in entries if 'size' not in dir(x)]
    
    all_files = [{'name': x.name, 'path_lower': x.path_lower, 'server_modified': x.server_modified} for x in entries if 'size' in dir(x) and (x.name.endswith('jpg') or x.name.endswith('png'))]
    
    sorted_by_recency = sorted(all_files, key=lambda x: x['server_modified'], reverse=True)
    weights = list(range(len(sorted_by_recency), 0, -1))

    # 2/ Sample based on recency
    sample = random.choices(sorted_by_recency, weights=weights, k=SAMPLE_SIZE)

    # Clear out any previous downloads & dreams
    for f in os.listdir(TMP_DIR):
        os.remove(os.path.join(TMP_DIR, f))
    
    for f in os.listdir(TMP_ART_DIR):
        os.remove(os.path.join(TMP_ART_DIR, f))

    # 3/ Download locally & prep, and dream! 
    print("Initiating dream session against sample: {}".format([s['name'] for s in sample]))
    dream_paths = map(partial(dream_from_file, dbx), sample)
        
    #5/ create folder and upload to DB
    folder_path = "{}Creations from {}".format(AI_ART_PATH, time.strftime("%B %d, %Y"))
    target_folder = dbx.files_create_folder(folder_path, autorename = True)
    
    for idx, dream_path in enumerate(dream_paths):
        dream = open(dream_path, "rb").read()
        write_path = "{}/{}-img2img.png".format(target_folder.path_display, idx)
        dbx.files_upload(dream, write_path, mode = (dropbox.files.WriteMode.overwrite))
        print("Generated: ", idx)

    #6/ Share! 
    sharing_folder = dbx.sharing_share_folder(target_folder.path_display)
    if sharing_folder.is_complete():
        sharing_folder_data = sharing_folder.get_complete()
    if sharing_folder.is_async_job_id():
        async_job_id = sharing_folder.get_async_job_id()
        # helper function will block until async sharing job completes
        retry_sharing_job(async_job_id)
        sharing_folder_job = dbx.sharing_check_share_job_status(async_job_id)
        sharing_folder_data = sharing_folder_job.get_complete()

    member_selector = dropbox.sharing.MemberSelector.email(SHARING_EMAIL)
    member =  dropbox.sharing.AddMember(member_selector, access_level=dropbox.sharing.AccessLevel.editor)
    res = dbx.sharing_add_folder_member(sharing_folder_data.shared_folder_id, [member])
    print(f"Shared folder {target_folder.path_display} with email {SHARING_EMAIL}")

    
