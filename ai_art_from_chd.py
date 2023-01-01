import io
import os
import warnings
import time
import concurrent.futures


from PIL import Image
from stability_sdk import client
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation

MAX_GENERATIONS = 50
SOURCE_DIR = './tmp/'
TARGET_DIR = './ai_images/'+time.strftime("%Y%m%d-%H%M%S")+'/'

if not os.path.exists(TARGET_DIR):
    os.makedirs(TARGET_DIR)

# Sign up for an account at the following link to get an API Key.
# https://beta.dreamstudio.ai/membership

# Click on the following link once you have created an account to be taken to your API Key.
# https://beta.dreamstudio.ai/membership?tab=apiKeys

# Set up our connection to the API.
stability_api = client.StabilityInference(
    key=os.environ['STABILITY_KEY'], # API Key reference.
    verbose=True, # Print debug messages.
    engine="stable-diffusion-v1-5", # Set the engine to use for generation. For SD 2.0 use "stable-diffusion-v2-0".
    # Available engines: stable-diffusion-v1 stable-diffusion-v1-5 stable-diffusion-512-v2-0 stable-diffusion-768-v2-0 
    # stable-diffusion-512-v2-1 stable-diffusion-768-v2-1 stable-inpainting-v1-0 stable-inpainting-512-v2-0
)

def art_from_art(img_path, title): 
    img = Image.open(img_path)
    # Set up our initial generation parameters.
    answers2 = stability_api.generate(
        # prompt="Abstract art done using oil paint. Artefacts of the painting are clearly visible including brush strokes and paint grains."+
        # "Themes are happy and joyful and evoke nature and the sea / sand. " +
        # "Sand, wood, glass, magazine cutouts and other materials are layered in the painting and mixed in with the paint creating visible textures.",
        prompt = "Abstract, oil painting, highly detailed visibile brush strokes and paint grains add texture. Some other materials like sand or magazine cutouts are included. Evokes the see and nature, happy mood.",
        init_image=img, # Assign our previously generated img as our Initial Image for transformation.
        start_schedule=0.8, # Set the strength of our prompt in relation to our initial image.
        steps=50, # Amount of inference steps performed on image generation. Defaults to 30. 
        cfg_scale=9.0, # Influences how strongly your generation is guided to match your prompt.
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
                img2.save(TARGET_DIR+title+"-img2img.png") # Save our generated image with its seed number as the filename and the img2img suffix so that we know this is our transformed image.

    return "Got inspired by: "+title


# Get a list of all files in the directory
files = os.listdir(SOURCE_DIR)
# Filter the list to only include images
images = [f for f in files if f.lower().endswith('.jpg') or f.lower().endswith('.png')]
# count = 0
# for image in images:
#     if(count < MAX_GENERATIONS):
#         print("Inspired by:" + image)
#         art_from_art(SOURCE_DIR+image, image)
#         count+=1


with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_GENERATIONS) as executor:
     # Submit the queries to the executor
    results = [executor.submit(art_from_art, SOURCE_DIR+image, image) for image in images[:MAX_GENERATIONS]]

    # Wait for the results to be completed
    concurrent.futures.wait(results)

    # Print the results
    for future in results:
        print(future.result())
