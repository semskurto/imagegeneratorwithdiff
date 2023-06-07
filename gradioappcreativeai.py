# %% [code]
!pip install --upgrade diffusers accelerate transformers
!pip install git+https://github.com/facebookresearch/segment-anything.git
!pip install opencv-python pycocotools matplotlib onnxruntime onnx
!wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
!pip install gradio

import gradio as gr
import gc
from PIL import Image
import cv2
from io import BytesIO
import numpy as np
import torch
from segment_anything import SamPredictor, sam_model_registry
device = "cuda"

from diffusers import StableDiffusionInpaintPipeline
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting",
    torch_dtype=torch.float16,
)
pipe.to(device)

#subprocess.run(['wget', 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth'])
def generate_mask(image_path):
    #/kaggle/working/sam_vit_b_01ec64.pth
    sam = sam_model_registry["vit_b"](checkpoint="./sam_vit_b_01ec64.pth")
    sam.to(device="cpu")
    predictor = SamPredictor(sam)

    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

    predictor.set_image(image)
    image_shape = image.shape
    input_point = np.array([[image_shape[1] // 2, image_shape[0] // 2]])
    input_label = np.array([1])

    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )

    img_depth_inv = np.logical_not(masks[0])
    image_mask = Image.fromarray(img_depth_inv.astype(np.uint8) * 255)
    return image_mask


def process_text_and_image(image_path, text, num_samples):
    resized_mask = generate_mask(image_path).resize((512, 512))
    resized_image = Image.open(image_path).resize((512, 512))

    image = pipe(prompt=text,
                 image=resized_image,
                 mask_image=resized_mask,
                 num_images_per_prompt=num_samples
                 ).images

    return image


block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown("## Stable Diffusion with SAM")
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(source='upload', type="filepath")
            prompt = gr.Textbox(label="Prompt")
            run_button = gr.Button(label="Run")
            with gr.Accordion("Advanced options", open=False):
                num_samples = gr.Slider(label="Images", minimum=1, maximum=12, value=3, step=1)
                #image_resolution = gr.Slider(label="Image Resolution", minimum=256, maximum=768, value=512, step=64)
                #strength = gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
                #guess_mode = gr.Checkbox(label='Guess Mode', value=False)
                #low_threshold = gr.Slider(label="Canny low threshold", minimum=1, maximum=255, value=100, step=1)
                #high_threshold = gr.Slider(label="Canny high threshold", minimum=1, maximum=255, value=200, step=1)
                #ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=20, step=1)
                #scale = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=9.0, step=0.1)
                #seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, randomize=True)
                #eta = gr.Number(label="eta (DDIM)", value=0.0)
                #a_prompt = gr.Textbox(label="Added Prompt", value='best quality, extremely detailed')
                #n_prompt = gr.Textbox(label="Negative Prompt",
                #                      value='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality')
        with gr.Column():
            result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery").style(grid=2,
                                                                                                   height='auto')
    #ips = [input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength,
    #       scale, seed, eta, low_threshold, high_threshold]
    run_button.click(fn=process_text_and_image, inputs=[input_image, prompt, num_samples,],
                      outputs=[result_gallery])

block.launch(server_name='0.0.0.0', share=True)
