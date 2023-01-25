
import argparse
from datetime import datetime as dt
import json
from slugify import slugify
import os
from PIL import Image
import torch
from diffusers import StableDiffusionPipeline

def get_parser():
   parser = argparse.ArgumentParser()
   # TODO: add args
   return parser

def get_output_dir():
   datetime_format = '%Y-%m-%d_%H.%M.%S.%f'
   cur_time = dt.now().strftime(datetime_format)

   out_dir = f'./results/{cur_time}' # TODO: add some more arguments / flags / info to this directory name.
   os.makedirs(out_dir, exist_ok=True)
   return out_dir

def get_caption_out_fname(output_dir, annotation):
   out_fname = f"{annotation['image_id']}_{slugify(annotation['caption'])}.jpg"
   out_fname = os.path.join(output_dir, out_fname)
   return out_fname

def get_pipe():
   pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, revision="fp16")
   # pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16, revision="fp16")
   pipe = pipe.to("cuda")
   return pipe

def get_captions():
   with open('coco/annotations/captions_val2014.json', 'r') as f:
      caption_data = json.load(f)

   return caption_data['annotations']

def main():
   out_dir = get_output_dir()
   pipe = get_pipe()
   captions = get_captions()

   out_pointers = []

   for i,anno in enumerate(captions[0:20000]):

      # TODO: do I need to reset the model memory or anything?
      model_output = pipe(prompt=anno['caption'], num_images_per_prompt=1)
      img = model_output['images'][0] # this is a PIL image.

      out_fname = get_caption_out_fname(out_dir, anno)

      img.save(out_fname)

      out_pointers.append({
         'image_id': anno['image_id'],
         'caption': anno['caption'],
         'generated_image': os.path.basename(out_fname),
      })

      if i%100 == 0:
         with open(os.path.join(out_dir, f'generated_ckpt_{i}.json'), 'w') as f:
            json.dump(out_pointers, f)

   with open(os.path.join(out_dir, 'generated.json'), 'w') as f:
      json.dump(out_pointers, f)


if __name__ == "__main__":
   main()
