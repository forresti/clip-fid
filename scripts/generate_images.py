"""
TODO:

1. load hugging face model
2. load list of captions
3. decide where to save output
   3.1. data format...
        outputs/<datetime>_<args>/
                                  captions.txt # do I need this?
                                  images/
                                         00000_<caption>.jpg
                                         00001_<caption>.jpg

4. run model on captions and save the output

"""

from datetime import datetime as dt
import json
from slugify import slugify
import os
from PIL import Image
import torch
from diffusers import StableDiffusionPipeline

def get_output_dir():
   datetime_format = '%Y-%m-%d_%H.%M.%S.%f'
   cur_time = dt.now().strftime(datetime_format)

   out_dir = f'./results/{cur_time}' # TODO: add some more arguments / flags / info to this directory name.
   os.makedirs(out_dir, exist_ok=True)
   return out_dir

def get_caption_out_fname(output_dir, annotation):
   out_fname = f"{annotation['id']}_{slugify(annotation['caption'])}.jpg"
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

   for anno in captions[0:10]:

      # TODO: do I need to reset the model memory or anything?
      model_output = pipe(prompt=anno['caption'], num_images_per_prompt=1)
      img = model_output['images'][0] # this is a PIL image.

      out_fname = get_caption_out_fname(out_dir, anno)

      img.save(out_fname)




if __name__ == "__main__":
   main()
