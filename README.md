# clip-fid

most of this is from:
https://wandb.ai/dalle-mini/dalle-mini/reports/CLIP-score-vs-FID-pareto-curves--VmlldzoyMDYyNTAy

### installing things
```
pip install -r requirements.txt

# the CLIP score implementation from wandb is in jax. In future, we will probably switch to a pytorch one.
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### running it

(in top level folder of this repo)

bash ./scripts/coco_get_data.sh

python ./scripts/coco_prepare_data.sh
