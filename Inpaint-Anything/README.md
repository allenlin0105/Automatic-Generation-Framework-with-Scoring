# Inpaint Anything: SAM Meets Image Inpainting
Inpaint Anything can inpaint anything in images. Users can select any object in an image by clicking on it. With [SAM](https://arxiv.org/abs/2304.02643), [LaMa](https://arxiv.org/abs/2109.07161) and [Stable Diffusion (SD)](https://arxiv.org/abs/2112.10752), Inpaint Anything is able to remove the object smoothly. Further, prompted by user input text, Inpaint Anything can fill the object with any desired content.

You can change `--coords_type key_in` to `--coords_type click` if your machine has a display device. If `click` is set, after running the above command, the image will be displayed.
(1) Use *left-click* to record the coordinates of the click. It supports modifying points, and only last point coordinates are recorded.
(2) Use *right-click* to finish the selection.

## fill-anything

Click on an object, type in what you want to fill, and Inpaint Anything will fill it.
- Click on an object;
- [SAM](https://segment-anything.com/) segments the object out;
- Input a text prompt;
- Text-prompt-guided inpainting models fill the "hole" according to the text.

### Installation
Requires `python>=3.8`
```bash
python -m pip install torch torchvision torchaudio
python -m pip install -e segment_anything
python -m pip install diffusers transformers accelerate scipy safetensors
pip install opencv-python
pip install matplotlib
pip install timm
```

### Usage
Download the model checkpoints provided in [Segment Anything](./segment_anything/README.md) (e.g., [sam_vit_h_4b8939.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)) and put them into `./pretrained_models`.

Place sample images into `../test_images`. The result will be saved in `../result_images`.

Specify an image, a point and text prompt, and run:
```bash
python ./Inpaint-Anything/fill_anything.py \
    --input_img ./test_images/test3.jpg \
    --coords_type key_in \
    --point_coords 302 405 \
    --point_labels 1 \
    --text_prompt "a man wearing jacket" \
    --dilate_kernel_size 50 \
    --output_dir ./test_images/results \
    --sam_model_type "vit_h" \
    --sam_ckpt ./Inpaint-Anything/pretrained_models/sam_vit_h_4b8939.pth
```
