# Segment Anything Local

## Usage

Following usage is running on your computer. 

- Install Segment Anything（[more details about install Segment Anything](https://github.com/facebookresearch/segment-anything#installation):
  ```console
  $ pip install git+https://github.com/facebookresearch/segment-anything.git
  ```

- Make a new folder named `checkpoints` under this project，and put the downloaded weights files in `checkpoints`。You can download the weights using following URLs：

  - `vit_h`: [ViT-H SAM model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)
  - `vit_l`: [ViT-L SAM model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth)
  - `vit_b`: [ViT-B SAM model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)

- Run：
  ```
  python app_2.py
  ```

**Note：** Default model is `vit_l`，the demo can run on CPU. Default device is `cpu`
