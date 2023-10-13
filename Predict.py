!pip install torch torchvision torchaudio --extra-index-url https://dowmload.pytorch.org/whl/cu116

!pip install 'git+https://github.com/facebookresearch/detectron2.git'

!pip install gradio opencv-python scipy

import torch,detectron2
!nvcc --version
TORCH_VERSION= ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch:",TORCH_VERSION,";cuda: ",CUDA_VERSION)
#print("detectron2:",detectron2.__version__)


!pip install -q git+https://github.com/huggingface/transformers.git

def coco_panoptic_palette():
    return [
        [220, 20, 60],   # Red
        [255, 99, 71],   # Tomato
        [255, 165, 0],   # Orange
        [255, 215, 0],   # Gold
        [0, 255, 0],     # Lime
        [0, 255, 255],   # Cyan
        [0, 0, 255],     # Blue
        [255, 0, 255],   # Magenta
        [128, 128, 0],   # Olive
        [255, 192, 203]  # Pink
    ]

def cityscapes_palette():
    return [
        [128, 64, 128],  # Road
        [244, 35, 232],  # Sidewalk
        [70, 70, 70],     # Building
        [102, 102, 156],  # Wall
        [190, 153, 153],  # Fence
        [153, 153, 153],  # Pole
        [250, 170, 30],   # Traffic Light
        [220, 220, 0],    # Traffic Sign
        [107, 142, 35],   # Vegetation
        [152, 251, 152],  # Terrain
        [0, 130, 180],    # Sky
        [220, 20, 60],    # Person
        [255, 0, 0],      # Rider
        [0, 0, 142],      # Car
        [0, 0, 70],       # Truck
        [0, 60, 100],      # Bus
        [0, 80, 100],      # Train
        [0, 0, 230],       # Motorcycle
        [119, 11, 32]      # Bicycle
    ]

def ade_palette():
    return [
        [0, 0, 0],       # Background
        [120, 120, 120],  # Wall
        [180, 120, 120],  # Building
        [6, 230, 230],    # Pavement
        [80, 50, 50],     # Tree
        [4, 200, 3],      # SignSymbol
        [120, 120, 80],   # Fence
        [140, 140, 140],  # Car
        [204, 5, 255],    # Pedestrian
        [230, 230, 230],  # Bicyclist
    ]


import torch
import  random
import numpy as np
from PIL import Image
from collections import defaultdict
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer

from transformers import MaskFormerImageProcessor, Mask2FormerForUniversalSegmentation




from transformers import Mask2FormerConfig,Mask2FormerModel

configuration=Mask2FormerConfig()

model= Mask2FormerModel(configuration)

configuration=model.config
configuration


import json
!pip install torch

device = "cuda" if torch.cuda.is_available() else "cpu" 


def load_model_and_processor(model_ckpt:str,device:str):
  model=Mask2FormerForUniversalSegmentation.from_pretrained(model_ckpt).to(torch.device(device))
  model.eval()
  image_preprocessor=MaskFormerImageProcessor.from_pretrained(model_ckpt)
  return model,image_preprocessor




def load_default_ckpt(segmentation_task:str):
  if segmentation_task=="semantic":
    default_pretrained_ckpt="facebook/mask2former-swin-base-ade-semantic"
  elif segmentation_task=="panoptic":
    default_pretrained_ckpt="facebook/mask2former-swin-large-coco-panoptic"
  return default_pretrained_ckpt


def draw_panoptic_segmentation(predicted_segmentation_map,seg_info,image):
  metadata=MetadataCatalog.get("coco_2017_val_panoptic")
  for res in seg_info:
    res['category_id']=res.pop('label_id')
    pred_class=res['category_id']
    isthing=pred_class in metadata.thing_dataset_id_to_contiguous_id.values()
    res['isthing']=bool(isthing)
  visualizer = Visualizer(np.array(image)[:, :, ::-1], metadata=metadata, instance_mode=ColorMode.IMAGE)
  out=visualizer.draw_panoptic_seg_predictions(
  predicted_segmentation_map.cpu(),seg_info,alpha=0.5
  )
  output_img=Image.fromarray(out.get_image())
  return output_img

def draw_semantic_segmentation(segmentation_map,image,palette):
  color_segmentation_map=np.zeros((segmentation_map.shape[0], segmentation_map.shape[1],3),dtype=np.uint8)
  for label,color in enumerate(palette):
    color_segmentation_map[segmentation_map-1== label,:]=color
  ground_truth_color_seg=color_segmentation_map[...,::-1]

  img=np.array(image)*0.5+ground_truth_color_seg*0.5
  img=img.astype(np.uint8)
  return img
def predict_masks(input_img_path: str, segmentation_task: str):
    
   try:
        default_pretrained_ckpt = load_default_ckpt(segmentation_task)
        model, image_processor = load_model_and_processor(default_pretrained_ckpt, device)

        image = Image.open(input_img_path)
        inputs = image_processor(images=image, return_tensors="pt").to(torch.device(device))

        with torch.no_grad():
            outputs = model(**inputs)

        if segmentation_task == "semantic":
            result = image_processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
            predicted_segmentation_map = result.cpu().numpy()
            palette = ade_palette()
            output_result = draw_semantic_segmentation(predicted_segmentation_map, image, palette)
        else:
            result = image_processor.post_process_panoptic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
            predicted_segmentation_map = result["segmentation"]
            seg_info = result['segments_info']
            output_result = draw_panoptic_segmentation(predicted_segmentation_map, seg_info, image)

        return output_result
   except Exception as e:
        return json.dumps({"error": str(e)})


import gradio as gr


with gr.Blocks() as Mask2former_GUI:
    with gr.Box():
        with gr.Row():
            choose_segmentation = gr.Dropdown(["semantic", "panoptic"], value="panoptic", label="Panoptic or Semantic")
        with gr.Box():
            with gr.Row():
                with gr.Column(scale=1):
                    input_image = gr.Image(type='filepath', label="Place your image here:", show_label=True)
                with gr.Column(scale=3):
                    output_mask = gr.Image(label="Mask2Former Output:", show_label=True)

    with gr.Box():
        with gr.Row():
            with gr.Column(scale=1):
                start_run = gr.Button("Run ")
           

    start_run.click(predict_masks, inputs=[input_image, choose_segmentation], outputs=output_mask)

if __name__ == "__main__":
    Mask2former_GUI.launch(share=True, debug=True)


