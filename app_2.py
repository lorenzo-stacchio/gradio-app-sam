import numpy as np
import os
from PIL import Image, ImageOps, ImageDraw
import gradio as gr
import torch
from segment_anything import sam_model_registry, SamPredictor

# --------- config ----------
MODEL_TYPE = os.getenv("SAM_MODEL_TYPE", "vit_l")  # vit_b|vit_l|vit_h
CKPT_PATH  = os.getenv("SAM_CKPT", "checkpoints/sam_vit_l_0b3195.pth")
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

# --------- load SAM once ----------
sam = sam_model_registry[MODEL_TYPE](checkpoint=CKPT_PATH)
sam.to(device=DEVICE); sam.eval()
predictor = SamPredictor(sam)

# --------- session state ----------
STATE = {"img": None, "points": []}  # points: (x,y,label)

def _pil_to_rgb(pil: Image.Image):
    if pil.mode != "RGB": pil = pil.convert("RGB")
    return np.array(pil)

def _overlay(image_rgb: np.ndarray, mask_bool: np.ndarray, alpha=0.6):
    out = image_rgb.copy()
    color = np.array([255, 0, 255], dtype=np.uint8)  # magenta
    out[mask_bool] = (alpha * color + (1 - alpha) * out[mask_bool]).astype(np.uint8)
    return out



def on_load(img_pil: Image.Image):
    # reset points, compute embedding once
    STATE["points"].clear()
    print("INLOAD")
    rgb = _pil_to_rgb(img_pil)
    STATE["img"] = rgb
    predictor.set_image(rgb)

def on_click(pil_image, evt: gr.SelectData, label_mode: str):
    # print()
    # if STATE["img"] is None or evt is None or evt.value is None:
    #     return gr.update()
    x, y = evt.index  # pixel coords
    label = 1 if label_mode == "Foreground" else 0
    STATE["points"].append((float(x), float(y), int(label)))
    print(STATE["points"])
    with open("test.txt", "w") as f:
        f.writelines(" ".join(str(STATE["points"])))      

    # draw a cross on a copy of the base image
    draw = ImageDraw.Draw(pil_image)
    size = 6
    color = (0, 255, 0) if label == 1 else (255, 0, 0)  # green or red
    draw.line((x - size, y, x + size, y), fill=color, width=2)
    draw.line((x, y - size, x, y + size), fill=color, width=2)
    
    return pil_image, f"Clicked at (x={x}, y={y})"


def on_segment(multimask: bool, direct:bool):
    # caso 1: non stiamo facendo segmentazione subito
    # caso 2: ho accumulato i punti ma non ho niente
    if (not direct or STATE["img"] is None) and (STATE["img"] is None or not STATE["points"]):
    # if STATE["img"] is None or not STATE["points"]:
        return None, "Add at least one point."
    pts = np.array([[p[0], p[1]] for p in STATE["points"]], dtype=np.float32)
    lbl = np.array([p[2] for p in STATE["points"]], dtype=np.int32)
    masks, scores, _ = predictor.predict(
        point_coords=pts, point_labels=lbl, multimask_output=bool(multimask)
    )
    best = int(np.argmax(scores))
    overlay = _overlay(STATE["img"], masks[best], 0.6)
    return overlay, f"Scores={scores.tolist()} (best={best})"

def invert_colors(img) :
    return img.convert('L')

with gr.Blocks() as demo:
    gr.Markdown("## Minimal image load & process test")

    with gr.Row():
        inp = gr.Image(type="pil", label="Input Image", interactive=True)
        out = gr.Image(type="pil", label="Processed Image")
    with gr.Row():
        label_mode = gr.Radio(["Foreground", "Background"], value="Foreground", label="Click label")
        
        info = gr.Textbox(label="Output")
    
    multimask = gr.Checkbox(value=True, label="Try multiple masks")
    direct = gr.Checkbox(value=True, label="Try direct segmentation")

    
    inp.upload(on_load, inputs=[inp])
    select_evt  = inp.select(on_click, inputs=[inp,label_mode], outputs=[inp, info])

    select_evt.then(on_segment, inputs=[multimask, direct], outputs=[out, info])

    btn = gr.Button("Segment")

    btn.click(on_segment, inputs=[multimask,direct], outputs=[out, info])


if __name__ == "__main__":
    demo.launch(debug=True, show_error=True, share= True)
    
    
