from pathlib import Path
import torch
import gradio as gr
from torch import nn

LABELS = Path("Gradio/Sketch-Recognition/class_names.txt").read_text().splitlines()

model = nn.Sequential(
    nn.Conv2d(1, 32, 3, padding="same"),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(32, 64, 3, padding="same"),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(64, 128, 3, padding="same"),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(1152, 256),
    nn.ReLU(),
    nn.Linear(256, len(LABELS)),
)
state_dict = torch.load("Gradio/Sketch-Recognition/pytorch_model.bin", map_location="cpu", weights_only=False)
model.load_state_dict(state_dict, strict=False)
model.eval()


def predict(im):
    if im is None:
        return {}
    
    try:
        import numpy as np
        
        # Debug: print input type and structure
        print(f"Input type: {type(im)}")
        if isinstance(im, dict):
            print(f"Dict keys: {im.keys()}")
        
        # Handle different input formats from sketchpad
        if isinstance(im, dict):
            if "composite" in im and im["composite"] is not None:
                im = im["composite"]
                print(f"Using composite, shape: {np.array(im).shape}")
            elif "layers" in im and im["layers"] and len(im["layers"]) > 0:
                im = np.array(im["layers"][0])
                print(f"Using layers[0], shape: {im.shape}")
            else:
                print("No valid data in dict")
                return {}
        
        if im is None:
            return {}
        
        # Convert to numpy array first
        if not isinstance(im, np.ndarray):
            im = np.array(im)
        
        print(f"Image shape: {im.shape}, dtype: {im.dtype}")
        
        # Convert to tensor
        x = torch.tensor(im, dtype=torch.float32)
        
        # Handle different image formats
        if x.dim() == 3:
            if x.shape[-1] == 4:  # RGBA
                # For sketchpad, the drawing is usually in RGB channels, alpha might be constant
                # Try RGB first, then alpha if needed
                rgb = x[:, :, :3].mean(dim=-1)
                alpha = x[:, :, 3]
                
                # Check which channel has more variation (likely the drawing)
                if rgb.std() > alpha.std():
                    x = 255 - rgb  # Invert RGB (black drawing on white background)
                else:
                    x = alpha  # Use alpha channel
                    
            elif x.shape[-1] == 3:  # RGB
                # Convert to grayscale and invert (assuming black drawing on white)
                x = 255 - x.mean(dim=-1)
        else:
            # Already grayscale, check if we need to invert
            if x.mean() > 127:  # Mostly white background
                x = 255 - x  # Invert so drawing is white on black
        
        # Normalize to 0-1
        x = x / 255.0
        
        print(f"Processed tensor shape: {x.shape}, min: {x.min():.3f}, max: {x.max():.3f}, mean: {x.mean():.3f}")
        
        # Resize to 28x28 (model expects this size)
        x = torch.nn.functional.interpolate(
            x.unsqueeze(0).unsqueeze(0), size=(28, 28), mode='bilinear', align_corners=False
        )
        
        with torch.no_grad():
            out = model(x)
        
        probabilities = torch.nn.functional.softmax(out[0], dim=0)
        values, indices = torch.topk(probabilities, 5)
        
        result = {LABELS[i]: v.item() for i, v in zip(indices, values)}
        print(f"Top prediction: {LABELS[indices[0]]} ({values[0]:.3f})")
        
        return result
    
    except Exception as e:
        print(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return {}

with gr.Blocks(theme="default", title="🎨 Pictionary AI") as interface:
    gr.Markdown("# 🎨 Pictionary AI")
    gr.Markdown("Draw any object and watch the AI guess what you're drawing in real-time!")
    
    with gr.Row():
        with gr.Column():
            sketch = gr.Sketchpad(
                label="Draw here!",
                height=400,
                width=400
            )
            with gr.Row():
                clear_btn = gr.Button("🗑️ Clear", variant="secondary")
                predict_btn = gr.Button("🔍 Predict", variant="primary")
        
        with gr.Column():
            output = gr.Label(
                label="AI Predictions",
                num_top_classes=5,
                show_label=True
            )
    
    # Event handlers for real-time prediction
    sketch.change(predict, inputs=sketch, outputs=output, show_progress=False)
    predict_btn.click(predict, inputs=sketch, outputs=output)
    clear_btn.click(lambda: (None, {}), outputs=[sketch, output])
    
    gr.Markdown("### 💡 Tips:")
    gr.Markdown("- Draw simple, recognizable objects\n- Use clear, bold strokes\n- The AI works best with common objects")

interface.launch(share=True, show_error=True)