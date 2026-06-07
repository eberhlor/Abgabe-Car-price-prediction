import base64
import json
import os

import gradio as gr
import torch
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image as PILImage
from transformers import CLIPModel, CLIPProcessor, pipeline

load_dotenv()

OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client  = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# Path to the locally saved fine-tuned ViT (produced by training notebook).
# If you pushed to the Hub, replace with your Hub model ID, e.g.:
# VIT_MODEL_DIR = "your-hf-username/car-make-model-vit"
VIT_MODEL_DIR = os.getenv("VIT_MODEL_DIR", "eberhlor/car-prediction-model-vit")

# ── Transfer Learning ViT (fine-tuned on Stanford Cars) ───────────────────────
vit_classifier = pipeline("image-classification", model=VIT_MODEL_DIR)

# ── CLIP Zero-Shot ────────────────────────────────────────────────────────────
CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"
clip_model      = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
clip_processor  = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
clip_model.eval()

device     = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = clip_model.to(device)

# Build text prompts from the ViT model's id2label mapping so both models
# share the exact same 196 class names.
from transformers import ViTForImageClassification

_vit_cfg   = ViTForImageClassification.from_pretrained(VIT_MODEL_DIR).config
ID2LABEL   = _vit_cfg.id2label                       # {0: "Acura Integra Type R 2001", …}
LABEL_LIST = [ID2LABEL[i] for i in range(len(ID2LABEL))]
TEXT_PROMPTS = [f"a photo of a {name}" for name in LABEL_LIST]

# Pre-compute text features once at startup
_text_inputs = clip_processor(
    text=TEXT_PROMPTS, return_tensors="pt", padding=True, truncation=True
).to(device)
with torch.no_grad():
    _text_out      = clip_model.get_text_features(**_text_inputs)
    _text_features = _text_out if isinstance(_text_out, torch.Tensor) else _text_out.pooler_output
    _text_features = _text_features / _text_features.norm(dim=-1, keepdim=True)


# ── Helper ────────────────────────────────────────────────────────────────────
def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ── Classifiers ───────────────────────────────────────────────────────────────
def classify_with_vit(image_path: str) -> dict:
    """Fine-tuned ViT: returns top-5 car make/model predictions."""
    results = vit_classifier(image_path, top_k=5)
    return {r["label"]: round(r["score"], 4) for r in results}


def classify_with_clip(image_path: str) -> dict:
    """CLIP zero-shot: scores the image against all 196 class prompts."""
    image        = PILImage.open(image_path).convert("RGB")
    image_inputs = clip_processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        image_out      = clip_model.get_image_features(**image_inputs)
        image_features = image_out if isinstance(image_out, torch.Tensor) else image_out.pooler_output
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    logits = (image_features @ _text_features.T) * clip_model.logit_scale.exp()
    probs  = logits.softmax(dim=-1)[0].cpu()

    top5_values, top5_idxs = probs.topk(5)
    return {LABEL_LIST[i.item()]: round(v.item(), 4) for i, v in zip(top5_idxs, top5_values)}


def classify_with_openai(image_path: str) -> dict:
    """OpenAI Vision model as a third opinion."""
    if openai_client is None:
        return {
            "error": (
                "Missing OPENAI_API_KEY. Add it to your environment or .env file "
                "to enable OpenAI classification."
            )
        }

    prompt = (
        "Look at this image of a car and identify the make (manufacturer), model name, "
        "and approximate year. Be as specific as possible. "
        "Return valid JSON with exactly these keys: "
        "make (string), model (string), year (integer or null), "
        "label (string combining make model year, e.g. 'BMW 3 Series 2012'), "
        "confidence (number between 0 and 1), reasoning (short explanation)."
    )

    base64_image = encode_image(image_path)
    response = openai_client.responses.create(
        model=OPENAI_MODEL,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text",  "text": prompt},
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{base64_image}",
                    },
                ],
            }
        ],
    )

    try:
        clean = response.output_text.strip()
        if clean.startswith("```"):
            clean = clean.split("```", 2)[1]
            if clean.startswith("json"):
                clean = clean[4:]
            clean = clean.rsplit("```", 1)[0].strip()
        return json.loads(clean)
    except json.JSONDecodeError:
        return {
            "raw_response": response.output_text,
            "warning": "OpenAI response was not valid JSON.",
        }


# ── Disagreement explanation (CV → LLM integration) ──────────────────────────
def explain_disagreement(image_path: str, vit_top1: str, clip_top1: str) -> str:
    """
    Called only when ViT and CLIP disagree on the top-1 prediction.
    OpenAI Vision acts as an arbiter: it inspects the image and explains
    which prediction is more plausible and why the two CV models diverged.

    This is the integration point between the CV block (visual predictions)
    and the LLM block (natural-language interpretation of visual features).
    """
    if openai_client is None:
        return (
            "OpenAI API key not set — cannot generate explanation. "
            "Add OPENAI_API_KEY to your .env file to enable this feature."
        )

    prompt = (
        f"Two computer vision models disagree on the make and model of this car:\n"
        f"  • Fine-tuned ViT predicted:  \"{vit_top1}\"\n"
        f"  • CLIP zero-shot predicted:  \"{clip_top1}\"\n\n"
        "Please look at the image carefully and:\n"
        "1. State which prediction you think is more likely correct and why.\n"
        "2. Explain what visual features in the image may have caused the disagreement "
        "(e.g. camera angle, partial occlusion, similar body styles across model years, "
        "lighting conditions, background clutter).\n"
        "3. Give a confidence assessment: high / medium / low.\n"
        "Answer in 3–5 concise sentences."
    )

    base64_image = encode_image(image_path)
    try:
        response = openai_client.responses.create(
            model=OPENAI_MODEL,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text",  "text": prompt},
                        {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    ],
                }
            ],
        )
        return response.output_text.strip()
    except Exception as e:
        return f"OpenAI explanation failed: {e}"


# ── Main classify function ────────────────────────────────────────────────────
def classify_car(image):
    vit_output    = classify_with_vit(image)
    clip_output   = classify_with_clip(image)
    openai_output = classify_with_openai(image)

    # Top-1 label from each CV model
    vit_top1  = max(vit_output,  key=vit_output.get)  if isinstance(vit_output,  dict) and "error" not in vit_output  else None
    clip_top1 = max(clip_output, key=clip_output.get) if isinstance(clip_output, dict) and "error" not in clip_output else None

    result = {
        "ViT Transfer Learning – Top 5 predictions": vit_output,
        "CLIP Zero-Shot – Top 5 predictions":        clip_output,
        "OpenAI Vision Classification":              openai_output,
    }

    # Integration point: when the two CV models disagree, OpenAI Vision
    # interprets the visual features and explains which result is more plausible.
    if vit_top1 and clip_top1 and vit_top1 != clip_top1:
        result["Models disagree – OpenAI visual explanation"] = explain_disagreement(
            image, vit_top1, clip_top1
        )

    return result


# ── Gradio Interface ──────────────────────────────────────────────────────────
iface = gr.Interface(
    fn=classify_car,
    inputs=gr.Image(type="filepath"),
    outputs=gr.JSON(),
    title="Car Make & Model Classification",
    description=(
        "Upload a car photo and compare three approaches to identify the make, model, and year:\n\n"
        "• **ViT Transfer Learning** – a Vision Transformer fine-tuned on 196 Stanford Cars classes\n"
        "• **CLIP Zero-Shot** – OpenAI's CLIP model without any task-specific training\n"
        "• **OpenAI Vision** – GPT-4 vision model with free-form reasoning (requires OPENAI_API_KEY)\n\n"
        "When ViT and CLIP disagree, OpenAI Vision automatically inspects the image and explains "
        "in natural language which prediction is more plausible and what visual features caused the disagreement."
    ),
    examples=[
        ["example_images/car1.jpg"],
        ["example_images/car2.jpg"],
        ["example_images/car3.jpg"],
    ],
)

iface.launch()
