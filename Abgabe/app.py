import base64
import json
import os
import pickle

import gradio as gr
import numpy as np
import torch
from openai import OpenAI
from PIL import Image as PILImage
from transformers import CLIPModel, CLIPProcessor, ViTForImageClassification, pipeline

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
VIT_MODEL_DIR  = os.getenv("VIT_MODEL_DIR", "eberhlor/car-prediction-model-vit")
INR_TO_CHF     = 0.0083
LUXURY_BRANDS  = [
    'Mercedes-Benz', 'BMW', 'Audi', 'Porsche',
    'Jaguar', 'Land Rover', 'Volvo', 'Lexus'
]

# ---------------------------------------------------------------------------
# Model loading – ML
# ---------------------------------------------------------------------------
with open("car_model.pkl", "rb") as f:
    ml_model = pickle.load(f)

# ---------------------------------------------------------------------------
# Model loading – Computer Vision
# ---------------------------------------------------------------------------
vit_classifier = pipeline("image-classification", model=VIT_MODEL_DIR)

CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"
clip_model     = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
clip_model.eval()
device     = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = clip_model.to(device)

_vit_cfg     = ViTForImageClassification.from_pretrained(VIT_MODEL_DIR).config
ID2LABEL     = _vit_cfg.id2label
LABEL_LIST   = [ID2LABEL[i] for i in range(len(ID2LABEL))]
TEXT_PROMPTS = [f"a photo of a {name}" for name in LABEL_LIST]

_text_inputs = clip_processor(
    text=TEXT_PROMPTS, return_tensors="pt", padding=True, truncation=True
).to(device)
with torch.no_grad():
    _text_out      = clip_model.get_text_features(**_text_inputs)
    _text_features = _text_out if isinstance(_text_out, torch.Tensor) else _text_out.pooler_output
    _text_features = _text_features / _text_features.norm(dim=-1, keepdim=True)

# ---------------------------------------------------------------------------
# OpenAI client
# ---------------------------------------------------------------------------
def _get_client() -> OpenAI:
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY ist nicht gesetzt.")
    return OpenAI(api_key=OPENAI_API_KEY)

def _call_llm_json(system_prompt: str, user_prompt: str) -> str:
    client = _get_client()
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
    )
    return response.choices[0].message.content

def _parse_json(raw: str, required_keys: tuple) -> dict:
    cleaned = (raw or "").strip()
    if not cleaned:
        raise ValueError("LLM hat eine leere Antwort zurückgegeben.")
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM hat kein gültiges JSON zurückgegeben: {cleaned[:300]}") from exc
    missing = [k for k in required_keys if k not in parsed]
    if missing:
        raise ValueError(f"Fehlende Schlüssel im LLM-JSON: {', '.join(missing)}")
    return parsed

# ---------------------------------------------------------------------------
# ML helpers
# ---------------------------------------------------------------------------
def predict_price(make: str, year: int, kilometer: int, engine_cc: int,
                  max_power_bhp: float, transmission: str, owner: str, fuel_type: str) -> float:
    car_age            = 2026 - year
    km_per_year        = kilometer / car_age if car_age > 0 else kilometer
    power_per_cc       = max_power_bhp / engine_cc if engine_cc > 0 else 0
    is_automatic       = 1 if transmission == "Automatic" else 0
    is_first_owner     = 1 if owner == "First" else 0
    is_luxury          = 1 if make in LUXURY_BRANDS else 0
    is_electric_hybrid = 1 if fuel_type in ["Electric", "Hybrid"] else 0

    features = np.array([[
        car_age, kilometer, engine_cc, max_power_bhp,
        is_automatic, is_first_owner, km_per_year, power_per_cc,
        is_luxury, is_electric_hybrid,
    ]])
    price_inr = float(ml_model.predict(features)[0])
    return round(price_inr * INR_TO_CHF, 2)

# ---------------------------------------------------------------------------
# NLP helpers (Text-Tab)
# ---------------------------------------------------------------------------
def extract_preferences(user_text: str) -> dict:
    system_prompt = (
        "Du bist ein Assistent, der Fahrzeugbeschreibungen analysiert. "
        "Extrahiere aus dem Text folgende Informationen als JSON:\n"
        "- make: Fahrzeugmarke (string, z.B. 'BMW', 'Toyota', 'Honda')\n"
        "- year: Baujahr (integer, z.B. 2018)\n"
        "- kilometer: Kilometerstand (integer, z.B. 50000)\n"
        "- engine_cc: Hubraum in Kubikzentimeter (integer, z.B. 1500)\n"
        "- max_power_bhp: Motorleistung in PS/bhp (float, z.B. 120.0)\n"
        "- transmission: Getriebe (string, entweder 'Manual' oder 'Automatic')\n"
        "- owner: Anzahl Vorbesitzer (string, eines von: 'First', 'Second', 'Third', 'Fourth', '4 or More')\n"
        "- fuel_type: Kraftstofftyp (string, eines von: 'Petrol', 'Diesel', 'CNG', 'LPG', 'Electric', 'Hybrid')\n"
        "Fehlende Werte mit sinnvollen Standardwerten ergänzen (year: 2018, kilometer: 50000, "
        "engine_cc: 1500, max_power_bhp: 100.0, transmission: 'Manual', owner: 'First', fuel_type: 'Petrol').\n"
        "Antworte NUR mit einem JSON-Objekt ohne zusätzlichen Text.\n"
        'Beispiel: {"make": "Toyota", "year": 2019, "kilometer": 45000, "engine_cc": 1998, '
        '"max_power_bhp": 148.0, "transmission": "Manual", "owner": "First", "fuel_type": "Diesel"}'
    )
    raw = _call_llm_json(system_prompt, user_text)
    return _parse_json(raw, ("make", "year", "kilometer", "engine_cc", "max_power_bhp",
                             "transmission", "owner", "fuel_type"))

def generate_explanation_text(preferences: dict, prediction: float) -> str:
    """Erklärung für den Text-Tab (nur Fahrzeugattribute bekannt)."""
    system_prompt = (
        "Du bist ein freundlicher Fahrzeugexperte. "
        "Erkläre die Preisschätzung auf Deutsch in 3-4 Sätzen. "
        "Gehe auf die wichtigsten Merkmale (Marke, Baujahr, Kilometerstand, Motorleistung) ein und füge "
        "einen kurzen Hinweis zur Unsicherheit der Schätzung hinzu. "
        "Berechne keinen eigenen Preis – verwende nur den angegebenen Schätzwert. "
        'Antworte NUR mit einem JSON-Objekt: {"answer": "..."}'
    )
    formatted_price = f"{prediction:.2f} CHF"
    user_prompt = (
        f"Fahrzeugdetails: {json.dumps(preferences, ensure_ascii=False)}\n"
        f"Geschätzter Fahrzeugpreis: {formatted_price}\n\n"
        f"Verwende in deiner Erklärung den Preis exakt in diesem Format: {formatted_price}. "
        "Bitte erkläre diese Schätzung auf Deutsch."
    )
    raw    = _call_llm_json(system_prompt, user_prompt)
    parsed = _parse_json(raw, ("answer",))
    return parsed["answer"]

def generate_explanation_cv(preferences: dict, prediction: float, cv_model_label: str) -> str:
    """
    Erklärung für den Bild-Tab.
    Nutzt zusätzlich das vom CV-Block erkannte spezifische Modell (z.B. 'BMW 3 Series 2012'),
    um eine genauere Begründung zu liefern.
    """
    system_prompt = (
        "Du bist ein freundlicher Fahrzeugexperte. "
        "Ein Computer-Vision-Modell hat das Fahrzeug auf einem Bild als folgendes Modell erkannt: "
        f"\"{cv_model_label}\". "
        "Erkläre die Preisschätzung auf Deutsch in 3-4 Sätzen. "
        "Nutze sowohl die strukturierten Fahrzeugdaten als auch das erkannte Modell, "
        "um die Schätzung zu begründen (z.B. typische Preislage dieses Modells, Alter, Laufleistung). "
        "Berechne keinen eigenen Preis – verwende nur den angegebenen Schätzwert. "
        "Füge einen kurzen Hinweis zur Unsicherheit hinzu. "
        'Antworte NUR mit einem JSON-Objekt: {"answer": "..."}'
    )
    formatted_price = f"{prediction:.2f} CHF"
    user_prompt = (
        f"Fahrzeugdetails: {json.dumps(preferences, ensure_ascii=False)}\n"
        f"Vom CV-Modell erkanntes Modell: {cv_model_label}\n"
        f"Geschätzter Fahrzeugpreis: {formatted_price}\n\n"
        f"Verwende in deiner Erklärung den Preis exakt in diesem Format: {formatted_price}. "
        "Bitte erkläre diese Schätzung auf Deutsch."
    )
    raw    = _call_llm_json(system_prompt, user_prompt)
    parsed = _parse_json(raw, ("answer",))
    return parsed["answer"]

# ---------------------------------------------------------------------------
# CV helpers (Bild-Tab)
# ---------------------------------------------------------------------------
def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def classify_with_vit(image_path: str) -> dict:
    results = vit_classifier(image_path, top_k=5)
    return {r["label"]: round(r["score"], 4) for r in results}

def classify_with_clip(image_path: str) -> dict:
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

def extract_make_from_label(label: str) -> str:
    """
    Extrahiert die Marke aus einem ViT/CLIP-Label wie 'BMW 3 Series 2012'.
    Vergleicht gegen bekannte Marken; fallback: erstes Wort.
    """
    known_makes = [
        'Acura', 'Aston Martin', 'Audi', 'Bentley', 'BMW', 'Bugatti',
        'Buick', 'Cadillac', 'Chevrolet', 'Chrysler', 'Dodge', 'Ferrari',
        'FIAT', 'Ford', 'GMC', 'Honda', 'Hyundai', 'Infiniti', 'Jaguar',
        'Jeep', 'Kia', 'Lamborghini', 'Land Rover', 'Lexus', 'Lincoln',
        'Maserati', 'Maybach', 'Mazda', 'McLaren', 'Mercedes-Benz', 'MINI',
        'Mitsubishi', 'Nissan', 'Porsche', 'Ram', 'Rolls-Royce', 'Scion',
        'smart', 'Spyker', 'Subaru', 'Suzuki', 'Tesla', 'Toyota',
        'Volkswagen', 'Volvo',
    ]
    for make in sorted(known_makes, key=len, reverse=True):
        if label.lower().startswith(make.lower()):
            return make
    return label.split()[0]

def get_cv_consensus(vit_preds: dict, clip_preds: dict) -> tuple[str, str]:
    """
    Bestimmt das beste Label per Score-Kombination aus ViT + CLIP.
    Gibt (bestes_label, methode) zurück.
    """
    combined = {}
    for label, score in vit_preds.items():
        combined[label] = combined.get(label, 0) + score
    for label, score in clip_preds.items():
        combined[label] = combined.get(label, 0) + score

    best_label  = max(combined, key=combined.get)
    vit_top1    = max(vit_preds,  key=vit_preds.get)
    clip_top1   = max(clip_preds, key=clip_preds.get)
    agree       = vit_top1 == clip_top1
    method      = "ViT + CLIP einig" if agree else "ViT + CLIP kombiniert (uneinig)"
    return best_label, method

def explain_cv_disagreement(image_path: str, vit_top1: str, clip_top1: str) -> str:
    """OpenAI Vision erklärt, warum ViT und CLIP sich unterscheiden."""
    if not OPENAI_API_KEY:
        return "OPENAI_API_KEY nicht gesetzt – keine Erklärung möglich."
    prompt = (
        f"Two computer vision models disagree on the make and model of this car:\n"
        f"  • Fine-tuned ViT predicted:  \"{vit_top1}\"\n"
        f"  • CLIP zero-shot predicted:  \"{clip_top1}\"\n\n"
        "Please look at the image carefully and:\n"
        "1. State which prediction you think is more likely correct and why.\n"
        "2. Explain what visual features may have caused the disagreement.\n"
        "3. Give a confidence assessment: high / medium / low.\n"
        "Answer in 3–5 concise sentences."
    )
    base64_image = encode_image(image_path)
    try:
        client   = _get_client()
        response = client.responses.create(
            model=OPENAI_MODEL,
            input=[{"role": "user", "content": [
                {"type": "input_text",  "text": prompt},
                {"type": "input_image", "image_url": f"data:image/jpeg;base64,{base64_image}"},
            ]}],
        )
        return response.output_text.strip()
    except Exception as e:
        return f"OpenAI Erklärung fehlgeschlagen: {e}"

# ---------------------------------------------------------------------------
# Pipeline – Text-Tab
# ---------------------------------------------------------------------------
def run_text_pipeline(user_text: str):
    if not user_text or not user_text.strip():
        return {}, 0.0, "Bitte gib eine Fahrzeugbeschreibung ein."
    try:
        preferences   = extract_preferences(user_text)
        make          = str(preferences.get("make", ""))
        year          = int(preferences.get("year", 2018))
        kilometer     = int(preferences.get("kilometer", 50000))
        engine_cc     = int(preferences.get("engine_cc", 1500))
        max_power_bhp = float(preferences.get("max_power_bhp", 100.0))
        transmission  = str(preferences.get("transmission", "Manual"))
        owner         = str(preferences.get("owner", "First"))
        fuel_type     = str(preferences.get("fuel_type", "Petrol"))

        if not make or year <= 0 or kilometer < 0:
            return preferences, 0.0, "Fehler: Marke, Baujahr und Kilometerstand müssen angegeben werden."

        prediction  = predict_price(make, year, kilometer, engine_cc, max_power_bhp, transmission, owner, fuel_type)
        explanation = generate_explanation_text(preferences, prediction)
        return preferences, prediction, explanation

    except ValueError as exc:
        return {}, 0.0, f"Fehler: {exc}"
    except Exception as exc:
        return {}, 0.0, f"Unerwarteter Fehler: {exc}"

# ---------------------------------------------------------------------------
# Pipeline – Bild-Tab
# ---------------------------------------------------------------------------
def run_image_pipeline(image_path: str, year: int, kilometer: int, engine_cc: int,
                       max_power_bhp: float, transmission: str, owner: str, fuel_type: str):
    """
    CV-Block erkennt Marke und Modell aus dem Bild.
    Marke → predict_price() (ML-Block)
    Erkanntes Modell + Preis → generate_explanation_cv() (NLP-Block)
    """
    if image_path is None:
        return {}, "", 0.0, "Bitte lade ein Fahrzeugbild hoch."
    try:
        # ── Schritt 1: CV – Fahrzeug erkennen ──────────────────────────────
        vit_preds  = classify_with_vit(image_path)
        clip_preds = classify_with_clip(image_path)
        best_label, cv_method = get_cv_consensus(vit_preds, clip_preds)

        vit_top1  = max(vit_preds,  key=vit_preds.get)
        clip_top1 = max(clip_preds, key=clip_preds.get)
        disagree_explanation = ""
        if vit_top1 != clip_top1:
            disagree_explanation = explain_cv_disagreement(image_path, vit_top1, clip_top1)

        # ── Schritt 2: Marke aus CV-Label extrahieren ───────────────────────
        detected_make = extract_make_from_label(best_label)

        cv_summary = {
            "Erkanntes Fahrzeug (bestes Label)": best_label,
            "Erkannte Marke (für Preisberechnung)": detected_make,
            "Methode": cv_method,
            "ViT Top-5":  vit_preds,
            "CLIP Top-5": clip_preds,
        }
        if disagree_explanation:
            cv_summary["Modelle uneinig – OpenAI Erklärung"] = disagree_explanation

        # ── Schritt 3: ML – Preis berechnen ────────────────────────────────
        preferences = {
            "make":          detected_make,
            "year":          year,
            "kilometer":     kilometer,
            "engine_cc":     engine_cc,
            "max_power_bhp": max_power_bhp,
            "transmission":  transmission,
            "owner":         owner,
            "fuel_type":     fuel_type,
        }
        prediction = predict_price(
            detected_make, year, kilometer, engine_cc,
            max_power_bhp, transmission, owner, fuel_type
        )

        # ── Schritt 4: NLP – Erklärung mit CV-Modellinfo ───────────────────
        explanation = generate_explanation_cv(preferences, prediction, best_label)

        return cv_summary, detected_make, prediction, explanation

    except ValueError as exc:
        return {}, "", 0.0, f"Fehler: {exc}"
    except Exception as exc:
        return {}, "", 0.0, f"Unerwarteter Fehler: {exc}"

# ---------------------------------------------------------------------------
# Pipeline – kombiniert (Text + Bild zusammen)
# ---------------------------------------------------------------------------
def run_combined_pipeline(user_text: str, image_path,
                          year: int, kilometer: int, engine_cc: int,
                          max_power_bhp: float, transmission: str,
                          owner: str, fuel_type: str):
    """
    Pflicht: Textbeschreibung + Fahrzeugfoto müssen beide vorhanden sein.

    Ablauf:
    1. NLP extrahiert Attribute aus dem Text (inkl. Marke als Fallback)
    2. CV erkennt Marke und Modell aus dem Bild → überschreibt die Textmarke
    3. ML berechnet den Preis mit der CV-Marke
    4. NLP generiert Erklärung mit CV-Modellinfo
    """
    if not user_text or not user_text.strip():
        return {}, {}, "", 0.0, "Bitte gib eine Fahrzeugbeschreibung ein."
    if image_path is None:
        return {}, {}, "", 0.0, "Bitte lade ein Fahrzeugfoto hoch."

    try:
        # ── Schritt 1: NLP – Text auswerten ────────────────────────────────
        preferences = extract_preferences(user_text)
        # Manuelle Felder aus der UI überschreiben NLP-Defaults wenn ausgefüllt
        preferences["year"]          = int(year)          if year          else preferences.get("year", 2018)
        preferences["kilometer"]     = int(kilometer)     if kilometer     else preferences.get("kilometer", 50000)
        preferences["engine_cc"]     = int(engine_cc)     if engine_cc     else preferences.get("engine_cc", 1500)
        preferences["max_power_bhp"] = float(max_power_bhp) if max_power_bhp else preferences.get("max_power_bhp", 100.0)
        preferences["transmission"]  = transmission  or preferences.get("transmission", "Manual")
        preferences["owner"]         = owner         or preferences.get("owner", "First")
        preferences["fuel_type"]     = fuel_type     or preferences.get("fuel_type", "Petrol")

        # ── Schritt 2: CV – Marke und Modell aus Bild erkennen ─────────────
        vit_preds  = classify_with_vit(image_path)
        clip_preds = classify_with_clip(image_path)
        best_label, cv_method = get_cv_consensus(vit_preds, clip_preds)

        vit_top1  = max(vit_preds,  key=vit_preds.get)
        clip_top1 = max(clip_preds, key=clip_preds.get)
        disagree_explanation = ""
        if vit_top1 != clip_top1:
            disagree_explanation = explain_cv_disagreement(image_path, vit_top1, clip_top1)

        # CV-Marke überschreibt die aus dem Text extrahierte Marke
        detected_make = extract_make_from_label(best_label)
        preferences["make"] = detected_make

        cv_summary = {
            "Erkanntes Fahrzeug (bestes Label)": best_label,
            "Erkannte Marke (für Preisberechnung)": detected_make,
            "Methode": cv_method,
            "ViT Top-5":  vit_preds,
            "CLIP Top-5": clip_preds,
        }
        if disagree_explanation:
            cv_summary["Modelle uneinig – OpenAI Erklärung"] = disagree_explanation

        # ── Schritt 3: ML – Preis berechnen ────────────────────────────────
        prediction = predict_price(
            detected_make,
            preferences["year"],
            preferences["kilometer"],
            preferences["engine_cc"],
            preferences["max_power_bhp"],
            preferences["transmission"],
            preferences["owner"],
            preferences["fuel_type"],
        )

        # ── Schritt 4: NLP – Erklärung mit CV-Modellinfo ───────────────────
        explanation = generate_explanation_cv(preferences, prediction, best_label)

        return preferences, cv_summary, detected_make, prediction, explanation

    except ValueError as exc:
        return {}, {}, "", 0.0, f"Fehler: {exc}"
    except Exception as exc:
        return {}, {}, "", 0.0, f"Unerwarteter Fehler: {exc}"


# ---------------------------------------------------------------------------
# Gradio UI – ein einziger Tab
# ---------------------------------------------------------------------------
with gr.Blocks(title="Auto-Preisrechner") as demo:
    gr.Markdown(
        "# Auto-Preisrechner\n"
        "Beschreibe das Fahrzeug auf Deutsch **und** lade ein Foto hoch. "
        "Das Computer-Vision-Modell erkennt die Marke automatisch aus dem Bild – "
        "alle weiteren Details werden aus deiner Beschreibung extrahiert."
    )

    with gr.Row():
        # ── Linke Spalte: Eingaben ──────────────────────────────────────────
        with gr.Column():
            text_input = gr.Textbox(
                label="Fahrzeugbeschreibung",
                lines=4,
                placeholder="z.B. BMW 320d, 2017, Automatik, 75000 km, Diesel, Zweitbesitzer, 2.0L, 190 PS",
            )
            img_input = gr.Image(type="filepath", label="Fahrzeugfoto (Pflicht)")
            submit_btn = gr.Button("Marke erkennen & Preis schätzen", variant="primary")

            gr.Examples(
                examples=[
                    ["Ich verkaufe meinen AM General Hummer SUV 2000 aus dem Jahr 2009, Automatik, ca. 375'000 km, Diesel, viertbesitzer, 2.0L Motor mit 120 PS.", "example_images/car1.jpg"],
                    ["Ich verkaufe meinen BMW 1 Series Convertible aus dem Jahr 2012, Automatik, ca. 75'000 km, Diesel, Zweitbesitzer, 2.0L Motor mit 280 PS..", "example_images/car2.jpg"],
                    ["Audi S6 Sedan 2011, Benziner, Automatik, Erstbesitzer, 30'000 km, 1.5L, 320 PS.", "example_images/car3.jpg"],
                ],
                inputs=[text_input, img_input],
            )

        # ── Rechte Spalte: Ausgaben ─────────────────────────────────────────
        with gr.Column():
            out_text_json  = gr.JSON(label="Extrahierte Fahrzeugdaten (aus Text)")
            out_cv_json    = gr.JSON(label="CV-Ergebnis (erkannte Marke & Modell)")
            out_make       = gr.Textbox(label="Erkannte Marke (verwendet für Preisberechnung)")
            out_price      = gr.Number(label="Geschätzter Fahrzeugpreis (CHF)")
            out_expl       = gr.Textbox(label="Erklärung", lines=6)

    submit_btn.click(
        fn=run_combined_pipeline,
        inputs=[
            text_input, img_input,
            gr.Number(value=0, visible=False),   # year     – aus Text extrahiert
            gr.Number(value=0, visible=False),   # kilometer
            gr.Number(value=0, visible=False),   # engine_cc
            gr.Number(value=0, visible=False),   # max_power_bhp
            gr.Textbox(value="", visible=False), # transmission
            gr.Textbox(value="", visible=False), # owner
            gr.Textbox(value="", visible=False), # fuel_type
        ],
        outputs=[out_text_json, out_cv_json, out_make, out_price, out_expl],
    )

demo.launch()
