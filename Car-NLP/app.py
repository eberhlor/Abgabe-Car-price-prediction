import json
import os
import pickle

import gradio as gr
import numpy as np
from openai import OpenAI

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
with open("car_model.pkl", "rb") as f:
    model = pickle.load(f)

# ---------------------------------------------------------------------------
# Wechselkurs INR → CHF
# ---------------------------------------------------------------------------
INR_TO_CHF = 0.0083

# ---------------------------------------------------------------------------
# Luxusmarken-Liste (identisch mit Training)
# ---------------------------------------------------------------------------
LUXURY_BRANDS = ['Mercedes-Benz', 'BMW', 'Audi', 'Porsche']

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _get_client() -> OpenAI:
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY ist nicht gesetzt. Bitte den Secret in Hugging Face konfigurieren.")
    return OpenAI(api_key=OPENAI_API_KEY)


def _call_llm_json(system_prompt: str, user_prompt: str) -> str:
    client = _get_client()
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
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
# Pipeline steps
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
    return _parse_json(raw, ("make", "year", "kilometer", "engine_cc", "max_power_bhp", "transmission", "owner", "fuel_type"))


def predict_price(make: str, year: int, kilometer: int, engine_cc: int,
                  max_power_bhp: float, transmission: str, owner: str, fuel_type: str) -> float:
    # Berechnete Features erstellen (identisch mit Training)
    car_age = 2026 - year
    km_per_year = kilometer / car_age if car_age > 0 else kilometer
    power_per_cc = max_power_bhp / engine_cc if engine_cc > 0 else 0
    is_automatic = 1 if transmission == "Automatic" else 0
    is_first_owner = 1 if owner == "First" else 0
    is_luxury = 1 if make in LUXURY_BRANDS else 0
    is_electric_hybrid = 1 if fuel_type in ["Electric", "Hybrid"] else 0

    # Feature-Vektor exakt wie im Training (10 Features)
    features = np.array([[
        car_age,
        kilometer,
        engine_cc,
        max_power_bhp,
        is_automatic,
        is_first_owner,
        km_per_year,
        power_per_cc,
        is_luxury,
        is_electric_hybrid,
    ]])

    price_inr = float(model.predict(features)[0])
    price_chf = round(price_inr * INR_TO_CHF, 2)
    return price_chf


def generate_explanation(preferences: dict, prediction: float) -> str:
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
    raw = _call_llm_json(system_prompt, user_prompt)
    parsed = _parse_json(raw, ("answer",))
    return parsed["answer"]


# ---------------------------------------------------------------------------
# End-to-end pipeline (called by Gradio)
# ---------------------------------------------------------------------------
def run_pipeline(user_text: str):
    if not user_text or not user_text.strip():
        return {}, 0.0, "Bitte gib eine Fahrzeugbeschreibung ein."

    try:
        preferences = extract_preferences(user_text)

        make = str(preferences.get("make", ""))
        year = int(preferences.get("year", 2018))
        kilometer = int(preferences.get("kilometer", 50000))
        engine_cc = int(preferences.get("engine_cc", 1500))
        max_power_bhp = float(preferences.get("max_power_bhp", 100.0))
        transmission = str(preferences.get("transmission", "Manual"))
        owner = str(preferences.get("owner", "First"))
        fuel_type = str(preferences.get("fuel_type", "Petrol"))

        if not make or year <= 0 or kilometer < 0:
            return preferences, 0.0, "Fehler: Marke, Baujahr und Kilometerstand müssen angegeben werden."

        prediction = predict_price(make, year, kilometer, engine_cc, max_power_bhp, transmission, owner, fuel_type)
        explanation = generate_explanation(preferences, prediction)

        return preferences, prediction, explanation

    except ValueError as exc:
        return {}, 0.0, f"Fehler: {exc}"
    except Exception as exc:
        return {}, 0.0, f"Unerwarteter Fehler: {exc}"


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
with gr.Blocks(title="Auto-Preisrechner") as demo:
    gr.Markdown(
        """
        # Auto-Preisrechner
        Beschreibe das Fahrzeug auf Deutsch. Der Assistent extrahiert die Details,
        schätzt den Preis mit einem trainierten Random-Forest-Modell und erklärt das Ergebnis.

        **Beispiel:** „Ich verkaufe meinen BMW 320d aus dem Jahr 2017, Automatik, ca. 75'000 km,
        Diesel, Zweitbesitzer, 2.0L Motor mit 190 PS."
        """
    )

    user_input = gr.Textbox(
        label="Fahrzeugbeschreibung",
        lines=4,
        placeholder="Beschreibe das Fahrzeug mit Marke, Baujahr, Kilometerstand, Motor, Getriebe, Kraftstoff und Vorbesitzer...",
    )
    submit_btn = gr.Button("Preis schätzen", variant="primary")

    extracted_json = gr.JSON(label="Extrahierte Eingaben")
    price_output = gr.Number(label="Geschätzter Fahrzeugpreis (CHF)")
    explanation_output = gr.Textbox(label="Erklärung", lines=6)

    gr.Examples(
        examples=[
            ["Ich verkaufe meinen BMW 320d aus dem Jahr 2017, Automatik, ca. 75'000 km, Diesel, Zweitbesitzer, 2.0L Motor mit 190 PS."],
            ["Toyota Fortuner 2019, Diesel, Manual, Erstbesitzer, 69'000 km, 2.4L, 148 PS."],
            ["Honda City 2020, Benziner, Automatik, Erstbesitzer, 30'000 km, 1.5L, 119 PS."],
            ["Audi A4 2016, Diesel, Automatik, Drittbesitzer, 120'000 km, 2.0L, 150 PS."],
        ],
        inputs=user_input,
    )

    submit_btn.click(
        fn=run_pipeline,
        inputs=[user_input],
        outputs=[extracted_json, price_output, explanation_output],
    )

demo.launch()
