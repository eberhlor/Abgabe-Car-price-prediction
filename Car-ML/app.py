import gradio as gr
import pickle
import numpy as np
import pandas as pd
import re
from sklearn.ensemble import RandomForestRegressor

# ---------------------------------------------------------------------------
# Modell laden
# ---------------------------------------------------------------------------
model_filename = "car_model.pkl"
with open(model_filename, "rb") as f:
    random_forest_model = pickle.load(f)

# ---------------------------------------------------------------------------
# Wechselkurs INR → CHF (Stand Juni 2025)
# ---------------------------------------------------------------------------
INR_TO_CHF = 0.01075

# ---------------------------------------------------------------------------
# Luxusmarken-Liste (identisch mit Training)
# ---------------------------------------------------------------------------
luxury_brands = ['Mercedes-Benz', 'BMW', 'Audi', 'Porsche', 'Land Rover', 'Jaguar', 'Volvo', 'MINI']

all_makes = sorted(luxury_brands + [
    'Honda', 'Hyundai', 'Maruti Suzuki', 'Toyota', 'Tata', 'Mahindra',
    'Ford', 'Volkswagen', 'Skoda', 'Renault', 'Nissan', 'Kia',
    'Jeep', 'Chevrolet', 'Fiat', 'Other'
])

# ---------------------------------------------------------------------------
# Vorhersagefunktion
# ---------------------------------------------------------------------------
def predict_car(make, year, kilometer, engine_cc, max_power_bhp, transmission, owner, fuel_type):
    # Berechnete Features erstellen
    car_age = 2024 - year
    km_per_year = kilometer / car_age if car_age > 0 else kilometer
    power_per_cc = max_power_bhp / engine_cc if engine_cc > 0 else 0
    is_automatic = 1 if transmission == 'Automatic' else 0
    is_first_owner = 1 if owner == 'First' else 0
    is_luxury = 1 if make in luxury_brands else 0
    is_electric_hybrid = 1 if fuel_type in ['Electric', 'Hybrid'] else 0

    # Feature-Vektor aufbauen (10 Features, Reihenfolge wie im Training)
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

    prediction = random_forest_model.predict(features)
    price_inr = float(prediction[0])
    price_chf = round(price_inr * INR_TO_CHF, 2)
    return price_chf

# ---------------------------------------------------------------------------
# Gradio Interface
# ---------------------------------------------------------------------------
demo = gr.Interface(
    fn=predict_car,
    inputs=[
        gr.Dropdown(choices=all_makes, label="Marke (Make)"),
        gr.Number(label="Baujahr (Year)", value=2018),
        gr.Number(label="Kilometerstand (Kilometer)", value=50000),
        gr.Number(label="Hubraum in cc (Engine cc)", value=1500),
        gr.Number(label="Max. Leistung in bhp (Max Power bhp)", value=120),
        gr.Dropdown(choices=['Manual', 'Automatic'], label="Getriebe (Transmission)"),
        gr.Dropdown(choices=['First', 'Second', 'Third', 'Fourth', '4 or More'], label="Vorbesitzer (Owner)"),
        gr.Dropdown(choices=['Petrol', 'Diesel', 'CNG', 'LPG', 'Electric', 'Hybrid'], label="Kraftstoff (Fuel Type)"),
    ],
    outputs=[gr.Number(label="Geschätzter Fahrzeugpreis (CHF)")],
    examples=[
        ['Toyota', 2018, 69000, 2393, 148, 'Manual', 'First', 'Diesel'],
        ['BMW', 2017, 75000, 1995, 188, 'Automatic', 'Second', 'Diesel'],
    ],
    title="Auto-Preisrechner",
    description="Geben Sie die Details des Fahrzeugs ein, um eine Preisschätzung zu erhalten.",
)

demo.launch()
