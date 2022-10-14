# Map weird-looking feature names to more human-friendly names
from re import M


pretty_feature_names = {
    "Foley": "Urine Output (Foley)",
    "Void": "Urine Output (Void)",
    "Ventilator Tank #1": "Ventilator Tank #1 Volume",
    "Ventilator Tank #2": "Ventilator Tank #2 Volume",
    "Vti High": "Maximum Inhaled Tidal Volume",
    "Glucose finger stick (range 70-100)": "Glucose (finger-stick)",
    "Tidal Volume (set)": "Preset Tidal Volume",
    "Tidal Volume (observed)": "Delivered Tidal Volume",
    "Tidal Volume (spontaneous)": "Spontaneous Tidal Volume",
    "Temperature Fahrenheit": "Body Temperature",
    "Height (cm)": "Height",
    "Blood Pressure Systolic": "Systolic Blood Pressure",
    "Blood Pressure Diastolic": "Diastolic Blood Pressure",
    "Non Invasive Blood Pressure systolic": "Systolic Blood Pressure",
    "Non Invasive Blood Pressure diastolic": "Diastolic Blood Pressure",
    "Non Invasive Blood Pressure mean": "Mean Blood Pressure",
    "O2 saturation pulseoxymetry": "Pulse Oximetry",
    "O2 Saturation Pulseoxymetry Alarm - High": "Pulse Oximetry Alarm - High",
}
