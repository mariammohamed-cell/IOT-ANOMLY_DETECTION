[README.md](https://github.com/user-attachments/files/34971667/README.md)[Uploading READ# Thermostat device — TON_IoT attack detection

Part of the multi-source TON_IoT pipeline. Same method, 10-second windows, hash-block split and export format as the Weather device.

## Files
- `Thermostat_IoT_Detection_final.ipynb` — clean notebook (run it to regenerate everything)
- `Thermostat_IoT_Detection_final_with_outputs.ipynb` — same notebook with outputs and plots
- `validate_export.py` — 35 sanity checks on the exported files
- `thermostat_export/` — `thermostat_events.csv`, `thermostat_predictions.csv`, `thermostat_config.json`, `thermostat_model.joblib`, `results/`

## Data
`IoT_Thermostat.csv` (TON_IoT, 442,228 rows) is not stored in this repo. Put it next to the notebook and set `DATA_PATH`.

## Run
    pip install numpy pandas scikit-learn xgboost lightgbm catboost matplotlib seaborn joblib
    DATA_PATH=IoT_Thermostat.csv python -m jupyter nbconvert --to notebook --execute Thermostat_IoT_Detection_final.ipynb
    python validate_export.py thermostat_export thermostat

## Result (block split, calibrated LightGBM, threshold 0.40)
Val F1 0.848 | Test F1 0.875 | 35/35 export checks pass.
Rare types (scanning, xss, injection) have almost no windows in val/test, so multi-class macro-F1 is not reliable.

The `.joblib` model was saved with scikit-learn 1.9.1, lightgbm 4.7.0, xgboost 3.4.1, catboost 1.2.10; load it with matching versions.
ME.md…]()
