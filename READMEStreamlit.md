# Track Prediction App

This Streamlit app loads `final_proj.pkl`, accepts the model's 18 feature values, and shows the predicted class plus the full class-probability distribution.

## Install

From the project directory:

```powershell
python -m pip install -r requirements.txt
```

## Run

Keep `final_proj.pkl` next to `app.py`, then run:

```powershell
streamlit run app.py
```

The supplied artifact is loaded with `joblib.load`, with a standard pickle fallback. The app validates that it is a scikit-learn pipeline containing a `model` step with fitted feature and class metadata.

## Edit feature_config.json

Replace the 18 placeholder entries with your feature labels, help text, defaults, bounds, and categorical options. Each entry requires `name`, `type`, `default`, `min`, `max`, and `help`. Numeric entries use `min`, `max`, and optional `step`; categorical entries also require a non-empty `options` list.

The current model exposes `feature_names_in_` as `Q1` through `Q18`, so those names and their order are used automatically. The configuration must still contain exactly 18 entries. The first three placeholder controls use numeric categorical options because this model expects the encoded values used during training.
