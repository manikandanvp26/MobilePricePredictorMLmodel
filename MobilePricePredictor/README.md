# Mobile Price Predictor AI

Dataset-driven Flask application using the supplied 836-row Mobile Price Prediction dataset. The project keeps the lime + cherry-red visual theme and centred top navigation.

## Features
- Brand and model choices populated directly from the dataset
- Price prediction with a 90th-percentile held-out residual range
- Ridge, Random Forest, Gradient Boosting and Extra Trees evaluation
- SHAP explanations (requires SHAP installation)
- What-if specification simulator and scenario-based depreciation forecast
- Smart recommendations, phone comparison and price fairness estimate
- Analytics dashboard, batch CSV/Excel upload, price curve
- PDF report, SQLite wishlist and in-app target alerts

## Important limitations
The supplied data contains 836 listings and no dated historical price series. Future price is therefore an assumption-based depreciation scenario, not a learned time-series forecast. In-app alerts compare the model estimate to a saved target; they do not monitor live retailer prices. Fairness compares the estimate with nearest records in the dataset and is a heuristic, not a regulated valuation.

## Windows / VS Code setup
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python train.py
python app.py
```
Open http://127.0.0.1:5000. The app also trains a model on first launch if no model file exists.

## Dataset
`data/mobile_phones.csv` is derived from the uploaded `Mobile Price Prediction Datatset.csv`. The `Brand me` text was split into `brand` and `model`; source numeric columns were mapped to readable feature names. Rows and original listing names are retained.

## Deployment
Use Python 3.11 or 3.12 for broad ML package compatibility. Render build command: `pip install -r requirements.txt`; start command: `gunicorn app:app` (add `gunicorn>=22` to requirements if deploying this way). For local development use `python app.py`. Persist `data/user_features.sqlite3` with a persistent disk if wishlist and alerts must survive redeploys.
