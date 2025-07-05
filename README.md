# 🌍 Advanced Climate Risk & Hazard Modelling Tool

This Streamlit app provides:
- Multi-asset, multi-hazard, ensemble climate projections
- Uncertainty quantification
- Asset mapping and vulnerability assignment
- Impact assessment (productivity loss, logistics, asset value, health, equipment)
- Actionable insights for decision-makers

## Features

- **Granular timeframes**: Analyze any period 2015-2100
- **Uncertainty quantification**: Ensemble mean and percentiles from multiple CMIP6 models
- **Asset mapping**: Upload asset CSV or select sample asset; interactive map (with folium)
- **Impact assessment**: Calculate productivity loss, logistics, health, asset value impacts
- **Vulnerability**: Assign vulnerability scores per asset/location
- **Actionable insights**: Automated recommendations based on results
- **Downloadable results**: Export all results as CSV

## Getting Started

### Deploy on Streamlit Community Cloud

1. Upload `app.py` and `requirements.txt` to a public GitHub repo.
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud) and link your repo.

### Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

### Asset Data Format

Upload a CSV with columns: `name,lat,lon,vulnerability`  
Example:
```csv
name,lat,lon,vulnerability
Factory A,28.6,77.2,0.7
Warehouse B,19.1,72.8,0.5
```

### Optional: Interactive Map

Install `streamlit-folium` and `folium` for asset visualization.

## Notes

- Data is fetched live from the [Pangeo CMIP6 cloud catalog](https://pangeo-data.github.io/pangeo-cmip6/).
- First run may take a few minutes.
- For best results, use well-known coordinates and major cities.

---
