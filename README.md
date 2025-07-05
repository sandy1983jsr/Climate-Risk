# Climate Risk & Hazard Modelling Tool

This Streamlit app provides climate hazard projections (heatwave, drought, precipitation extreme) for any latitude/longitude using open CMIP6 model data.

## Features

- Input any latitude and longitude
- Select hazards (heatwave, drought, precipitation extreme)
- Choose IPCC scenario (SSP1-2.6, SSP2-4.5, SSP5-8.5)
- View annual projections (2021-2050) for selected hazards

## Getting Started

### Deploy on Streamlit Community Cloud

1. Upload `app.py` and `requirements.txt` to a public GitHub repo.
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud) and link your repo.

### Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
