import streamlit as st
import numpy as np
import pandas as pd
import xarray as xr
import intake
import matplotlib.pyplot as plt

# Pangeo CMIP6 catalog URL
CATALOG_URL = "https://storage.googleapis.com/cmip6/pangeo-cmip6.json"

# Hazard definitions
HAZARD_OPTIONS = {
    "Heatwave": {"variable": "tasmax", "description": "Days with max temperature > 35°C"},
    "Drought": {"variable": "pr", "description": "Months with precipitation < 50mm"},
    "Precipitation Extreme": {"variable": "pr", "description": "Days with rainfall > 30mm"}
}

# IPCC scenarios
SCENARIOS = {
    "SSP1-2.6 (Low)": "ssp126",
    "SSP2-4.5 (Intermediate)": "ssp245",
    "SSP5-8.5 (High)": "ssp585"
}

YEARS = (2021, 2050)

@st.cache_data(show_spinner="Loading CMIP6 catalog...")
def get_cmip6_catalog():
    return intake.open_esm_datastore(CATALOG_URL)

def get_nearest_grid(ds, lat, lon):
    """Find nearest point in gridded dataset."""
    abs_lat = np.abs(ds['lat'] - lat)
    abs_lon = np.abs(ds['lon'] - lon)
    min_lat = abs_lat.argmin().item()
    min_lon = abs_lon.argmin().item()
    return min_lat, min_lon

@st.cache_data(show_spinner="Loading climate data...", suppress_st_warning=True)
def load_cmip6_data(variable, scenario, lat, lon, years=YEARS):
    """Load CMIP6 data for a variable, scenario, and coordinates from the catalog."""
    cat = get_cmip6_catalog()
    df = cat.df
    # Find matching datasets
    sel = (
        (df['variable_id'] == variable) &
        (df['experiment_id'] == scenario) &
        (df['table_id'].str.contains("day|Amon"))  # daily or monthly
    )
    subset = df[sel]
    if subset.empty:
        return None
    ds_url = subset.iloc[0].zstore
    ds = xr.open_zarr(ds_url, consolidated=True)
    # Time slice
    ds = ds.sel(time=slice(f"{years[0]}-01-01", f"{years[1]}-12-31"))
    # Nearest grid point
    lat_idx, lon_idx = get_nearest_grid(ds, lat, lon)
    point_data = ds.isel(lat=lat_idx, lon=lon_idx)
    return point_data

def calculate_hazard(hazard, data):
    if hazard == "Heatwave":
        # Days with max temp > 35C
        tasmax = data["tasmax"] - 273.15  # convert K to C
        hot_days = (tasmax > 35).groupby("time.year").sum()
        return hot_days
    elif hazard == "Drought":
        # Months with pr < 50mm (pr is in kg/m2/s, convert to mm/month)
        pr = data["pr"] * 86400 * 30  # rough monthly sum
        pr_monthly = pr.resample(time="1M").sum()
        dry_months = (pr_monthly < 50).groupby("time.year").sum()
        return dry_months
    elif hazard == "Precipitation Extreme":
        # Days with pr > 30mm (pr is in kg/m2/s, convert to mm/day)
        pr = data["pr"] * 86400  # 1kg/m2/s = 1mm/day
        wet_days = (pr > 30).groupby("time.year").sum()
        return wet_days
    return None

def plot_projection(years, values, hazard_name):
    fig, ax = plt.subplots()
    ax.plot(years, values, marker="o")
    ax.set_xlabel("Year")
    ax.set_ylabel("Annual Frequency")
    ax.set_title(f"{hazard_name} Projection")
    ax.grid(True)
    st.pyplot(fig)

st.set_page_config(
    page_title="Climate Risk & Hazard Modelling Tool",
    page_icon="🌍",
    layout="centered"
)
st.title("🌍 Climate Risk & Hazard Modelling Tool")
st.write("""
This tool provides climate hazard projections for any latitude/longitude using CMIP6 model data. 
Select your location, IPCC scenario, and hazards to see risk projections for 2021-2050.
""")

with st.form("input_form"):
    lat = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=28.6, format="%.4f")
    lon = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=77.2, format="%.4f")
    scenario = st.selectbox("IPCC Scenario", options=list(SCENARIOS.keys()))
    hazards = st.multiselect(
        "Select Hazards", 
        options=list(HAZARD_OPTIONS.keys()), 
        default=["Heatwave", "Precipitation Extreme"]
    )
    submit = st.form_submit_button("Run Analysis")

if submit:
    scenario_code = SCENARIOS[scenario]
    for hazard in hazards:
        st.subheader(f"{hazard}: {HAZARD_OPTIONS[hazard]['description']}")
        variable = HAZARD_OPTIONS[hazard]["variable"]
        st.info(f"Fetching {variable} data for {scenario}...")
        data = load_cmip6_data(variable, scenario_code, lat, lon)
        if data is None:
            st.warning(f"No data found for {hazard} at this location/scenario.")
            continue
        st.success("Data loaded. Calculating hazard projection...")
        hz = calculate_hazard(hazard, data)
        if hz is not None:
            plot_projection(hz["year"].values, hz.values, hazard)
            st.dataframe(pd.DataFrame({"Year": hz["year"].values, "Frequency": hz.values}))
        else:
            st.warning("Could not calculate hazard for this selection.")

st.markdown("---")
st.caption("Powered by [CMIP6 Pangeo Data](https://pangeo-data.github.io/pangeo-cmip6/), Streamlit, and Open Source Libraries. | Copilot 2025")
