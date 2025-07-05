import streamlit as st
import numpy as np
import pandas as pd
import xarray as xr
import intake
import matplotlib.pyplot as plt

try:
    import folium
    from streamlit_folium import st_folium
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False

CATALOG_URL = "https://storage.googleapis.com/cmip6/pangeo-cmip6.json"

HAZARD_OPTIONS = {
    "Heatwave": {
        "variable": "tasmax",
        "description": "Max temp > 35°C",
        "logic": lambda data: ((data["tasmax"] - 273.15) > 35).groupby("time.year").sum()
    },
    "Extreme Cold": {
        "variable": "tasmin",
        "description": "Min temp < -5°C",
        "logic": lambda data: ((data["tasmin"] - 273.15) < -5).groupby("time.year").sum()
    },
    "Drought": {
        "variable": "pr",
        "description": "Monthly pr < 50mm",
        "logic": lambda data: (
            (data["pr"] * 86400 * 30).resample(time="1M").sum() < 50
        ).groupby("time.year").sum()
    },
    "Water Stress": {
        "variable": "pr",
        "description": "Annual precipitation < 500mm",
        "logic": lambda data: (
            (data["pr"] * 86400).resample(time="1Y").sum() < 500
        ).groupby("time.year").sum()
    },
    "Precipitation Extreme": {
        "variable": "pr",
        "description": "Rain > 30mm/day",
        "logic": lambda data: ((data["pr"] * 86400) > 30).groupby("time.year").sum()
    },
    "Flooding": {
        "variable": "pr",
        "description": "Days with rain > 50mm/day (proxy for pluvial flooding)",
        "logic": lambda data: ((data["pr"] * 86400) > 50).groupby("time.year").sum()
    },
    "Tropical Cyclones": {
        "variable": "sfcWind",
        "description": "Days with wind speed > 25 m/s (proxy for cyclonic storms)",
        "logic": lambda data: ((data["sfcWind"]) > 25).groupby("time.year").sum()
    },
    "Wildfires": {
        "variable": "tasmax_pr",
        "description": "Days with temp > 32°C and precipitation < 1mm (proxy for fire weather)",
        "logic": lambda data: (
            ((data["tasmax"] - 273.15) > 32) & ((data["pr"] * 86400) < 1)
        ).groupby("time.year").sum()
    },
    "Ice Melt": {
        "variable": "tasmax",
        "description": "Days with temp > 0°C (proxy for melt conditions)",
        "logic": lambda data: ((data["tasmax"] - 273.15) > 0).groupby("time.year").sum()
    },
    "Sea Level Risk": {
        "variable": "zos",
        "description": "Annual mean sea surface height anomaly > 0.2m",
        "logic": lambda data: (data["zos"].resample(time="1Y").mean() > 0.2).groupby("time.year").sum()
    },
}

SCENARIOS = {
    "SSP1-2.6 (Low)": "ssp126",
    "SSP2-4.5 (Intermediate)": "ssp245",
    "SSP5-8.5 (High)": "ssp585",
}

@st.cache_data
def get_cmip6_catalog():
    return intake.open_esm_datastore(CATALOG_URL)

def get_nearest_grid(ds, lat, lon):
    abs_lat = np.abs(ds['lat'] - lat)
    abs_lon = np.abs(ds['lon'] - lon)
    min_lat = abs_lat.argmin().item()
    min_lon = abs_lon.argmin().item()
    return min_lat, min_lon

def load_cmip6_ensemble(hazard, scenario, lat, lon, years, n_models=3):
    cat = get_cmip6_catalog()
    df = cat.df
    hazard_info = HAZARD_OPTIONS[hazard]
    if hazard == "Wildfires":
        tasmax_sel = (
            (df['variable_id'] == "tasmax")
            & (df['experiment_id'] == scenario)
            & (df['table_id'].str.contains("day|Amon"))
        )
        pr_sel = (
            (df['variable_id'] == "pr")
            & (df['experiment_id'] == scenario)
            & (df['table_id'].str.contains("day|Amon"))
        )
        tasmax_df = df[tasmax_sel].reset_index()
        pr_df = df[pr_sel].reset_index()
        merged = pd.merge(
            tasmax_df, pr_df, on=["source_id", "member_id"], suffixes=('_tasmax', '_pr')
        )
        results = []
        for _, row in merged.head(n_models).iterrows():
            try:
                ds_tasmax = xr.open_zarr(row["zstore_tasmax"], consolidated=True)
                ds_pr = xr.open_zarr(row["zstore_pr"], consolidated=True)
                ds_tasmax = ds_tasmax.sel(time=slice(f"{years[0]}-01-01", f"{years[1]}-12-31"))
                ds_pr = ds_pr.sel(time=slice(f"{years[0]}-01-01", f"{years[1]}-12-31"))
                lat_idx, lon_idx = get_nearest_grid(ds_tasmax, lat, lon)
                data = {
                    "tasmax": ds_tasmax.isel(lat=lat_idx, lon=lon_idx)["tasmax"],
                    "pr": ds_pr.isel(lat=lat_idx, lon=lon_idx)["pr"]
                }
                results.append(data)
            except Exception:
                continue
        return results
    else:
        variable = hazard_info["variable"]
        sel = (
            (df['variable_id'] == variable)
            & (df['experiment_id'] == scenario)
            & (df['table_id'].str.contains("day|Amon"))
        )
        subset = df[sel].head(n_models)
        results = []
        for _, row in subset.iterrows():
            try:
                ds = xr.open_zarr(row.zstore, consolidated=True)
                ds = ds.sel(time=slice(f"{years[0]}-01-01", f"{years[1]}-12-31"))
                lat_idx, lon_idx = get_nearest_grid(ds, lat, lon)
                data = {variable: ds.isel(lat=lat_idx, lon=lon_idx)[variable]}
                results.append(data)
            except Exception:
                continue
        return results

def calculate_hazard_ensemble(hazard, data_list):
    logic = HAZARD_OPTIONS[hazard]["logic"]
    series_list = []
    for data in data_list:
        try:
            if isinstance(data, dict):
                ds = xr.Dataset(data)
                series = logic(ds)
            else:
                series = logic(data)
            series_list.append(series)
        except Exception:
            continue
    if not series_list:
        return pd.DataFrame()
    df = pd.DataFrame({i: s.values for i, s in enumerate(series_list)}, index=series_list[0]["year"].values)
    return df

def plot_ensemble(years, df, hazard_name):
    plt.figure(figsize=(7,3))
    if df.empty:
        st.warning("No data available for this hazard at this location/scenario.")
        return
    mean = df.mean(axis=1)
    perc10 = df.quantile(0.1, axis=1)
    perc90 = df.quantile(0.9, axis=1)
    plt.plot(years, mean, label="Mean", color="navy")
    plt.fill_between(years, perc10, perc90, color="skyblue", alpha=0.3, label="10th-90th percentile")
    plt.xlabel("Year")
    plt.ylabel("Frequency")
    plt.title(f"{hazard_name} Projection (Ensemble Mean, 10th-90th Perc.)")
    plt.legend()
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.close()

def asset_map(assets_df):
    if not FOLIUM_AVAILABLE:
        st.info("Install streamlit-folium to use asset mapping.")
        return
    m = folium.Map(location=[assets_df["lat"].mean(), assets_df["lon"].mean()], zoom_start=3)
    for i, row in assets_df.iterrows():
        folium.Marker(
            location=[row["lat"], row["lon"]],
            popup=f'{row["name"]}: Vulnerability {row["vulnerability"]:.2f}',
            tooltip=row["name"],
            icon=folium.Icon(color="red" if row["vulnerability"] > 0.7 else "orange" if row["vulnerability"] > 0.4 else "green"),
        ).add_to(m)
    st_folium(m, width=700, height=400)

def actionable_insights(hazard, mean_value, asset, thresholds):
    insights = []
    if hazard == "Heatwave":
        if mean_value > thresholds["heat_risk_days"]:
            insights.append("High frequency of extreme heat days: Adjust work hours, provide cooling/rest areas, increase hydration breaks.")
        elif mean_value > thresholds["heat_monitor_days"]:
            insights.append("Rising trend in heat days: Monitor employee health, review PPE, and plan for possible adaptation.")
    if hazard == "Precipitation Extreme":
        if mean_value > thresholds["logistics_days"]:
            insights.append("Frequent heavy rain impacts logistics: Diversify supply routes, implement drainage, review transport policies.")
    if hazard == "Drought":
        if mean_value > thresholds["drought_months"]:
            insights.append("Increasing drought: Consider water conservation, alternative sourcing, or crop adaptation.")
    if hazard == "Water Stress":
        if mean_value > thresholds["water_stress_years"]:
            insights.append("Significant water stress years: Explore alternative water sources and efficiency measures.")
    if hazard == "Flooding":
        if mean_value > thresholds["flood_days"]:
            insights.append("High risk of pluvial flooding: Consider site drainage improvements and flood insurance.")
    if hazard == "Tropical Cyclones":
        if mean_value > thresholds["cyclone_days"]:
            insights.append("Cyclone risk: Review asset fortification, emergency planning, and insurance.")
    if hazard == "Wildfires":
        if mean_value > thresholds["wildfire_days"]:
            insights.append("Wildfire-prone: Maintain fire breaks, review emergency response, and ensure insurance coverage.")
    if hazard == "Ice Melt":
        if mean_value > thresholds["melt_days"]:
            insights.append("Frequent thaw: Assess infrastructure/foundation adaptation needs.")
    if hazard == "Sea Level Risk":
        if mean_value > thresholds["sealvl_years"]:
            insights.append("Rising sea level: Prepare for flood defenses or relocation.")
    if asset["vulnerability"] > 0.7 and mean_value > 0:
        insights.append("High vulnerability asset at risk: Consider insurance, asset diversification, or relocation.")
    if not insights:
        insights = ["No major risks identified; continue monitoring and reassess annually."]
    return insights

st.set_page_config(
    page_title="Climate Risk & Hazard Modelling Tool",
    page_icon="🌍",
    layout="centered"
)
st.title("🌍 Advanced Climate Risk & Hazard Modelling Tool")
st.write("Multi-asset, multi-hazard, ensemble projections, uncertainty quantification, asset mapping, vulnerability, actionable insights, and financial impact assessment.")

col1, col2 = st.columns(2)
with col1:
    lat = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=28.6, format="%.4f")
    lon = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=77.2, format="%.4f")
with col2:
    start_year = st.number_input("Start Year", min_value=2015, max_value=2100, value=2021)
    end_year = st.number_input("End Year", min_value=start_year, max_value=2100, value=2050)
scenario = st.selectbox("IPCC Scenario", options=list(SCENARIOS.keys()))
hazards = st.multiselect(
    "Select Hazards",
    options=list(HAZARD_OPTIONS.keys()),
    default=["Heatwave", "Flooding", "Drought", "Extreme Cold", "Precipitation Extreme", "Tropical Cyclones", "Wildfires", "Water Stress", "Ice Melt", "Sea Level Risk"]
)

# Asset mapping/upload
st.subheader("Asset Mapping")
uploaded = st.file_uploader("Upload asset CSV (name,lat,lon,vulnerability)", type=["csv"])
if uploaded:
    assets_df = pd.read_csv(uploaded)
    st.success("Assets uploaded:")
    st.dataframe(assets_df)
else:
    assets_df = pd.DataFrame([{"name": "Sample Asset", "lat": lat, "lon": lon, "vulnerability": 0.5}])
    st.info("No asset file uploaded. Using sample asset.")

if FOLIUM_AVAILABLE:
    asset_map(assets_df)

if not uploaded:
    st.subheader("Vulnerability Mapping")
    vuln = st.slider("Select vulnerability for sample asset", 0.0, 1.0, 0.5, step=0.05)
    assets_df["vulnerability"] = vuln

# Impact parameters and thresholds
st.subheader("Impact Assessment Parameters")
prod_loss = st.number_input("Productivity loss per heatwave day (%)", value=2)
health_risk_days = st.number_input("Heat stress risk threshold (annual days)", value=15)
health_monitor_days = st.number_input("Heat monitoring threshold (annual days)", value=8)
equip_temp_limit = st.number_input("Equipment max operating temp (°C)", value=40)
logistics_days = st.number_input("Heavy rain logistics impact threshold (days/year)", value=10)
drought_months = st.number_input("Drought impact threshold (months/year)", value=4)
water_stress_years = st.number_input("Water stress impact threshold (years/period)", value=2)
flood_days = st.number_input("Flooding days threshold (days/year)", value=3)
cyclone_days = st.number_input("Cyclone risk days threshold (days/year)", value=1)
wildfire_days = st.number_input("Wildfire risk days threshold (days/year)", value=5)
melt_days = st.number_input("Ice melt days threshold (days/year)", value=10)
sealvl_years = st.number_input("Sea level risk threshold (years/period)", value=2)

thresholds = {
    "heat_risk_days": health_risk_days,
    "heat_monitor_days": health_monitor_days,
    "equip_temp_limit": equip_temp_limit,
    "logistics_days": logistics_days,
    "drought_months": drought_months,
    "water_stress_years": water_stress_years,
    "flood_days": flood_days,
    "cyclone_days": cyclone_days,
    "wildfire_days": wildfire_days,
    "melt_days": melt_days,
    "sealvl_years": sealvl_years,
}

st.subheader("Financial Impact Parameters (INR million per event)")
cost_params = {}
for hazard in hazards:
    cost_params[hazard] = st.number_input(
        f"Financial loss per {hazard} event (INR million)",
        min_value=0.0, value=10.0, key=f"cost_{hazard}"
    )

if st.button("Run Multi-Asset Analysis"):
    scenario_code = SCENARIOS[scenario]
    output_tables = []
    financial_loss_tables = []
    for idx, asset in assets_df.iterrows():
        st.markdown(f"### Asset: **{asset['name']}**  (Vulnerability: {asset['vulnerability']:.2f})")
        asset_results = []
        for hazard in hazards:
            st.markdown(f"**{hazard}: {HAZARD_OPTIONS[hazard]['description']}**")
            st.info("Loading ensemble data...")
            ens = load_cmip6_ensemble(
                hazard,
                scenario_code,
                asset["lat"],
                asset["lon"],
                (start_year, end_year),
                n_models=3,
            )
            if not ens:
                st.error("No data found for this asset/hazard.")
                continue
            hz_df = calculate_hazard_ensemble(hazard, ens)
            plot_ensemble(hz_df.index, hz_df, hazard)
            mean_hazard = hz_df.mean(axis=1)
            # Impact assessment
            if hazard == "Heatwave":
                annual_loss = mean_hazard * prod_loss * asset["vulnerability"]
                st.write("**Estimated annual productivity loss (%):**")
                st.dataframe(pd.DataFrame({"Year": hz_df.index, "Loss %": annual_loss}))
                asset_results.append(pd.DataFrame({
                    "Year": hz_df.index,
                    "Hazard": [hazard]*len(hz_df.index),
                    "Frequency": mean_hazard,
                    "Productivity Loss %": annual_loss,
                }))
            else:
                asset_results.append(pd.DataFrame({
                    "Year": hz_df.index,
                    "Hazard": [hazard]*len(hz_df.index),
                    "Frequency": mean_hazard,
                    "Productivity Loss %": [np.nan]*len(hz_df.index),
                }))
            # Financial loss calculation
            annual_fin_loss = mean_hazard * cost_params[hazard] * asset["vulnerability"]
            fin_table = pd.DataFrame({
                "Asset": asset["name"],
                "Hazard": hazard,
                "Source of Loss": hazard,
                "Year": hz_df.index,
                "Annual Frequency": mean_hazard.values,
                "Loss per Event (INR million)": cost_params[hazard],
                "Estimated Annual Loss (INR million)": annual_fin_loss.values
            })
            financial_loss_tables.append(fin_table)
            # Actionable insights
            insights = actionable_insights(hazard, mean_hazard.mean(), asset, thresholds)
            st.info("**Actionable Insights:**")
            for insight in insights:
                st.write(f"- {insight}")
        if asset_results:
            output_tables.append(pd.concat(asset_results, ignore_index=True))
    # Downloadable summary
    if output_tables:
        full_output = pd.concat(output_tables, ignore_index=True)
        st.download_button("Download Results as CSV", data=full_output.to_csv(index=False), file_name="hazard_assessment_results.csv")
    if financial_loss_tables:
        loss_df = pd.concat(financial_loss_tables, ignore_index=True)
        st.subheader("Estimated Financial Losses by Asset and Hazard (INR million)")
        st.dataframe(loss_df)
        st.download_button("Download Financial Loss Table as CSV", data=loss_df.to_csv(index=False), file_name="financial_loss_results.csv")

st.markdown("---")
st.caption("Demo: Multi-asset, multi-hazard, ensemble projection, uncertainty quantification, asset mapping, vulnerability, impact and actionable insights, and financial loss assessment.")
