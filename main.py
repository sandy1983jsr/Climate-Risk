import streamlit as st
import numpy as np
import pandas as pd
import xarray as xr
import intake
import matplotlib.pyplot as plt
import io

# Optional: for interactive maps, install streamlit-folium and folium
try:
    import folium
    from streamlit_folium import st_folium
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False

CATALOG_URL = "https://storage.googleapis.com/cmip6/pangeo-cmip6.json"
HAZARD_OPTIONS = {
    "Heatwave": {"variable": "tasmax", "description": "Max temp > 35°C"},
    "Drought": {"variable": "pr", "description": "Monthly pr < 50mm"},
    "Precipitation Extreme": {"variable": "pr", "description": "Rain > 30mm/day"},
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

def load_cmip6_ensemble(variable, scenario, lat, lon, years, n_models=3):
    cat = get_cmip6_catalog()
    df = cat.df
    sel = (
        (df['variable_id'] == variable) &
        (df['experiment_id'] == scenario) &
        (df['table_id'].str.contains("day|Amon"))
    )
    subset = df[sel].head(n_models)
    results = []
    for _, row in subset.iterrows():
        try:
            ds = xr.open_zarr(row.zstore, consolidated=True)
            ds = ds.sel(time=slice(f"{years[0]}-01-01", f"{years[1]}-12-31"))
            lat_idx, lon_idx = get_nearest_grid(ds, lat, lon)
            point_data = ds.isel(lat=lat_idx, lon=lon_idx)
            results.append(point_data)
        except Exception: continue
    return results

def calculate_hazard_ensemble(hazard, data_list):
    series_list = []
    for data in data_list:
        if hazard == "Heatwave":
            tasmax = data["tasmax"] - 273.15
            hot_days = (tasmax > 35).groupby("time.year").sum()
            series_list.append(hot_days)
        elif hazard == "Drought":
            pr = data["pr"] * 86400 * 30
            pr_monthly = pr.resample(time="1M").sum()
            dry_months = (pr_monthly < 50).groupby("time.year").sum()
            series_list.append(dry_months)
        elif hazard == "Precipitation Extreme":
            pr = data["pr"] * 86400
            wet_days = (pr > 30).groupby("time.year").sum()
            series_list.append(wet_days)
    # Align to years, convert to DataFrame
    df = pd.DataFrame({i: s.values for i, s in enumerate(series_list)}, index=series_list[0]["year"].values)
    return df

def plot_ensemble(years, df, hazard_name):
    plt.figure(figsize=(7,3))
    mean = df.mean(axis=1)
    std = df.std(axis=1)
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
    # Equipment
    if hazard == "Heatwave" and mean_value > thresholds["equip_temp_limit"]:
        insights.append("Heat exceeds equipment design: Invest in cooling, maintenance, or heat-resistant equipment.")
    # Asset value
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
st.write("""Multi-asset, multi-hazard, ensemble projections, uncertainty quantification, asset mapping, vulnerability, and actionable insights.""")

col1, col2 = st.columns(2)
with col1:
    lat = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=28.6, format="%.4f")
    lon = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=77.2, format="%.4f")
with col2:
    start_year = st.number_input("Start Year", min_value=2015, max_value=2100, value=2021)
    end_year = st.number_input("End Year", min_value=start_year, max_value=2100, value=2050)
scenario = st.selectbox("IPCC Scenario", options=list(SCENARIOS.keys()))
hazards = st.multiselect("Select Hazards", options=list(HAZARD_OPTIONS.keys()), default=["Heatwave"])

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

# Vulnerability
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

thresholds = {
    "heat_risk_days": health_risk_days,
    "heat_monitor_days": health_monitor_days,
    "equip_temp_limit": equip_temp_limit,
    "logistics_days": logistics_days,
    "drought_months": drought_months,
}

if st.button("Run Multi-Asset Analysis"):
    scenario_code = SCENARIOS[scenario]
    output_tables = []
    for idx, asset in assets_df.iterrows():
        st.markdown(f"### Asset: **{asset['name']}**  (Vulnerability: {asset['vulnerability']:.2f})")
        asset_results = []
        for hazard in hazards:
            st.markdown(f"**{hazard}: {HAZARD_OPTIONS[hazard]['description']}**")
            st.info("Loading ensemble data...")
            ens = load_cmip6_ensemble(
                HAZARD_OPTIONS[hazard]["variable"],
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
            elif hazard == "Precipitation Extreme":
                st.write("**Estimated annual logistics impact (days with heavy rain):**")
                st.dataframe(pd.DataFrame({"Year": hz_df.index, "Logistics Impact Days": mean_hazard}))
                asset_results.append(pd.DataFrame({
                    "Year": hz_df.index,
                    "Hazard": [hazard]*len(hz_df.index),
                    "Frequency": mean_hazard,
                    "Productivity Loss %": [np.nan]*len(hz_df.index),
                }))
            elif hazard == "Drought":
                st.write("**Estimated annual drought months:**")
                st.dataframe(pd.DataFrame({"Year": hz_df.index, "Drought Months": mean_hazard}))
                asset_results.append(pd.DataFrame({
                    "Year": hz_df.index,
                    "Hazard": [hazard]*len(hz_df.index),
                    "Frequency": mean_hazard,
                    "Productivity Loss %": [np.nan]*len(hz_df.index),
                }))
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
        csv = full_output.to_csv(index=False)
        st.download_button("Download Results as CSV", data=csv, file_name="hazard_assessment_results.csv")

st.markdown("---")
st.caption("Demo: Multi-asset, multi-hazard, ensemble projection, uncertainty quantification, asset mapping, vulnerability, impact assessment, and actionable insights.")
