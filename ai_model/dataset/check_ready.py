import pandas as pd
h = pd.read_csv("processed/hourly_sea_rain.csv")
t = pd.read_csv("processed/ais_traffic.csv")
print("hourly rows:", len(h), "| stations:", h.station.nunique(), "| AIS samples:", len(t))
print("\nDouglas sea state share (%):")
print((h.douglas_raw.value_counts(normalize=True).reindex(range(0, 10), fill_value=0) * 100).round(2).to_string())
bins = [-1, 0.1, 2.5, 7.6, 25, 50, 1e9]
names = ["dry (<=0.1)", "light (0.1-2.5)", "moderate (2.5-7.6)", "heavy (7.6-25)", "very heavy (25-50)", ">50"]
c = pd.cut(h.rain_mmph, bins=bins, labels=names).value_counts().reindex(names)
print("\nRain class (mm/h):")
print(pd.DataFrame({"hours": c, "share_%": (c / len(h) * 100).round(3)}).to_string())
print("\nAIS vessels within 12 NM: median", t.n_12nm.median(), "| p95", t.n_12nm.quantile(0.95),
      "| max", t.n_12nm.max(), "| share with >=30 vessels:", round((t.n_12nm >= 30).mean() * 100, 1), "%")
mp = t.minpair_12nm_m.dropna()
print("Samples with >=2 vessels:", len(mp), "| of those, min spacing <=120 m:", round((mp <= 120).mean() * 100, 1), "%")