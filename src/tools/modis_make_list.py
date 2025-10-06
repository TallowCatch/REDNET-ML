from datetime import date
from calendar import monthrange
from pathlib import Path

# AQUA_MODIS.YYYYMM01_YYYYMMDD.L3m.MO.CHL.x_chlor_a.nc
out = Path("data/chl_tiles/requested_files/modis_monthly.txt")
out.parent.mkdir(parents=True, exist_ok=True)

rows = []
for y in range(2015, 2025):
    for m in range(1, 13):
        d1 = date(y, m, 1)
        d2 = date(y, m, monthrange(y, m)[1])
        fn = f"AQUA_MODIS.{y}{m:02d}01_{y}{m:02d}{d2.day:02d}.L3m.MO.CHL.x_chlor_a.nc"
        # Official OB.DAAC “direct getfile” endpoint format
        url = f"https://oceandata.sci.gsfc.nasa.gov/ob/getfile/{fn}"
        rows.append(url)

out.write_text("\n".join(rows))
print("Wrote", out, "(", len(rows), "files )")
