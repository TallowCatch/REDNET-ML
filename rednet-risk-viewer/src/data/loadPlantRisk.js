import Papa from "papaparse";

export async function loadPlantRisk(url) {
  const res = await fetch(url);
  const text = await res.text();

  const { data } = Papa.parse(text, {
    header: true,
    dynamicTyping: true,
    skipEmptyLines: true,
  });

  return data.map(d => ({
    time: new Date(d.datetime),
    risk: d.hab_prob,
  }));
}
