// src/data/loadPlants.js
export async function loadPlants() {
    const res = await fetch("/data/plants.json");
    if (!res.ok) throw new Error("Failed to load /data/plants.json");
    return await res.json();
  }
  