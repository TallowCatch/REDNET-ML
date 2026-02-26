export async function loadOpsPayload() {
  const res = await fetch('/ops/ops_payload.json');
  if (!res.ok) {
    throw new Error(`Failed to load ops payload: ${res.status}`);
  }
  return res.json();
}
