export const PRESETS = {
  preview: {
    key: "preview",
    label: "Preview",
    iterations: 1500,
    downscaleFactor: 2,
    args: ["-n", "1500", "-d", "2", "--num-downscales", "2", "--val"]
  },
  standard: {
    key: "standard",
    label: "Standard",
    iterations: 7000,
    downscaleFactor: 1,
    args: ["-n", "7000", "-d", "1", "--num-downscales", "0", "--val"]
  },
  high: {
    key: "high",
    label: "High",
    iterations: 30000,
    downscaleFactor: 1,
    args: ["-n", "30000", "-d", "1", "--num-downscales", "0", "--val"]
  }
};

export function getPreset(key) {
  return PRESETS[key] ?? null;
}

export function listPresets() {
  return Object.values(PRESETS);
}
