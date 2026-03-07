import fs from "node:fs/promises";

export async function readTail(filePath, maxBytes) {
  try {
    const handle = await fs.open(filePath, "r");
    try {
      const stat = await handle.stat();
      const size = stat.size;
      const start = Math.max(0, size - maxBytes);
      const length = size - start;
      const buffer = Buffer.alloc(length);
      await handle.read(buffer, 0, length, start);
      return buffer.toString("utf8");
    } finally {
      await handle.close();
    }
  } catch {
    return "";
  }
}

export function parseFinalMetrics(logText) {
  const match = logText.match(
    /=== Validation[\s\S]*?PSNR:\s*([0-9.]+)\s+SSIM:\s*([0-9.]+)\s+L1:\s*([0-9.]+)\s+Gaussians:\s*([0-9]+)/m
  );

  if (!match) {
    return null;
  }

  return {
    psnr: Number(match[1]),
    ssim: Number(match[2]),
    l1: Number(match[3]),
    gaussians: Number(match[4])
  };
}
