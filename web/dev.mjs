import { spawn } from "node:child_process";

const children = [
  ["server", spawn(process.execPath, ["web/server.mjs"], { stdio: "inherit" })],
  ["worker", spawn(process.execPath, ["web/worker.mjs"], { stdio: "inherit" })]
];

for (const [name, child] of children) {
  console.log(`[dev] started ${name} pid=${child.pid}`);
  child.on("exit", (code, signal) => {
    console.log(`[dev] ${name} exited code=${code ?? "null"} signal=${signal ?? "null"}`);
  });
}

function shutdown() {
  for (const [, child] of children) {
    if (!child.killed) child.kill("SIGTERM");
  }
}

process.on("SIGINT", shutdown);
process.on("SIGTERM", shutdown);
