import { createWorker } from "./src/worker-app.mjs";

const worker = await createWorker();

process.on("SIGINT", async () => {
  await worker.close();
});

process.on("SIGTERM", async () => {
  await worker.close();
});

console.log(`msplat worker polling ${worker.config.databasePath}`);
await worker.run();
