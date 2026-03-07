import { createAppServer } from "./src/server-app.mjs";

const app = await createAppServer();
await app.start();
console.log(`msplat internal site listening on http://${app.config.host}:${app.config.port}`);
