function formatValue(value) {
  if (value === undefined) return null;
  if (value === null) return "null";
  if (typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }
  if (value instanceof Error) {
    return JSON.stringify(value.message);
  }
  return JSON.stringify(value);
}

function writeLine(method, scope, level, message, fields = {}) {
  const parts = [
    `[${new Date().toISOString()}]`,
    `[${scope}]`,
    `[${level}]`,
    message
  ];

  const fieldParts = Object.entries(fields)
    .map(([key, value]) => {
      const rendered = formatValue(value);
      return rendered === null ? null : `${key}=${rendered}`;
    })
    .filter(Boolean);

  const line = fieldParts.length > 0 ? `${parts.join(" ")} ${fieldParts.join(" ")}` : parts.join(" ");
  method(line);
}

export function createLogger(scope) {
  return {
    info(message, fields = {}) {
      writeLine(console.log, scope, "INFO", message, fields);
    },
    warn(message, fields = {}) {
      writeLine(console.warn, scope, "WARN", message, fields);
    },
    error(message, fields = {}) {
      writeLine(console.error, scope, "ERROR", message, fields);
    }
  };
}
