// @ts-check

import { BACKEND_URL } from "../config.js";

/**
 * @param {string} path
 * @returns {string}
 */
function buildUrl(path) {
  const base = BACKEND_URL.endsWith("/") ? BACKEND_URL : `${BACKEND_URL}/`;
  const normalized = path.startsWith("/") ? path.slice(1) : path;
  return `${base}${normalized}`;
}

/**
 * @returns {{ baseUrl: string, listJobs: () => Promise<object>, createJob: (body: FormData) => Promise<object>, deleteJob: (jobId: string) => Promise<object>, jobStream: () => EventSource, downloadUrl: (jobId: string) => string }}
 */
export function createBackendClient() {
  return {
    baseUrl: BACKEND_URL,
    listJobs: async () => {
      const response = await fetch(buildUrl("/api/jobs"));
      if (!response.ok) {
        throw new Error("backend.jobs.list_failed");
      }
      return response.json();
    },
    createJob: async (body) => {
      const response = await fetch(buildUrl("/api/jobs"), {
        method: "POST",
        body,
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(payload.error || "backend.jobs.create_failed");
      }
      return payload;
    },
    deleteJob: async (jobId) => {
      const response = await fetch(buildUrl(`/api/jobs/${jobId}`), {
        method: "DELETE",
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(payload.error || "backend.jobs.delete_failed");
      }
      return payload;
    },
    jobStream: () => new EventSource(buildUrl("/api/jobs/events")),
    downloadUrl: (jobId) => buildUrl(`/api/jobs/${jobId}/srt`),
  };
}
