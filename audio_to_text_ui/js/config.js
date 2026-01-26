// @ts-check

import { DEFAULT_BACKEND_URL } from "./constants.js";

/**
 * @typedef {{ backendUrl?: string }} RuntimeConfig
 */

/** @type {RuntimeConfig} */
const runtimeConfig = typeof window !== "undefined" ? window.__AUDIO_TO_TEXT_CONFIG__ || {} : {};

const defaultBackendUrl =
  typeof window !== "undefined" && window.location && window.location.hostname
    ? `${window.location.protocol}//${window.location.hostname}:8080`
    : DEFAULT_BACKEND_URL;

const configuredUrl =
  typeof runtimeConfig.backendUrl === "string" && runtimeConfig.backendUrl.trim()
    ? runtimeConfig.backendUrl.trim()
    : defaultBackendUrl;

export const BACKEND_URL = configuredUrl;
