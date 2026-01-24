// @ts-check

import { DEFAULT_BACKEND_URL } from "./constants.js";

/**
 * @typedef {{ backendUrl?: string }} RuntimeConfig
 */

/** @type {RuntimeConfig} */
const runtimeConfig = typeof window !== "undefined" ? window.__AUDIO_TO_TEXT_CONFIG__ || {} : {};

const configuredUrl =
  typeof runtimeConfig.backendUrl === "string" && runtimeConfig.backendUrl.trim()
    ? runtimeConfig.backendUrl.trim()
    : DEFAULT_BACKEND_URL;

export const BACKEND_URL = configuredUrl;
