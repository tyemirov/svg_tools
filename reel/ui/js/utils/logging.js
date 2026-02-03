// @ts-check

/**
 * @param {string} message
 * @param {unknown} error
 * @returns {void}
 */
export function logError(message, error) {
  if (error instanceof Error) {
    console.error(message, error.message);
    return;
  }
  console.error(message);
}
