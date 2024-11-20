/**
 * Shared utility functions for the ANPR frontend.
 */

/**
 * Format a timestamp to DD/MM/YYYY, HH:MM:SS format.
 * @param {string|Date} ts - Timestamp to format
 * @returns {string} Formatted date/time string
 */
export function formatTime(ts) {
  const date = new Date(ts);
  const day = date.getDate().toString().padStart(2, '0');
  const month = (date.getMonth() + 1).toString().padStart(2, '0');
  const year = date.getFullYear();
  const hours = date.getHours().toString().padStart(2, '0');
  const minutes = date.getMinutes().toString().padStart(2, '0');
  const seconds = date.getSeconds().toString().padStart(2, '0');
  return `${day}/${month}/${year}, ${hours}:${minutes}:${seconds}`;
}

/**
 * Format a timestamp to short time string (time only HH:MM:SS).
 * @param {string|Date} ts - Timestamp to format
 * @returns {string} Formatted time string
 */
export function formatTimeShort(ts) {
  const date = new Date(ts);
  const hours = date.getHours().toString().padStart(2, '0');
  const minutes = date.getMinutes().toString().padStart(2, '0');
  const seconds = date.getSeconds().toString().padStart(2, '0');
  return `${hours}:${minutes}:${seconds}`;
}

/**
 * Format a timestamp to DD/MM/YYYY format (date only).
 * @param {string|Date} ts - Timestamp to format
 * @returns {string} Formatted date string
 */
export function formatDate(ts) {
  const date = new Date(ts);
  const day = date.getDate().toString().padStart(2, '0');
  const month = (date.getMonth() + 1).toString().padStart(2, '0');
  const year = date.getFullYear();
  return `${day}/${month}/${year}`;
}

/**
 * Format file size in bytes to human-readable string.
 * @param {number} bytes - File size in bytes
 * @returns {string} Formatted size string (e.g., "1.5 MB")
 */
export function formatFileSize(bytes) {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}
