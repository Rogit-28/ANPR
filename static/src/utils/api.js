// ANPR API Client
const API_BASE = '/api/v1';

// Request helper with error handling
async function request(method, endpoint, data = null, options = {}) {
  const url = `${API_BASE}${endpoint}`;
  const config = {
    method,
    headers: {},
    ...options
  };

  if (data) {
    if (data instanceof FormData) {
      config.body = data;
    } else {
      config.headers['Content-Type'] = 'application/json';
      config.body = JSON.stringify(data);
    }
  }

  try {
    const response = await fetch(url, config);
    
    // Handle empty responses
    const text = await response.text();
    const result = text ? JSON.parse(text) : {};

    if (!response.ok) {
      throw new Error(result.detail?.message || result.detail || `HTTP ${response.status}: ${response.statusText}`);
    }

    // Extract data from SuccessResponse wrapper if present
    if (result.status === 'success' && result.data !== undefined) {
      return result.data;
    }
    return result;
  } catch (error) {
    if (error.name === 'TypeError' && error.message.includes('fetch')) {
      throw new Error('Unable to connect to server. Is the API running?');
    }
    throw error;
  }
}

// API Client
const api = {
  // Health & Stats
  getHealth() {
    return request('GET', '/health');
  },

  getStats() {
    return request('GET', '/stats');
  },

  // Jobs
  getJobs(params = {}) {
    const query = new URLSearchParams(params).toString();
    return request('GET', `/jobs${query ? `?${query}` : ''}`);
  },

  getJob(id) {
    return request('GET', `/jobs/${id}`);
  },

  getJobDetections(id) {
    return request('GET', `/jobs/${id}/detections`);
  },

  cancelJob(id) {
    return request('POST', `/jobs/${id}/cancel`);
  },

  deleteJob(id) {
    return request('DELETE', `/jobs/${id}`);
  },

  uploadVideo(file, onProgress = null) {
    return new Promise((resolve, reject) => {
      const formData = new FormData();
      formData.append('file', file);

      const xhr = new XMLHttpRequest();

      if (onProgress) {
        xhr.upload.onprogress = (e) => {
          if (e.lengthComputable) {
            onProgress(Math.round((e.loaded / e.total) * 100));
          }
        };
      }

      xhr.onload = () => {
        try {
          const response = JSON.parse(xhr.responseText);
          if (xhr.status >= 200 && xhr.status < 300) {
            resolve(response);
          } else {
            reject(new Error(response.detail || `Upload failed: ${xhr.status}`));
          }
        } catch (e) {
          reject(new Error('Invalid server response'));
        }
      };

      xhr.onerror = () => reject(new Error('Network error during upload'));
      xhr.ontimeout = () => reject(new Error('Upload timed out'));

      xhr.open('POST', `${API_BASE}/upload`);
      xhr.send(formData);
    });
  },

  // Queue
  getQueueStatus() {
    return request('GET', '/queue/status');
  },

  // Streaming
  startStream(sourceUrl, cameraId = 'camera_1') {
    return request('POST', '/stream/start', {
      video_source: sourceUrl,
      camera_id: cameraId
    });
  },

  getStreamStatus() {
    return request('GET', '/stream/status');
  },

  stopStream() {
    return request('POST', '/stream/stop');
  },

  // Video processing
  processVideo(filePath) {
    return request('POST', '/video/process', { video_path: filePath });
  },

  getVideoStatus(jobId) {
    return request('GET', `/video/status/${jobId}`);
  },

  // Detections
  getDetections(params = {}) {
    const query = new URLSearchParams();
    if (params.plate_text) query.append('plate_text', params.plate_text);
    if (params.min_confidence) query.append('min_confidence', params.min_confidence);
    if (params.start_time) query.append('start_time', params.start_time);
    if (params.end_time) query.append('end_time', params.end_time);
    if (params.source_identifier) query.append('source_identifier', params.source_identifier);
    if (params.limit) query.append('limit', params.limit);
    if (params.offset) query.append('offset', params.offset);
    
    const queryStr = query.toString();
    return request('GET', `/detections${queryStr ? `?${queryStr}` : ''}`);
  },

  getDetection(id) {
    return request('GET', `/detections/${id}`);
  },

  deleteDetection(id) {
    return request('DELETE', `/detections/${id}`);
  },

  getRecentDetections(limit = 10) {
    return request('GET', `/detections/recent?limit=${limit}`);
  },

  // Detection Events (grouped detections)
  getDetectionEvents(params = {}) {
    const query = new URLSearchParams();
    if (params.time_window) query.append('time_window', params.time_window);
    if (params.plate_text) query.append('plate_text', params.plate_text);
    if (params.source_identifier) query.append('source_identifier', params.source_identifier);
    if (params.start_time) query.append('start_time', params.start_time);
    if (params.end_time) query.append('end_time', params.end_time);
    if (params.min_confidence) query.append('min_confidence', params.min_confidence);
    if (params.limit) query.append('limit', params.limit);
    if (params.offset) query.append('offset', params.offset);
    
    const queryStr = query.toString();
    return request('GET', `/detections/events${queryStr ? `?${queryStr}` : ''}`);
  },

  getEventDetections(plateText, startTime, endTime, limit = 100) {
    const query = new URLSearchParams({
      start_time: startTime,
      end_time: endTime,
      limit: limit.toString()
    });
    return request('GET', `/detections/events/${encodeURIComponent(plateText)}?${query}`);
  },

  getSnapshotUrl(id) {
    return `${API_BASE}/snapshot/${id}`;
  },

  // Watchlist
  getWatchlist() {
    return request('GET', '/watchlist');
  },

  addToWatchlist(plateNumber) {
    return request('POST', '/watchlist', { plate_text: plateNumber });
  },

  removeFromWatchlist(plate) {
    return request('DELETE', `/watchlist/${encodeURIComponent(plate)}`);
  },

  checkWatchlist(plate) {
    return request('GET', `/watchlist/check/${encodeURIComponent(plate)}`);
  },

  clearWatchlist() {
    return request('DELETE', '/watchlist/clear');
  },

  // Alerts
  getAlertStats() {
    return request('GET', '/alerts/stats');
  },

  getAlertHistory(params = {}) {
    const query = new URLSearchParams(params).toString();
    return request('GET', `/alerts/history${query ? `?${query}` : ''}`);
  },

  enableAlerts() {
    return request('POST', '/alerts/enable');
  },

  disableAlerts() {
    return request('POST', '/alerts/disable');
  },

  acknowledgeAlert(alertId) {
    return request('POST', `/alerts/${alertId}/acknowledge`);
  },

  deleteAlert(alertId) {
    return request('DELETE', `/alerts/${alertId}`);
  },

  testAlert() {
    return request('POST', '/alerts/test');
  },

  // System & Configuration
  getSystemInfo() {
    return request('GET', '/system/info');
  },

  getConfig() {
    return request('GET', '/config');
  },

  updateConfig(data) {
    return request('PUT', '/config', data);
  },

  resetConfig() {
    return request('POST', '/config/reset');
  },

  restartServer() {
    return request('POST', '/server/restart');
  },

  // Single Image (nice-to-have)
  processImage(file) {
    const formData = new FormData();
    formData.append('image', file);
    return request('POST', '/image/process', formData);
  },

  // Unified analyze endpoint (images and videos)
  analyzeFile(file, onProgress = null) {
    return new Promise((resolve, reject) => {
      const formData = new FormData();
      formData.append('file', file);

      const xhr = new XMLHttpRequest();

      if (onProgress) {
        xhr.upload.onprogress = (e) => {
          if (e.lengthComputable) {
            onProgress(Math.round((e.loaded / e.total) * 100));
          }
        };
      }

      xhr.onload = () => {
        try {
          const response = JSON.parse(xhr.responseText);
          if (xhr.status >= 200 && xhr.status < 300) {
            // Extract data from SuccessResponse wrapper
            if (response.status === 'success' && response.data !== undefined) {
              resolve(response.data);
            } else {
              resolve(response);
            }
          } else {
            reject(new Error(response.detail?.message || response.detail || `Upload failed: ${xhr.status}`));
          }
        } catch (e) {
          reject(new Error('Invalid server response'));
        }
      };

      xhr.onerror = () => reject(new Error('Network error during upload'));
      xhr.ontimeout = () => reject(new Error('Upload timed out'));

      xhr.open('POST', `${API_BASE}/analyze`);
      xhr.send(formData);
    });
  },

  // Video URLs (for players)
  getAnnotatedVideoUrl(jobId) {
    return `${API_BASE}/jobs/${jobId}/video/annotated`;
  },

  getNormalizedVideoUrl(jobId) {
    return `${API_BASE}/jobs/${jobId}/video/normalized`;
  },

  // Image URLs (for display)
  getAnnotatedImageUrl(jobId) {
    return `${API_BASE}/jobs/${jobId}/image/annotated`;
  },

  getOriginalImageUrl(jobId) {
    return `${API_BASE}/jobs/${jobId}/image/original`;
  }
};

export default api;
