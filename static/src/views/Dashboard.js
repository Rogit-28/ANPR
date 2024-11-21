import api from '../api/client.js';
import { formatTime } from '../utils.js';

export default () => ({
  health: {},
  recentDetections: [],
  
  async init() {
    try {
      this.health = await api.getHealth();
    } catch (e) {
      console.error('Failed to fetch health', e);
      this.health = { status: 'error' };
    }
    
    try {
      const response = await api.getRecentDetections(10);
      // API returns { detections: [...], total: N, ... }
      this.recentDetections = response.detections || response.items || response || [];
    } catch (e) {
      console.error('Failed to fetch recent detections', e);
      this.recentDetections = [];
    }
  },

  getConfidenceClass(conf) {
    if (conf >= 0.9) return 'text-green-600 dark:text-green-400 font-medium';
    if (conf >= 0.7) return 'text-yellow-600 dark:text-yellow-400 font-medium';
    return 'text-red-600 dark:text-red-400 font-medium';
  },

  formatTime,

  getSnapshotUrl(id) {
    return api.getSnapshotUrl(id);
  },
  
  // Navigate to job results in grouped view
  viewJobResults(jobId) {
    this.$store.app.navigateToJobResults(jobId);
  },
  
  // Navigate to results and show detection modal
  viewDetection(detection) {
    this.$store.app.navigateToDetection(detection);
  }
});
