import api from '../api/client.js';
import { formatTimeShort } from '../utils.js';

export default () => ({
  streamUrl: '',
  detections: [],
  pollInterval: null,
  isStreaming: false,

  async init() {
    const status = await this.$store.stream.getStatus();
    this.isStreaming = status.active;
    if (this.isStreaming) {
      this.startPolling();
    }
  },

  async startStream() {
    try {
      await this.$store.stream.start(this.streamUrl);
      this.isStreaming = true;
      this.startPolling();
      this.$store.app.addToast('Stream started', 'success');
    } catch (e) {
      this.$store.app.addToast(e.message, 'error');
    }
  },

  async stopStream() {
    try {
      await this.$store.stream.stop();
      this.isStreaming = false;
      this.stopPolling();
      this.$store.app.addToast('Stream stopped', 'info');
    } catch (e) {
      this.$store.app.addToast(e.message, 'error');
    }
  },

  startPolling() {
    this.pollInterval = setInterval(async () => {
      // Poll for recent detections (simulated real-time feed)
      try {
        const recent = await api.getRecentDetections(10);
        this.detections = recent.items || recent;
        
        // Check stream status
        const status = await this.$store.stream.getStatus();
        if (!status.active && this.isStreaming) {
          this.isStreaming = false;
          this.stopPolling();
          this.$store.app.addToast('Stream ended unexpectedly', 'warning');
        }
      } catch (e) {
        console.error(e);
      }
    }, 1000);
  },

  stopPolling() {
    if (this.pollInterval) {
      clearInterval(this.pollInterval);
      this.pollInterval = null;
    }
  },

  formatTime: formatTimeShort,

  getConfidenceClass(conf) {
    if (conf >= 0.9) return 'text-green-500 font-bold';
    if (conf >= 0.7) return 'text-yellow-500 font-bold';
    return 'text-red-500 font-bold';
  },

  async addToWatchlist(plate) {
    try {
      await this.$store.watchlist.addPlate(plate);
      this.$store.app.addToast(`Added ${plate} to watchlist`, 'success');
    } catch (e) {
      this.$store.app.addToast(e.message, 'error');
    }
  }
});
