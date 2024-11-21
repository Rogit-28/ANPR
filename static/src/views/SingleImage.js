import api from '../api/client.js';
import { formatFileSize } from '../utils.js';

export default () => ({
  // State
  selectedFile: null,
  previewUrl: null,
  processing: false,
  result: null,
  error: null,
  isDragging: false,

  init() {
    // Reset state on init
    this.reset();
  },

  reset() {
    this.selectedFile = null;
    this.previewUrl = null;
    this.processing = false;
    this.result = null;
    this.error = null;
  },

  handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
      this.setFile(file);
    }
    e.target.value = ''; // Reset input
  },

  handleDrop(e) {
    this.isDragging = false;
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
      this.setFile(file);
    } else {
      this.$store.app.addToast('Please drop an image file', 'warning');
    }
  },

  setFile(file) {
    // Validate file type
    if (!file.type.startsWith('image/')) {
      this.$store.app.addToast('Please select an image file', 'error');
      return;
    }

    // Validate file size (max 10MB)
    if (file.size > 10 * 1024 * 1024) {
      this.$store.app.addToast('Image must be less than 10MB', 'error');
      return;
    }

    this.selectedFile = file;
    this.result = null;
    this.error = null;

    // Create preview URL
    if (this.previewUrl) {
      URL.revokeObjectURL(this.previewUrl);
    }
    this.previewUrl = URL.createObjectURL(file);
  },

  async processImage() {
    if (!this.selectedFile) return;

    this.processing = true;
    this.error = null;
    this.result = null;

    try {
      const result = await api.processImage(this.selectedFile);
      this.result = result;
      
      if (result.detections && result.detections.length > 0) {
        this.$store.app.addToast(
          `Found ${result.detections.length} plate(s)`, 
          'success'
        );
      } else {
        this.$store.app.addToast('No plates detected in image', 'info');
      }
    } catch (e) {
      this.error = e.message;
      this.$store.app.addToast('Processing failed: ' + e.message, 'error');
    } finally {
      this.processing = false;
    }
  },

  async addToWatchlist(plateText) {
    try {
      await this.$store.watchlist.addPlate(plateText);
      this.$store.app.addToast(`Added ${plateText} to watchlist`, 'success');
    } catch (e) {
      this.$store.app.addToast(e.message, 'error');
    }
  },

  formatFileSize,

  getConfidenceBadge(conf) {
    if (conf >= 0.9) return 'badge-success';
    if (conf >= 0.7) return 'badge-warning';
    return 'badge-danger';
  }
});
