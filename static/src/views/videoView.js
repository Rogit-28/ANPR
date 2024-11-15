import api from '../api/client.js';
import { formatFileSize } from '../utils.js';

export default () => ({
  files: [],
  uploading: false,
  isDragging: false,
  _pollIntervalId: null,

  async init() {
    this.$store.jobs.fetchJobs();
    this._pollIntervalId = setInterval(() => this.$store.jobs.fetchJobs(), 5000);
  },

  destroy() {
    if (this._pollIntervalId) {
      clearInterval(this._pollIntervalId);
      this._pollIntervalId = null;
    }
  },

  get hasCompleted() {
    return this.files.some(f => f.status === 'completed');
  },

  get hasPending() {
    return this.files.some(f => f.status === 'pending');
  },

  handleFileSelect(e) {
    this.addFiles(e.target.files);
    e.target.value = ''; // Reset input
  },

  handleDrop(e) {
    this.isDragging = false;
    this.addFiles(e.dataTransfer.files);
  },

  addFiles(fileList) {
    const newFiles = Array.from(fileList)
      .filter(f => f.type.startsWith('video/'))
      .map(f => ({
        file: f,
        status: 'pending',
        progress: 0,
        error: null
      }));
    this.files.push(...newFiles);
  },

  removeFile(index) {
    this.files.splice(index, 1);
  },

  clearCompleted() {
    this.files = this.files.filter(f => f.status !== 'completed');
  },

  async uploadAll() {
    this.uploading = true;
    
    for (const item of this.files) {
      if (item.status !== 'pending') continue;

      item.status = 'uploading';
      try {
        await api.uploadVideo(item.file, (progress) => {
          item.progress = progress;
        });
        item.status = 'completed';
        this.$store.app.addToast(`Uploaded ${item.file.name}`, 'success');
      } catch (err) {
        item.status = 'error';
        item.error = err.message;
        this.$store.app.addToast(`Failed to upload ${item.file.name}`, 'error');
      }
    }

    this.uploading = false;
    this.$store.jobs.fetchJobs();
  },

  formatFileSize,

  getStatusBadge(status) {
    switch (status) {
      case 'completed': return 'badge-success';
      case 'uploading': return 'badge-info';
      case 'error': return 'badge-danger';
      default: return 'badge-neutral';
    }
  },

  getJobStatusBadge(status) {
    switch (status) {
      case 'completed': return 'badge-success';
      case 'processing': return 'badge-warning';
      case 'failed': return 'badge-danger';
      case 'cancelled': return 'badge-neutral';
      default: return 'badge-info';
    }
  }
});
