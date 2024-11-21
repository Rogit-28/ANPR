import api from '../api/client.js';
import { formatFileSize, formatTime } from '../utils.js';

export default () => ({
  // Upload state
  files: [],
  uploading: false,
  isDragging: false,
  
  // Job expansion tracking
  expandedJobs: {},
  
  // Job detections cache
  jobDetections: {},
  
  // Detection sorting: 'time' or 'confidence'
  detectionSort: 'time',
  sortDescending: false,
  
  // Polling
  _pollIntervalId: null,
  
  // Video player refs (managed per job)
  videoPlayers: {},

  async init() {
    await this.$store.jobs.fetchJobs();
    this._pollIntervalId = setInterval(() => this.$store.jobs.fetchJobs(), 3000);
    
    // Auto-expand the most recent completed job
    this.$nextTick(() => {
      this.autoExpandLatestCompleted();
    });
  },

  destroy() {
    if (this._pollIntervalId) {
      clearInterval(this._pollIntervalId);
      this._pollIntervalId = null;
    }
  },
  
  autoExpandLatestCompleted() {
    const jobs = this.$store.jobs.items || [];
    const completedJob = jobs.find(j => j.status === 'completed');
    if (completedJob && Object.keys(this.expandedJobs).length === 0) {
      this.toggleJobExpanded(completedJob.id);
    }
  },

  // File handling
  handleFileSelect(e) {
    this.addFiles(e.target.files);
    e.target.value = '';
  },

  handleDrop(e) {
    this.isDragging = false;
    this.addFiles(e.dataTransfer.files);
  },

  addFiles(fileList) {
    const imageTypes = ['image/jpeg', 'image/png', 'image/webp', 'image/bmp'];
    const videoTypes = ['video/mp4', 'video/avi', 'video/quicktime', 'video/x-matroska', 'video/webm'];
    const allowedTypes = [...imageTypes, ...videoTypes];
    
    const newFiles = Array.from(fileList)
      .filter(f => {
        // Check by MIME type or extension
        if (allowedTypes.includes(f.type)) return true;
        const ext = f.name.split('.').pop()?.toLowerCase();
        const allowedExts = ['jpg', 'jpeg', 'png', 'webp', 'bmp', 'mp4', 'avi', 'mov', 'mkv', 'webm', 'm4v'];
        return allowedExts.includes(ext);
      })
      .map(f => ({
        file: f,
        status: 'pending',
        progress: 0,
        error: null,
        jobId: null,
        isImage: f.type.startsWith('image/') || ['jpg', 'jpeg', 'png', 'webp', 'bmp'].includes(f.name.split('.').pop()?.toLowerCase())
      }));
    
    this.files.push(...newFiles);
  },

  removeFile(index) {
    this.files.splice(index, 1);
  },

  clearCompleted() {
    this.files = this.files.filter(f => f.status !== 'completed');
  },

  get hasCompleted() {
    return this.files.some(f => f.status === 'completed');
  },

  get hasPending() {
    return this.files.some(f => f.status === 'pending');
  },

  async uploadAll() {
    this.uploading = true;
    
    for (const item of this.files) {
      if (item.status !== 'pending') continue;

      item.status = 'uploading';
      try {
        const result = await api.analyzeFile(item.file, (progress) => {
          item.progress = progress;
        });
        
        item.status = 'completed';
        item.jobId = result.job_id;
        
        // For images, they're processed immediately - expand the job
        if (item.isImage && result.status === 'completed') {
          this.$store.app.addToast(`Processed ${item.file.name}: ${result.message}`, 'success');
        } else {
          this.$store.app.addToast(`Uploaded ${item.file.name}`, 'success');
        }
        
        // Refresh jobs
        await this.$store.jobs.fetchJobs();
        
        // Auto-expand the newly created job
        if (result.job_id) {
          this.expandedJobs = { [result.job_id]: true };
          this.loadJobDetections(result.job_id);
        }
        
      } catch (err) {
        item.status = 'error';
        item.error = err.message;
        this.$store.app.addToast(`Failed: ${item.file.name} - ${err.message}`, 'error');
      }
    }

    this.uploading = false;
  },

  // Job expansion
  async toggleJobExpanded(jobId) {
    if (this.expandedJobs[jobId]) {
      delete this.expandedJobs[jobId];
    } else {
      this.expandedJobs = { [jobId]: true };  // Collapse others, expand this one
      await this.loadJobDetections(jobId);
    }
  },
  
  isJobExpanded(jobId) {
    return !!this.expandedJobs[jobId];
  },
  
  async loadJobDetections(jobId) {
    // Skip if already loaded with detections
    if (this.jobDetections[jobId] && this.jobDetections[jobId].length > 0) return;
    
    try {
      const result = await api.getJobDetections(jobId);
      // Use spread to ensure Alpine reactivity
      this.jobDetections = { 
        ...this.jobDetections, 
        [jobId]: result.detections || [] 
      };
      console.log(`Loaded ${result.detections?.length || 0} detections for job ${jobId}`);
    } catch (e) {
      console.error('Failed to load job detections', e);
      this.jobDetections = { 
        ...this.jobDetections, 
        [jobId]: [] 
      };
    }
  },
  
  getJobDetections(jobId) {
    return this.jobDetections[jobId] || [];
  },
  
  toggleSort(sort) {
    if (this.detectionSort === sort) {
      // Same sort field - toggle direction
      this.sortDescending = !this.sortDescending;
    } else {
      // New sort field - reset to ascending
      this.detectionSort = sort;
      this.sortDescending = false;
    }
  },
  
  getSortedDetections(jobId) {
    const detections = this.getJobDetections(jobId);
    if (!detections.length) return [];
    
    const sorted = [...detections];
    if (this.detectionSort === 'confidence') {
      sorted.sort((a, b) => {
        const diff = (b.confidence || 0) - (a.confidence || 0);
        return this.sortDescending ? -diff : diff;
      });
    } else {
      // Default: sort by time/frame_number
      sorted.sort((a, b) => {
        const diff = (a.frame_number || 0) - (b.frame_number || 0);
        return this.sortDescending ? -diff : diff;
      });
    }
    return sorted;
  },

  // Video player seek
  seekToDetection(jobId, detection) {
    const job = this.getJobById(jobId);
    if (!job || !job.fps || !detection.frame_number) return;
    
    const videoEl = document.getElementById(`video-player-${jobId}`);
    if (videoEl) {
      const time = detection.frame_number / job.fps;
      videoEl.pause();
      videoEl.currentTime = time;
    }
  },
  
  formatVideoTime(frameNumber, fps) {
    if (!frameNumber || !fps) return '0:00';
    const totalSeconds = Math.floor(frameNumber / fps);
    const minutes = Math.floor(totalSeconds / 60);
    const seconds = totalSeconds % 60;
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
  },

  // Job helpers
  getJobById(jobId) {
    return (this.$store.jobs.items || []).find(j => j.id === jobId);
  },
  
  getRecentJobs() {
    // Return up to 10 most recent jobs
    return (this.$store.jobs.items || []).slice(0, 10);
  },

  // URLs
  getAnnotatedVideoUrl(jobId) {
    return api.getAnnotatedVideoUrl(jobId);
  },
  
  getAnnotatedImageUrl(jobId) {
    return api.getAnnotatedImageUrl(jobId);
  },
  
  getSnapshotUrl(id) {
    return api.getSnapshotUrl(id);
  },

  // Formatting
  formatFileSize,
  formatTime,

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
      case 'annotating': return 'badge-info';
      case 'normalizing': return 'badge-info';
      case 'failed': return 'badge-danger';
      case 'cancelled': return 'badge-neutral';
      default: return 'badge-info';
    }
  },

  getConfidenceBadge(conf) {
    if (conf >= 0.9) return 'badge-success';
    if (conf >= 0.7) return 'badge-warning';
    return 'badge-danger';
  },
  
  // Job actions
  async cancelJob(jobId) {
    try {
      await this.$store.jobs.cancelJob(jobId);
      this.$store.app.addToast('Job cancelled', 'info');
    } catch (e) {
      this.$store.app.addToast(e.message, 'error');
    }
  },
  
  async deleteJob(jobId) {
    if (!confirm('Delete this job and all its detections?')) return;
    try {
      await this.$store.jobs.deleteJob(jobId);
      delete this.expandedJobs[jobId];
      delete this.jobDetections[jobId];
      this.$store.app.addToast('Job deleted', 'success');
    } catch (e) {
      this.$store.app.addToast(e.message, 'error');
    }
  },

  // Navigate to results
  viewInResults(jobId) {
    this.$store.app.navigateToJobResults(jobId);
  }
});
