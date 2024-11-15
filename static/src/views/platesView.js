import api from '../api/client.js';
import { formatTime, formatTimeShort } from '../utils.js';

export default () => ({
  viewMode: 'events',  // 'events' (default), 'grouped', or 'grid'
  
  // Modal state
  showModal: false,
  selectedDetection: null,
  selectedEvent: null,
  snapshotLoading: false,
  snapshotError: false,
  
  // Export state
  exporting: false,
  
  // Store bound event handlers for proper cleanup
  _boundViewJobResults: null,
  _boundShowDetectionModal: null,

  async init() {
    // Store bound references for cleanup
    this._boundViewJobResults = this.handleViewJobResults.bind(this);
    this._boundShowDetectionModal = this.handleShowDetectionModal.bind(this);
    
    // Listen for navigation events from Dashboard
    window.addEventListener('view-job-results', this._boundViewJobResults);
    window.addEventListener('show-detection-modal', this._boundShowDetectionModal);
    
    // Use events view by default (much better performance)
    if (this.viewMode === 'events') {
      await this.$store.detections.fetchEvents();
    } else {
      await this.$store.detections.fetch();
    }
  },

  // Called when view is destroyed/hidden
  destroy() {
    if (this._boundViewJobResults) {
      window.removeEventListener('view-job-results', this._boundViewJobResults);
    }
    if (this._boundShowDetectionModal) {
      window.removeEventListener('show-detection-modal', this._boundShowDetectionModal);
    }
  },
  
  // Handle navigation to job results
  handleViewJobResults(event) {
    const { jobId } = event.detail;
    // Switch to grouped view
    this.viewMode = 'grouped';
    
    // Pre-set the expanded job BEFORE fetching so groupByJob() doesn't auto-expand first job
    this.$store.detections.expandedJobs = { [jobId]: true };
    this.$store.detections.page = 1;
    this.$store.detections.fetch();
  },
  
  // Handle showing detection modal
  handleShowDetectionModal(event) {
    const { detection } = event.detail;
    if (detection) {
      this.showDetail(detection);
    }
  },

  // Manual refresh
  refresh() {
    this.$store.detections.page = 1;  // Reset to first page
    if (this.viewMode === 'events') {
      this.$store.detections.fetchEvents();
    } else {
      this.$store.detections.fetch();
    }
  },
  
  // Switch view mode
  setViewMode(mode) {
    this.viewMode = mode;
    this.$store.detections.page = 1;  // Reset to first page when switching views
    if (mode === 'events') {
      this.$store.detections.fetchEvents();
    } else {
      this.$store.detections.fetch();
    }
  },

  // Events pagination
  nextEventsPage() {
    this.$store.detections.nextEventsPage();
  },

  prevEventsPage() {
    this.$store.detections.prevEventsPage();
  },

  nextPage() {
    const { page, pageSize, total } = this.$store.detections;
    if (page * pageSize < total) {
      this.$store.detections.page++;
      this.$store.detections.fetch();
    }
  },

  prevPage() {
    if (this.$store.detections.page > 1) {
      this.$store.detections.page--;
      this.$store.detections.fetch();
    }
  },

  async deleteDetection(id) {
    if (!confirm('Delete this detection?')) return;
    try {
      await this.$store.detections.deleteDetection(id);
      this.$store.app.addToast('Detection deleted', 'success');
    } catch (e) {
      this.$store.app.addToast(e.message, 'error');
    }
  },

  // Reset all filters to default
  resetFilters() {
    this.$store.detections.filters = {
      plateText: '',
      minConfidence: '',
      startDate: null,
      endDate: null,
      sourceType: ''
    };
    this.$store.detections.page = 1;
    this.refresh();
  },

  // Show detection detail modal (from individual detection)
  showDetail(detection) {
    this.selectedDetection = detection;
    this.selectedEvent = null;
    this.snapshotLoading = true;
    this.snapshotError = false;
    this.showModal = true;
  },
  
  // Show event detail modal (from event card)
  showEventDetail(event) {
    // Create a detection-like object from the event for the modal
    this.selectedDetection = {
      id: event.best_detection_id,
      plate_text: event.plate_text,
      confidence: event.best_confidence,
      timestamp: event.last_seen,
      source_type: event.source_type,
      source_identifier: event.source_identifier,
      frame_number: event.best_frame_number,
      job_id: event.job_id,
      camera_id: event.camera_id,
      snapshot_path: event.best_snapshot_path
    };
    this.selectedEvent = event;
    this.snapshotLoading = true;
    this.snapshotError = false;
    this.showModal = true;
  },

  // Handle snapshot image load success
  onSnapshotLoad() {
    this.snapshotLoading = false;
    this.snapshotError = false;
  },

  // Handle snapshot image load error
  onSnapshotError() {
    this.snapshotLoading = false;
    this.snapshotError = true;
  },

  // Close modal
  closeModal() {
    this.showModal = false;
    this.selectedDetection = null;
    this.selectedEvent = null;
    this.snapshotLoading = false;
    this.snapshotError = false;
  },

  // Add detection to watchlist from modal
  async addToWatchlist(plateText) {
    try {
      await this.$store.watchlist.addPlate(plateText);
      this.$store.app.addToast(`Added ${plateText} to watchlist`, 'success');
    } catch (e) {
      this.$store.app.addToast(e.message, 'error');
    }
  },
  
  // Event-specific methods
  getEventKey(event) {
    return `${event.plate_text}-${event.first_seen}`;
  },
  
  async toggleEventExpanded(event) {
    const key = this.getEventKey(event);
    await this.$store.detections.toggleEventExpanded(key);
  },
  
  isEventExpanded(event) {
    const key = this.getEventKey(event);
    return this.$store.detections.isEventExpanded(key);
  },
  
  getExpandedDetections(event) {
    const key = this.getEventKey(event);
    return this.$store.detections.getExpandedDetections(key);
  },
  
  // Format duration between first and last seen
  formatEventDuration(event) {
    const first = new Date(event.first_seen);
    const last = new Date(event.last_seen);
    const diffMs = last - first;
    const diffSecs = Math.floor(diffMs / 1000);
    
    if (diffSecs < 1) return 'instant';
    if (diffSecs < 60) return `${diffSecs}s`;
    if (diffSecs < 3600) return `${Math.floor(diffSecs / 60)}m ${diffSecs % 60}s`;
    return `${Math.floor(diffSecs / 3600)}h ${Math.floor((diffSecs % 3600) / 60)}m`;
  },
  
  // Format source name from path
  formatSourceName(path) {
    if (!path) return 'Unknown';
    const parts = path.split(/[/\\]/);
    return parts[parts.length - 1];
  },

  // Toggle job expansion (legacy)
  toggleJob(jobId) {
    this.$store.detections.toggleJobExpanded(jobId);
  },

  // Check if job is expanded
  isJobExpanded(jobId) {
    return this.$store.detections.isJobExpanded(jobId);
  },

  // Expand all jobs
  expandAll() {
    this.$store.detections.expandAll();
  },

  // Collapse all jobs
  collapseAll() {
    this.$store.detections.collapseAll();
  },

  // Format job source name for display
  formatJobSource(group) {
    if (group.source_identifier) {
      // Extract filename from path
      const parts = group.source_identifier.split(/[/\\]/);
      return parts[parts.length - 1];
    }
    return group.source_type || 'Unknown';
  },

  // Format video timestamp from frame number (assuming 30fps)
  formatVideoTime(frameNumber, fps = 30) {
    if (!frameNumber) return null;
    const totalSeconds = Math.floor(frameNumber / fps);
    const minutes = Math.floor(totalSeconds / 60);
    const seconds = totalSeconds % 60;
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
  },

  // Get relative time string
  getRelativeTime(timestamp) {
    if (!timestamp) return 'Unknown';
    const now = new Date();
    const then = new Date(timestamp);
    const diffMs = now - then;
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);

    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    if (diffDays < 7) return `${diffDays}d ago`;
    return formatTimeShort(timestamp);
  },

  // Get source type icon class
  getSourceTypeIcon(sourceType) {
    switch (sourceType?.toLowerCase()) {
      case 'video':
        return 'M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z';
      case 'camera':
      case 'stream':
        return 'M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z';
      case 'image':
        return 'M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z';
      default:
        return 'M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z';
    }
  },

  // Export detections to CSV
  async exportCSV() {
    this.exporting = true;
    try {
      // For events view, export events summary
      if (this.viewMode === 'events') {
        const events = this.$store.detections.events;
        if (events.length === 0) {
          this.$store.app.addToast('No events to export', 'warning');
          return;
        }
        
        const headers = ['Plate Number', 'Detection Count', 'Best Confidence', 'First Seen', 'Last Seen', 'Duration', 'Source Type', 'Source'];
        const rows = events.map(e => [
          e.plate_text,
          e.detection_count,
          (e.best_confidence * 100).toFixed(1) + '%',
          new Date(e.first_seen).toLocaleString(),
          new Date(e.last_seen).toLocaleString(),
          this.formatEventDuration(e),
          e.source_type,
          this.formatSourceName(e.source_identifier)
        ]);
        
        const csvContent = [
          headers.join(','),
          ...rows.map(r => r.map(cell => `"${cell}"`).join(','))
        ].join('\n');
        
        this.downloadFile(csvContent, 'detection_events.csv', 'text/csv');
        this.$store.app.addToast(`Exported ${events.length} events to CSV`, 'success');
      } else {
        // Legacy: export raw detections
        const items = this.$store.detections.items;
        if (items.length === 0) {
          this.$store.app.addToast('No detections to export', 'warning');
          return;
        }

        const headers = ['ID', 'Plate Number', 'Confidence', 'Source Type', 'Job ID', 'Frame Number', 'Video Time', 'Timestamp'];
        const rows = items.map(d => [
          d.id,
          d.plate_text,
          (d.confidence * 100).toFixed(1) + '%',
          d.source_type,
          d.job_id || 'N/A',
          d.frame_number || 'N/A',
          this.formatVideoTime(d.frame_number) || 'N/A',
          new Date(d.timestamp).toLocaleString()
        ]);

        const csvContent = [
          headers.join(','),
          ...rows.map(r => r.map(cell => `"${cell}"`).join(','))
        ].join('\n');

        this.downloadFile(csvContent, 'detections.csv', 'text/csv');
        this.$store.app.addToast(`Exported ${items.length} detections to CSV`, 'success');
      }
    } catch (e) {
      this.$store.app.addToast('Export failed: ' + e.message, 'error');
    } finally {
      this.exporting = false;
    }
  },

  // Export detections to JSON
  async exportJSON() {
    this.exporting = true;
    try {
      const data = this.viewMode === 'events' 
        ? this.$store.detections.events 
        : this.$store.detections.items;
        
      if (data.length === 0) {
        this.$store.app.addToast('No data to export', 'warning');
        return;
      }

      const jsonContent = JSON.stringify(data, null, 2);
      const filename = this.viewMode === 'events' ? 'detection_events.json' : 'detections.json';
      this.downloadFile(jsonContent, filename, 'application/json');
      this.$store.app.addToast(`Exported ${data.length} items to JSON`, 'success');
    } catch (e) {
      this.$store.app.addToast('Export failed: ' + e.message, 'error');
    } finally {
      this.exporting = false;
    }
  },

  // Helper to download file
  downloadFile(content, filename, mimeType) {
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  },

  getSnapshotUrl(id) {
    if (!id) return '';
    return api.getSnapshotUrl(id);
  },

  getConfidenceClass(conf) {
    if (conf >= 0.9) return 'confidence-high';
    if (conf >= 0.7) return 'confidence-medium';
    return 'confidence-low';
  },

  getConfidenceBadge(conf) {
    if (conf >= 0.9) return 'badge-success';
    if (conf >= 0.7) return 'badge-warning';
    return 'badge-danger';
  },

  formatTime,
  formatTimeShort
});
