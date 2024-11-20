import api from '../api/client.js';

document.addEventListener('alpine:init', () => {
  
  // Global App Store
  Alpine.store('app', {
    // UI State
    sidebarOpen: true,
    darkMode: localStorage.getItem('darkMode') === 'true',
    currentView: 'dashboard',
    toasts: [],
    
    // System State
    health: null,
    isOnline: false,
    stats: {},
    watchlistCount: 0,
    activeJobs: [],
    
    // Polling control
    _statsPollingId: null,
    _jobsPollingId: null,
    _isPollingStats: false,
    _isPollingJobs: false,
    
    // Actions
    toggleSidebar() {
      this.sidebarOpen = !this.sidebarOpen;
    },
    toggleDarkMode() {
      this.darkMode = !this.darkMode;
      localStorage.setItem('darkMode', this.darkMode);
      document.documentElement.classList.toggle('dark', this.darkMode);
    },
    setView(view) {
      this.currentView = view;
      window.location.hash = `#/${view === 'dashboard' ? '' : view}`;
    },
    
    // Navigate to results with optional filters
    navigateToResults(options = {}) {
      const store = Alpine.store('detections');
      
      // Reset filters first
      store.filters.plateText = '';
      store.filters.minConfidence = 0;
      store.filters.startTime = null;
      store.filters.endTime = null;
      store.filters.sourceIdentifier = null;
      
      if (options.today) {
        // Set filter to today's date (start of day in ISO format)
        const today = new Date();
        today.setHours(0, 0, 0, 0);
        store.filters.startTime = today.toISOString();
      }
      
      store.page = 1;
      
      // Navigate to results view
      this.setView('results');
      
      // Force a fetch after a short delay to ensure view is ready
      setTimeout(() => {
        store.fetch();
      }, 50);
    },
    
    // Navigate to results with job filter (grouped view)
    navigateToJobResults(jobId) {
      const store = Alpine.store('detections');
      
      // Reset filters
      store.filters.plateText = '';
      store.filters.minConfidence = 0;
      store.filters.startTime = null;
      store.filters.endTime = null;
      store.filters.sourceIdentifier = null;
      store.page = 1;
      
      // Store the job_id to filter by
      store.filterByJobId = jobId;
      
      // Navigate to results view
      this.setView('results');
      
      // Force grouped view and fetch
      setTimeout(() => {
        // Trigger the Results view to switch to grouped mode and filter by job
        window.dispatchEvent(new CustomEvent('view-job-results', { detail: { jobId } }));
      }, 50);
    },
    
    // Navigate to results and show a specific detection in modal
    navigateToDetection(detection) {
      // Store the detection to show
      this.pendingDetectionModal = detection;
      
      // Navigate to results view
      this.setView('results');
      
      // Dispatch event to open modal after view loads
      setTimeout(() => {
        window.dispatchEvent(new CustomEvent('show-detection-modal', { detail: { detection } }));
      }, 100);
    },
    
    // Toast Notification System
    addToast(message, type = 'info') {
      const id = Date.now();
      // Determine duration based on type (errors/warnings stay longer)
      const duration = (type === 'error' || type === 'warning') ? 5000 : 3000;
      this.toasts.push({ id, message, type, visible: true });
      setTimeout(() => this.removeToast(id), duration);
    },
    removeToast(id) {
      const toast = this.toasts.find(t => t.id === id);
      if (toast) {
        toast.visible = false;
        setTimeout(() => {
          this.toasts = this.toasts.filter(t => t.id !== id);
        }, 300);
      }
    },

    async checkHealth() {
      try {
        const health = await api.getHealth();
        this.health = health;
        this.isOnline = true;
        return health;
      } catch (e) {
        this.isOnline = false;
        return null;
      }
    },

    async pollStats() {
      // Prevent concurrent polling
      if (this._isPollingStats) return;
      this._isPollingStats = true;

      try {
        const data = await api.getStats();
        this.stats = data;
        
        // Also update watchlist count occasionally
        const watchlist = await api.getWatchlist();
        this.watchlistCount = watchlist.length;
      } catch (e) {
        // Silent fail - don't spam console
      } finally {
        this._isPollingStats = false;
      }
      
      // Schedule next poll
      this._statsPollingId = setTimeout(() => this.pollStats(), 60000);
    },

    async pollJobs() {
      // Prevent concurrent polling
      if (this._isPollingJobs) return;
      this._isPollingJobs = true;

      try {
        const response = await api.getJobs({ limit: 5 });
        this.activeJobs = (response.jobs || []).filter(j => 
          ['processing', 'pending', 'normalizing', 'annotating'].includes(j.status)
        );
      } catch (e) {
        // Silent fail - don't spam console
      } finally {
        this._isPollingJobs = false;
      }
      
      // Schedule next poll (slower if no active jobs)
      const interval = this.activeJobs.length > 0 ? 3000 : 10000;
      this._jobsPollingId = setTimeout(() => this.pollJobs(), interval);
    },

    stopPolling() {
      if (this._statsPollingId) {
        clearTimeout(this._statsPollingId);
        this._statsPollingId = null;
      }
      if (this._jobsPollingId) {
        clearTimeout(this._jobsPollingId);
        this._jobsPollingId = null;
      }
    }
  });
});
