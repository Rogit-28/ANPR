import api from '../api/client.js';

document.addEventListener('alpine:init', () => {
  // Jobs Store
  Alpine.store('jobs', {
    items: [],
    activeJob: null,
    isLoading: false,
    _isFetching: false,  // Prevent concurrent fetches
    
    async fetchJobs() {
      // Prevent concurrent fetches
      if (this._isFetching) return;
      this._isFetching = true;
      this.isLoading = true;
      
      try {
        const response = await api.getJobs({ limit: 10 });
        this.items = response.jobs || [];
      } catch (e) {
        console.error('Failed to fetch jobs', e);
      } finally {
        this.isLoading = false;
        this._isFetching = false;
      }
    },
    
    async cancelJob(jobId) {
      try {
        await api.cancelJob(jobId);
        await this.fetchJobs();
      } catch (e) {
        console.error('Failed to cancel job', e);
      }
    },
    
    async deleteJob(jobId) {
      try {
        await api.deleteJob(jobId);
        this.items = this.items.filter(j => j.id !== jobId);
      } catch (e) {
        console.error('Failed to delete job', e);
      }
    },
    
    async markFailed(jobId) {
      try {
        // Use cancel endpoint to stop job, then update locally to failed status
        await api.cancelJob(jobId);
        const job = this.items.find(j => j.id === jobId);
        if (job) {
          job.status = 'failed';
        }
        Alpine.store('app').addToast('Job marked as failed', 'warning');
      } catch (e) {
        console.error('Failed to mark job as failed', e);
        Alpine.store('app').addToast('Failed to mark job', 'error');
      }
    }
  });
});
