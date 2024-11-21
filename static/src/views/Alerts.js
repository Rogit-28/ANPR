import { formatTime } from '../utils.js';

export default () => ({
  statusFilter: 'all', // all, unacknowledged, acknowledged

  async init() {
    this.$store.alerts.fetchStats();
    this.$store.alerts.fetchHistory();
  },

  get filteredHistory() {
    let filtered = this.$store.alerts.history;
    
    // Filter by acknowledged status
    if (this.statusFilter === 'unacknowledged') {
      filtered = filtered.filter(a => !a.acknowledged);
    } else if (this.statusFilter === 'acknowledged') {
      filtered = filtered.filter(a => a.acknowledged);
    }
    
    return filtered;
  },
  
  get unacknowledgedCount() {
    return this.$store.alerts.history.filter(a => !a.acknowledged).length;
  },

  async toggleAlerts() {
    try {
      await this.$store.alerts.toggle();
      const enabled = this.$store.alerts.enabled;
      this.$store.app.addToast(
        enabled ? 'Alerts enabled' : 'Alerts disabled', 
        enabled ? 'success' : 'info'
      );
    } catch (e) {
      this.$store.app.addToast(e.message, 'error');
    }
  },
  
  async acknowledgeAlert(alertId) {
    try {
      await this.$store.alerts.acknowledgeAlert(alertId);
      this.$store.app.addToast('Alert acknowledged', 'success');
    } catch (e) {
      this.$store.app.addToast('Failed to acknowledge alert: ' + e.message, 'error');
    }
  },
  
  async deleteAlert(alertId) {
    if (!confirm('Delete this alert from history?')) return;
    try {
      await this.$store.alerts.deleteAlert(alertId);
      this.$store.app.addToast('Alert deleted', 'success');
    } catch (e) {
      this.$store.app.addToast('Failed to delete alert: ' + e.message, 'error');
    }
  },
  
  async acknowledgeAll() {
    const unacknowledged = this.$store.alerts.history.filter(a => !a.acknowledged);
    if (unacknowledged.length === 0) {
      this.$store.app.addToast('No alerts to acknowledge', 'info');
      return;
    }
    
    if (!confirm(`Acknowledge all ${unacknowledged.length} unacknowledged alerts?`)) return;
    
    let count = 0;
    for (const alert of unacknowledged) {
      try {
        await this.$store.alerts.acknowledgeAlert(alert.id);
        count++;
      } catch (e) {
        console.error('Failed to acknowledge alert:', alert.id, e);
      }
    }
    
    this.$store.app.addToast(`Acknowledged ${count} alerts`, 'success');
  },

  formatTime
});
