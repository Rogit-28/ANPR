export default () => ({
  newPlate: '',
  validationError: '',
  clearConfirmPending: false,
  clearConfirmTimeout: null,

  async init() {
    this.$store.watchlist.fetch();
  },

  async addPlate() {
    // Basic format validation
    const plate = this.newPlate.toUpperCase().replace(/[^A-Z0-9]/g, '');
    if (plate.length < 8) {
      this.validationError = 'Invalid format (too short)';
      return;
    }
    
    try {
      await this.$store.watchlist.addPlate(plate);
      this.newPlate = '';
      this.validationError = '';
      this.$store.app.addToast(`Added ${plate} to watchlist`, 'success');
    } catch (e) {
      this.validationError = e.message;
    }
  },

  async removePlate(plate) {
    if (!confirm(`Remove ${plate} from watchlist?`)) return;
    try {
      await this.$store.watchlist.removePlate(plate);
      this.$store.app.addToast(`Removed ${plate}`, 'info');
    } catch (e) {
      this.$store.app.addToast(e.message, 'error');
    }
  },

  // 2-step clear all with timeout
  clearAllClick() {
    if (this.clearConfirmPending) {
      // Second click - actually clear
      this.clearConfirmPending = false;
      if (this.clearConfirmTimeout) {
        clearTimeout(this.clearConfirmTimeout);
        this.clearConfirmTimeout = null;
      }
      this.$store.watchlist.clearAll();
    } else {
      // First click - show confirmation
      this.clearConfirmPending = true;
      this.clearConfirmTimeout = setTimeout(() => {
        this.clearConfirmPending = false;
        this.clearConfirmTimeout = null;
      }, 3000); // 3 second timeout
    }
  },

  cancelClear() {
    this.clearConfirmPending = false;
    if (this.clearConfirmTimeout) {
      clearTimeout(this.clearConfirmTimeout);
      this.clearConfirmTimeout = null;
    }
  }
});
