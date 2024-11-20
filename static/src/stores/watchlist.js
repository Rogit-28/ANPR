import api from '../api/client.js';

document.addEventListener('alpine:init', () => {
  // Watchlist Store
  Alpine.store('watchlist', {
    items: [],
    isLoading: false,
    isClearing: false,
    
    async fetch() {
      this.isLoading = true;
      try {
        this.items = await api.getWatchlist();
      } catch (e) {
        console.error('Failed to fetch watchlist', e);
      } finally {
        this.isLoading = false;
      }
    },
    
    async addPlate(plate) {
      await api.addToWatchlist(plate);
      await this.fetch();
    },
    
    async removePlate(plate) {
      await api.removeFromWatchlist(plate);
      this.items = this.items.filter(p => p.plate_text !== plate);
    },
    
    async clearAll() {
      this.isClearing = true;
      try {
        const result = await api.clearWatchlist();
        if (result.success) {
          this.items = [];
          Alpine.store('app').addToast(`Cleared ${result.cleared_count} entries from watchlist`, 'success');
        }
      } catch (e) {
        console.error('Failed to clear watchlist', e);
        Alpine.store('app').addToast('Failed to clear watchlist: ' + e.message, 'error');
      } finally {
        this.isClearing = false;
      }
    }
  });
});
