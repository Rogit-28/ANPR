import api from '../api/client.js';

document.addEventListener('alpine:init', () => {
  // Stream Store
  Alpine.store('stream', {
    isActive: false,
    status: null,
    url: '',
    detections: [],
    
    async start(url) {
      this.url = url;
      await api.startStream(url);
      this.isActive = true;
    },
    
    async stop() {
      await api.stopStream();
      this.isActive = false;
      this.status = null;
    },
    
    async getStatus() {
        return await api.getStreamStatus();
    }
  });
});
