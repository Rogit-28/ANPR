import api from '../api/client.js';

document.addEventListener('alpine:init', () => {
  
  // Settings Store
  Alpine.store('settings', {
    // System info (read-only)
    systemInfo: {
      api_version: '',
      python_version: '',
      cuda_available: false,
      cuda_version: null,
      gpu_name: null,
      gpu_memory_total: null,
      gpu_memory_used: null,
      gpu_utilization: null,
      database_path: '',
      config_file_path: '',
      uptime: null,
      uptime_seconds: null,
      onnx_providers: []
    },
    
    // Editable config
    config: {
      alerts_enabled: true,
      desktop_notifications: true,
      yolo_confidence: 0.4,
      yolo_weights_path: '',
      min_ocr_confidence: 0.7,
      plate_validation_enabled: true
    },
    
    // Original config for dirty checking
    originalConfig: {},
    
    // UI State
    isLoading: false,
    isSaving: false,
    isRestarting: false,
    restartRequired: false,
    lastError: null,
    lastSaveMessage: null,
    validationErrors: [],
    
    // Computed: check if config has changed
    get isDirty() {
      return JSON.stringify(this.config) !== JSON.stringify(this.originalConfig);
    },
    
    // Initialize store
    async init() {
      await this.loadSystemInfo();
      await this.loadConfig();
    },
    
    // Load system information
    async loadSystemInfo() {
      try {
        const info = await api.getSystemInfo();
        this.systemInfo = info;
      } catch (e) {
        console.error('Failed to load system info:', e);
        this.lastError = e.message;
      }
    },
    
    // Load configuration
    async loadConfig() {
      this.isLoading = true;
      try {
        const config = await api.getConfig();
        this.config = { ...config };
        this.originalConfig = { ...config };
        this.validationErrors = [];
        this.lastError = null;
      } catch (e) {
        console.error('Failed to load config:', e);
        this.lastError = e.message;
      } finally {
        this.isLoading = false;
      }
    },
    
    // Save configuration
    async saveConfig() {
      if (!this.isDirty) return;
      
      this.isSaving = true;
      this.lastError = null;
      this.lastSaveMessage = null;
      this.validationErrors = [];
      
      try {
        // Only send changed fields
        const changes = {};
        for (const key in this.config) {
          if (this.config[key] !== this.originalConfig[key]) {
            changes[key] = this.config[key];
          }
        }
        
        const result = await api.updateConfig(changes);
        
        if (result.success) {
          this.originalConfig = { ...this.config };
          this.restartRequired = result.restart_required;
          this.lastSaveMessage = result.message;
          
          if (result.restart_required) {
            Alpine.store('app').addToast('Settings saved. Restart required to apply changes.', 'warning');
          } else {
            Alpine.store('app').addToast('Settings saved successfully', 'success');
          }
        } else {
          this.validationErrors = result.validation_errors || [];
          this.lastError = result.message;
          Alpine.store('app').addToast(result.message || 'Failed to save settings', 'error');
        }
      } catch (e) {
        console.error('Failed to save config:', e);
        this.lastError = e.message;
        Alpine.store('app').addToast('Failed to save settings: ' + e.message, 'error');
      } finally {
        this.isSaving = false;
      }
    },
    
    // Reset to defaults
    async resetConfig() {
      if (!confirm('Are you sure you want to reset all settings to defaults? This cannot be undone.')) {
        return;
      }
      
      this.isSaving = true;
      try {
        const result = await api.resetConfig();
        
        if (result.success) {
          this.restartRequired = true;
          Alpine.store('app').addToast('Settings reset to defaults. Restart required.', 'warning');
          // Reload config to show defaults
          await this.loadConfig();
        }
      } catch (e) {
        console.error('Failed to reset config:', e);
        Alpine.store('app').addToast('Failed to reset settings: ' + e.message, 'error');
      } finally {
        this.isSaving = false;
      }
    },
    
    // Discard changes
    discardChanges() {
      this.config = { ...this.originalConfig };
      this.validationErrors = [];
      this.lastError = null;
    },
    
    // Trigger test alert
    async testAlert() {
      try {
        const result = await api.testAlert();
        if (result.success) {
          Alpine.store('app').addToast('Test alert triggered!', 'success');
        } else {
          Alpine.store('app').addToast('Failed to trigger test alert', 'error');
        }
      } catch (e) {
        console.error('Failed to trigger test alert:', e);
        Alpine.store('app').addToast('Failed to trigger test alert: ' + e.message, 'error');
      }
    },
    
    // Restart server
    async restartServer() {
      if (!confirm('Are you sure you want to restart the server? Active connections will be interrupted.')) {
        return;
      }
      
      this.isRestarting = true;
      try {
        await api.restartServer();
        Alpine.store('app').addToast('Server restarting... Please wait.', 'info');
        
        // Poll for server to come back up
        this._waitForServerRestart();
      } catch (e) {
        console.error('Failed to restart server:', e);
        Alpine.store('app').addToast('Failed to restart server: ' + e.message, 'error');
        this.isRestarting = false;
      }
    },
    
    // Wait for server to come back after restart
    async _waitForServerRestart() {
      let attempts = 0;
      const maxAttempts = 30; // 30 seconds max
      
      const checkServer = async () => {
        attempts++;
        try {
          await api.getHealth();
          // Server is back!
          this.isRestarting = false;
          this.restartRequired = false;
          Alpine.store('app').addToast('Server restarted successfully!', 'success');
          // Reload data
          await this.loadSystemInfo();
          await this.loadConfig();
        } catch (e) {
          if (attempts < maxAttempts) {
            setTimeout(checkServer, 1000);
          } else {
            this.isRestarting = false;
            Alpine.store('app').addToast('Server restart timed out. Please check manually.', 'error');
          }
        }
      };
      
      // Start checking after a short delay
      setTimeout(checkServer, 2000);
    },
    
    // Format uptime nicely
    formatUptime(seconds) {
      if (!seconds) return 'N/A';
      const days = Math.floor(seconds / 86400);
      const hours = Math.floor((seconds % 86400) / 3600);
      const minutes = Math.floor((seconds % 3600) / 60);
      if (days > 0) return `${days}d ${hours}h`;
      if (hours > 0) return `${hours}h ${minutes}m`;
      return `${minutes}m`;
    }
  });
});
