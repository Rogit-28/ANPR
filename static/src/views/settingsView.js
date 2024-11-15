export default () => ({
  // Local UI state
  showResetConfirm: false,
  
  async init() {
    // Initialize settings store
    await this.$store.settings.init();
  },
  
  // Computed getters that delegate to store
  get systemInfo() {
    return this.$store.settings.systemInfo;
  },
  
  get config() {
    return this.$store.settings.config;
  },
  
  get isDirty() {
    return this.$store.settings.isDirty;
  },
  
  get isLoading() {
    return this.$store.settings.isLoading;
  },
  
  get isSaving() {
    return this.$store.settings.isSaving;
  },
  
  get isRestarting() {
    return this.$store.settings.isRestarting;
  },
  
  get restartRequired() {
    return this.$store.settings.restartRequired;
  },
  
  get validationErrors() {
    return this.$store.settings.validationErrors;
  },
  
  // Actions
  async saveSettings() {
    await this.$store.settings.saveConfig();
  },
  
  discardChanges() {
    this.$store.settings.discardChanges();
  },
  
  async resetToDefaults() {
    await this.$store.settings.resetConfig();
  },
  
  async testAlert() {
    await this.$store.settings.testAlert();
  },
  
  async restartServer() {
    await this.$store.settings.restartServer();
  },
  
  // Format helpers
  formatUptime(seconds) {
    return this.$store.settings.formatUptime(seconds);
  },
  
  formatGpuMemory() {
    const info = this.$store.settings.systemInfo;
    if (!info.gpu_memory_used || !info.gpu_memory_total) return 'N/A';
    return `${info.gpu_memory_used} / ${info.gpu_memory_total}`;
  },
  
  formatGpuUtilization() {
    const util = this.$store.settings.systemInfo.gpu_utilization;
    if (util === null || util === undefined) return 'N/A';
    return `${util}%`;
  },
  
  formatOnnxProviders() {
    const providers = this.$store.settings.systemInfo.onnx_providers || [];
    if (providers.length === 0) return 'None';
    // Shorten provider names for display
    return providers.map(p => p.replace('ExecutionProvider', '')).join(', ');
  },
  
  // Validation helpers
  hasError(field) {
    return this.validationErrors.some(e => e.toLowerCase().includes(field.toLowerCase()));
  },
  
  getError(field) {
    return this.validationErrors.find(e => e.toLowerCase().includes(field.toLowerCase())) || '';
  },
  
  // Slider display helpers
  formatConfidence(value) {
    return (value * 100).toFixed(0) + '%';
  }
});
