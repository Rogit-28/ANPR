import api from '../api/client.js';

document.addEventListener('alpine:init', () => {
  // Alerts Store with Real-Time SSE (Server-Sent Events) Support
  Alpine.store('alerts', {
    enabled: true,
    history: [],
    stats: null,
    
    // SSE connection state
    eventSource: null,
    connected: false,
    reconnectAttempts: 0,
    maxReconnectAttempts: 10,
    reconnectDelay: 2000,
    _reconnectTimeout: null,
    serverShutdown: false,  // Track if server is shutting down
    
    // Browser notification state
    notificationPermission: typeof Notification !== 'undefined' ? Notification.permission : 'denied',
    
    // Audio for watchlist alerts
    _alertSound: null,
    soundEnabled: localStorage.getItem('alertSoundEnabled') !== 'false',
    
    // Initialize the alerts system
    init() {
      console.log('[Alerts] Initializing alerts system...');
      
      // Setup notification permission state
      this._setupNotificationPermission();
      
      // Create alert sound
      this._initAlertSound();
      
      // Connect to SSE stream
      this.connect();
      
      // Fetch initial data
      this.fetchStats();
      this.fetchHistory();
    },
    
    // SSE Connection Management
    connect() {
      if (this.eventSource) {
        // Already have a connection, close it first
        this.eventSource.close();
      }
      
      const sseUrl = '/api/v1/alerts/stream';
      console.log('[Alerts] Connecting to SSE:', sseUrl);
      
      try {
        this.eventSource = new EventSource(sseUrl);
        
        // Connection opened
        this.eventSource.onopen = (e) => {
          console.log('[Alerts] SSE connection opened, readyState:', this.eventSource.readyState);
          
          // Show reconnected toast if we were previously disconnected due to shutdown
          if (this.serverShutdown) {
            const appStore = Alpine.store('app');
            if (appStore) {
              appStore.addToast('Reconnected to alert server', 'success');
            }
            // Refresh data after reconnect
            this.fetchStats();
            this.fetchHistory();
          }
          
          this.connected = true;
          this.reconnectAttempts = 0;
          this.serverShutdown = false;  // Reset shutdown flag
        };
        
        // Handle 'connected' event from server
        this.eventSource.addEventListener('connected', (e) => {
          try {
            const data = JSON.parse(e.data);
            console.log('[Alerts] Server confirmed connection:', data);
          } catch (err) {
            console.error('[Alerts] Failed to parse connected event:', err);
          }
        });
        
        // Handle 'alert' events
        this.eventSource.addEventListener('alert', (e) => {
          try {
            const alertData = JSON.parse(e.data);
            console.log('[Alerts] Received alert:', alertData);
            this._handleAlert(alertData);
          } catch (err) {
            console.error('[Alerts] Failed to parse alert:', err);
          }
        });
        
        // Handle 'shutdown' event from server
        this.eventSource.addEventListener('shutdown', (e) => {
          try {
            const data = JSON.parse(e.data);
            console.log('[Alerts] Server shutdown event received:', data);
            this.serverShutdown = true;
            this.connected = false;
            
            // Show notification to user
            const appStore = Alpine.store('app');
            if (appStore) {
              appStore.addToast('Server is shutting down. Alerts will reconnect when available.', 'warning');
            }
            
            // Close connection gracefully
            if (this.eventSource) {
              this.eventSource.close();
              this.eventSource = null;
            }
            
            // Schedule reconnect with longer delay for server restart
            this._scheduleReconnect(5000);
          } catch (err) {
            console.error('[Alerts] Failed to parse shutdown event:', err);
          }
        });
        
        // Generic message handler (for debugging)
        this.eventSource.onmessage = (e) => {
          console.log('[Alerts] SSE message:', e.data);
        };
        
        // Handle errors
        this.eventSource.onerror = (error) => {
          console.error('[Alerts] SSE error, readyState:', this.eventSource?.readyState, error);
          this.connected = false;
          
          // Check readyState: 0=CONNECTING, 1=OPEN, 2=CLOSED
          if (this.eventSource?.readyState === EventSource.CLOSED) {
            console.log('[Alerts] SSE connection closed, scheduling reconnect...');
            this.eventSource = null;
            this._scheduleReconnect();
          }
          // If CONNECTING, EventSource will auto-retry
        };
        
      } catch (error) {
        console.error('[Alerts] Failed to create EventSource:', error);
        this._scheduleReconnect();
      }
    },
    
    disconnect() {
      if (this._reconnectTimeout) {
        clearTimeout(this._reconnectTimeout);
        this._reconnectTimeout = null;
      }
      if (this.eventSource) {
        this.eventSource.close();
        this.eventSource = null;
      }
      this.connected = false;
      console.log('[Alerts] SSE disconnected');
    },
    
    _scheduleReconnect(overrideDelay = null) {
      // If server is shutting down, use longer delays and more attempts
      const maxAttempts = this.serverShutdown ? 30 : this.maxReconnectAttempts;
      
      if (this.reconnectAttempts >= maxAttempts) {
        console.warn('[Alerts] Max reconnection attempts reached');
        const appStore = Alpine.store('app');
        if (appStore) {
          appStore.addToast('Unable to connect to alert server. Please refresh the page.', 'error');
        }
        return;
      }
      
      // Use override delay or calculate exponential backoff
      let delay;
      if (overrideDelay) {
        delay = overrideDelay;
      } else {
        // Longer delays if server is shutting down (waiting for restart)
        const baseDelay = this.serverShutdown ? 3000 : this.reconnectDelay;
        delay = Math.min(
          baseDelay * Math.pow(1.5, this.reconnectAttempts) + Math.random() * 1000,
          this.serverShutdown ? 60000 : 30000  // Max 60s for shutdown, 30s otherwise
        );
      }
      
      console.log(`[Alerts] Reconnecting in ${Math.round(delay / 1000)}s (attempt ${this.reconnectAttempts + 1}${this.serverShutdown ? ', server shutdown' : ''})`);
      
      this._reconnectTimeout = setTimeout(() => {
        this.reconnectAttempts++;
        this.connect();
      }, delay);
    },
    
    // Handle incoming alert
    _handleAlert(alertData) {
      // Add to history (at the beginning)
      this.history.unshift({
        timestamp: alertData.timestamp,
        type: alertData.alert_type,
        plate: alertData.plate_text,
        camera: alertData.camera_id,
        confidence: alertData.confidence,
        message: alertData.message,
        metadata: alertData.metadata
      });
      
      // Keep history to reasonable size
      if (this.history.length > 100) {
        this.history = this.history.slice(0, 100);
      }
      
      // Show toast notification
      this._showAlertToast(alertData);
      
      // Send browser notification
      this._sendBrowserNotification(alertData);
      
      // Play sound for watchlist matches
      if (alertData.alert_type === 'watchlist_match') {
        this._playAlertSound();
      }
      
      // Update stats
      this.fetchStats();
    },
    
    // Toast Notification - Enhanced with rich formatting
    _showAlertToast(alertData) {
      const appStore = Alpine.store('app');
      if (!appStore) return;
      
      const isWatchlist = alertData.alert_type === 'watchlist_match';
      const type = isWatchlist ? 'alert-watchlist' : 'alert-detection';
      
      // Format confidence as percentage
      const confidence = alertData.confidence 
        ? (alertData.confidence * 100).toFixed(0)
        : null;
      
      // Format timestamp - show only time (HH:MM:SS) in 24hr format
      let timeStr = '';
      if (alertData.timestamp) {
        try {
          const date = new Date(alertData.timestamp);
          timeStr = date.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
        } catch (e) {
          timeStr = '';
        }
      }
      
      // Build structured message object for rich toast
      const toastData = {
        type: type,
        alertType: alertData.alert_type,
        plate: alertData.plate_text,
        confidence: confidence,
        camera: alertData.camera_id || null,
        time: timeStr,
        message: alertData.message || '',
        isWatchlist: isWatchlist
      };
      
      // Use rich alert toast
      this._addAlertToast(appStore, toastData);
    },
    
    // Add enhanced alert toast with rich content
    _addAlertToast(appStore, data) {
      const id = Date.now();
      const duration = data.isWatchlist ? 10000 : 5000; // Watchlist alerts stay longer
      
      // Create a rich toast object
      const toast = {
        id,
        type: data.type,
        isRichAlert: true,
        alertType: data.alertType,
        plate: data.plate,
        confidence: data.confidence,
        camera: data.camera,
        time: data.time,
        message: data.message,
        isWatchlist: data.isWatchlist,
        visible: true
      };
      
      appStore.toasts.push(toast);
      setTimeout(() => appStore.removeToast(id), duration);
    },
    
    // Browser Notification
    _setupNotificationPermission() {
      if (typeof Notification !== 'undefined') {
        this.notificationPermission = Notification.permission;
      }
    },
    
    async requestNotificationPermission() {
      if (typeof Notification === 'undefined') {
        console.warn('[Alerts] Browser does not support notifications');
        return false;
      }
      
      if (Notification.permission === 'granted') {
        this.notificationPermission = 'granted';
        return true;
      }
      
      if (Notification.permission !== 'denied') {
        const permission = await Notification.requestPermission();
        this.notificationPermission = permission;
        return permission === 'granted';
      }
      
      return false;
    },
    
    _sendBrowserNotification(alertData) {
      if (this.notificationPermission !== 'granted') return;
      if (document.hasFocus()) return; // Don't notify if tab is focused
      
      const isWatchlist = alertData.alert_type === 'watchlist_match';
      
      const title = isWatchlist 
        ? `WATCHLIST MATCH` 
        : `ANPR Alert`;
      
      const body = `Plate: ${alertData.plate_text}\n` +
        (alertData.camera_id ? `Camera: ${alertData.camera_id}\n` : '') +
        (alertData.confidence ? `Confidence: ${(alertData.confidence * 100).toFixed(0)}%` : '');
      
      try {
        const notification = new Notification(title, {
          body: body,
          icon: '/static/favicon.ico',
          tag: `anpr-${alertData.plate_text}-${Date.now()}`,
          requireInteraction: isWatchlist,
        });
        
        notification.onclick = () => {
          window.focus();
          notification.close();
          const appStore = Alpine.store('app');
          if (appStore) {
            appStore.setView('alerts');
          }
        };
        
        if (!isWatchlist) {
          setTimeout(() => notification.close(), 5000);
        }
      } catch (e) {
        console.error('[Alerts] Failed to send browser notification:', e);
      }
    },
    
    // Alert Sound
    _initAlertSound() {
      // Simple beep tone encoded as base64 WAV
      const audioData = 'data:audio/wav;base64,UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YVoGAACAgICAgICAgICAgICAgICAgICAgICAgH9/f3+AgIB/f39/gICAgH9/f4CAgH9/f4CAgICAgICAgICBgYKCgoODhISFhYaGh4eHh4eHh4aGhYWEg4OCgYGAgH9/fn59fX19fX19fXx8fX19fn5/f4CAgYGCgoODhISFhYaGhoaHh4aGhoWFhISDgoKBgYB/f359fXx8e3t7e3t7e3t8fHx9fX5+f4CAgYGCgoODhIWFhoaGhoaGhoWFhISEg4KCgYGAf399fXx8e3t7enp6enp7e3t7fHx9fX5/f4CAgYGCgoOEhIWFhYWFhYWFhYWEhISEg4OCgYGAf39+fXx8e3t6enp6enp6ent7e3x8fX5+f4CAgYGCgoODhISEhYWFhYWFhISEg4ODgoKBgIB/fn59fHx7e3p6eXl5eXp6ent7fHx9fn5/gICBgYKCg4ODhISEhISEhISEhIODg4KCgYGAf39+fX18e3t6enl5eXl5eXp6e3t7fH19fn9/gICBgYKCg4ODhISEhISEhISEg4ODgoKBgYCAf35+fXx8e3t6eXl5eXl5enp6e3t8fH1+fn+AgIGBgoKDg4ODhISEhISEhISDg4OCgoKBgIB/fn59fHx7e3p6eXl5eXl5enp7e3x8fX5+f4CAgIGBgoKDg4ODhISEhISEg4ODg4KCgYGAgH9+fn18fHt7enp5eXl5eXl6ent7fHx9fX5/f4CAgYGBgoKDg4ODg4ODg4ODg4KCgoKBgYCAf359fXx8e3t6enl5eXl5eXp6e3t7fHx9fn5/gICAhoaGhoaGhoaGhoaGhoaGhoaGhoaGhoaGhoaGhoaGhoaGhoaGhoaGhoaGhoaGhoaGhoaGhoaGhoaGhoaGhoaGhoaGhoaGhoaGhoaGhoaGhoaGhoaGhoaGhoaGhoaGhoaGhoaGhoaGhoaGhoaGhoaGhoaGhoaGhoaGhoaGhoaGhoaGhoaGhoaGhoaGhoaGhoaGhoaGhoaGhoaGhoaGhoaGhoaGhoaGhoaGhoaGhoaGhoaGhoaGhoaGhoaGhoaGhoaGhoaGhoaGhoaGhoaGhoaGhoaGhoaGhoaGhoaGhoaGhoaGhoaGhoaGhoaGhoaGhoaGhoaGhoaGhoaGhoaGhoaGhoaGhoaGhoaGhoaGhoaGhoaGhoaGhoaGhoaGhoaGhoaGhoaGhg==';
      
      this._alertSound = new Audio(audioData);
      this._alertSound.volume = 0.5;
    },
    
    _playAlertSound() {
      if (!this.soundEnabled || !this._alertSound) return;
      
      try {
        this._alertSound.currentTime = 0;
        this._alertSound.play().catch(e => {
          console.debug('[Alerts] Could not play alert sound:', e.message);
        });
      } catch (e) {
        console.error('[Alerts] Error playing alert sound:', e);
      }
    },
    
    toggleSound() {
      this.soundEnabled = !this.soundEnabled;
      localStorage.setItem('alertSoundEnabled', this.soundEnabled);
      
      if (this.soundEnabled) {
        this._playAlertSound();
      }
    },
    
    // API Methods
    async fetchStats() {
      try {
        this.stats = await api.getAlertStats();
        this.enabled = this.stats.enabled !== false;
      } catch (e) {
        console.error('Failed to fetch alert stats', e);
      }
    },
    
    async fetchHistory() {
      try {
        const response = await api.getAlertHistory();
        this.history = response.alerts || [];
      } catch (e) {
        console.error('Failed to fetch alert history', e);
        this.history = [];
      }
    },
    
    async toggle() {
      if (this.enabled) {
        await api.disableAlerts();
      } else {
        await api.enableAlerts();
      }
      this.enabled = !this.enabled;
    },
    
    async acknowledgeAlert(alertId) {
      try {
        await api.acknowledgeAlert(alertId);
        // Update local history
        const alert = this.history.find(a => a.id === alertId);
        if (alert) {
          alert.acknowledged = true;
        }
        return true;
      } catch (e) {
        console.error('[Alerts] Failed to acknowledge alert:', e);
        throw e;
      }
    },
    
    async deleteAlert(alertId) {
      try {
        await api.deleteAlert(alertId);
        // Remove from local history
        this.history = this.history.filter(a => a.id !== alertId);
        return true;
      } catch (e) {
        console.error('[Alerts] Failed to delete alert:', e);
        throw e;
      }
    }
  });
});
