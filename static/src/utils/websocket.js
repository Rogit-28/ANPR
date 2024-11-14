/**
 * WebSocket handler for real-time ANPR alerts and notifications.
 */

const WS_RECONNECT_DELAY = 3000;
const WS_MAX_RECONNECT_ATTEMPTS = 10;

class WebSocketHandler {
  constructor(endpoint, options = {}) {
    this.endpoint = endpoint;
    this.options = options;
    this.ws = null;
    this.reconnectAttempts = 0;
    this.isConnected = false;
    this.listeners = new Map();
    this.messageQueue = [];
  }

  connect() {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      return;
    }

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host;
    const url = `${protocol}//${host}${this.endpoint}`;

    try {
      this.ws = new WebSocket(url);

      this.ws.onopen = () => {
        console.log(`[WebSocket] Connected to ${this.endpoint}`);
        this.isConnected = true;
        this.reconnectAttempts = 0;
        this.emit('connected');

        // Send any queued messages
        while (this.messageQueue.length > 0) {
          const msg = this.messageQueue.shift();
          this.send(msg);
        }
      };

      this.ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          this.emit('message', data);

          // Emit specific event type if present
          if (data.type) {
            this.emit(data.type, data);
          }
        } catch (e) {
          console.error('[WebSocket] Failed to parse message:', e);
        }
      };

      this.ws.onclose = (event) => {
        console.log(`[WebSocket] Disconnected from ${this.endpoint}`, event.code, event.reason);
        this.isConnected = false;
        this.emit('disconnected');

        // Attempt to reconnect
        if (this.reconnectAttempts < WS_MAX_RECONNECT_ATTEMPTS) {
          this.reconnectAttempts++;
          console.log(`[WebSocket] Reconnecting... attempt ${this.reconnectAttempts}`);
          setTimeout(() => this.connect(), WS_RECONNECT_DELAY);
        } else {
          console.error('[WebSocket] Max reconnection attempts reached');
          this.emit('error', { message: 'Connection lost' });
        }
      };

      this.ws.onerror = (error) => {
        console.error('[WebSocket] Error:', error);
        this.emit('error', error);
      };
    } catch (e) {
      console.error('[WebSocket] Failed to connect:', e);
      this.emit('error', e);
    }
  }

  disconnect() {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
      this.isConnected = false;
    }
  }

  send(data) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(data));
    } else {
      // Queue message for when connected
      this.messageQueue.push(data);
    }
  }

  on(event, callback) {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, []);
    }
    this.listeners.get(event).push(callback);
  }

  off(event, callback) {
    if (this.listeners.has(event)) {
      const callbacks = this.listeners.get(event);
      const index = callbacks.indexOf(callback);
      if (index > -1) {
        callbacks.splice(index, 1);
      }
    }
  }

  emit(event, data) {
    if (this.listeners.has(event)) {
      this.listeners.get(event).forEach(callback => callback(data));
    }
  }
}

// Singleton instance for alerts
let alertsWebSocket = null;

export function getAlertsWebSocket() {
  if (!alertsWebSocket) {
    alertsWebSocket = new WebSocketHandler('/api/v1/alerts/ws');
  }
  return alertsWebSocket;
}

export default WebSocketHandler;
