import api from './api/client.js';
import Alpine from 'alpinejs';

// Import Stores
import './stores/index.js';

// Import View Controllers
import dashboardView from './views/Dashboard.js';
import uploadView from './views/Upload.js';
import liveStreamView from './views/LiveStream.js';
import resultsView from './views/Results.js';
import watchlistView from './views/Watchlist.js';
import alertsView from './views/Alerts.js';
import settingsView from './views/Settings.js';
import singleImageView from './views/SingleImage.js';
import analyzeView from './views/Analyze.js';



window.Alpine = Alpine;

document.addEventListener('alpine:init', () => {
    // Register Views
    Alpine.data('dashboardView', dashboardView);
    Alpine.data('uploadView', uploadView);
    Alpine.data('liveStreamView', liveStreamView);
    Alpine.data('resultsView', resultsView);
    Alpine.data('watchlistView', watchlistView);
    Alpine.data('alertsView', alertsView);
    Alpine.data('settingsView', settingsView);
    Alpine.data('singleImageView', singleImageView);
    Alpine.data('analyzeView', analyzeView);


    
    // Initialize App
    const appStore = Alpine.store('app');
    
    // Pollers
    const pollHealth = async () => {
        try {
            await api.getHealth();
            appStore.isOnline = true;
        } catch (e) {
            appStore.isOnline = false;
        }
        setTimeout(pollHealth, 30000);
    };
    
    // Handle routing
    const handleHash = () => {
        const hash = window.location.hash.slice(2) || 'dashboard';
        appStore.currentView = hash;
    };
    window.addEventListener('hashchange', handleHash);
    
    // Initialize
    const init = () => {
        // Dark mode
        if (appStore.darkMode) {
            document.documentElement.classList.add('dark');
        }
        
        handleHash();
        pollHealth();
        appStore.pollStats();
        appStore.pollJobs();
    };
    
    init();
});

Alpine.start();

// Initialize alerts WebSocket after Alpine is fully started
// This ensures the alerts store is registered and ready
setTimeout(() => {
    const alertsStore = Alpine.store('alerts');
    if (alertsStore && alertsStore.init) {
        console.log('[Main] Initializing alerts WebSocket...');
        alertsStore.init();
    } else {
        console.error('[Main] Alerts store not found!');
    }
}, 100);
