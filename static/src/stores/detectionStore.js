import api from '../api/client.js';

document.addEventListener('alpine:init', () => {
  // Detections Store
  Alpine.store('detections', {
    // Raw detections (for legacy grid/list views)
    items: [],
    groupedByJob: [],  // Detections grouped by job_id
    
    // Event-based view (new, performance-optimized)
    events: [],
    eventsTotal: 0,
    timeWindow: 30,  // seconds to group detections
    
    // Pagination
    total: 0,
    page: 1,
    pageSize: 50,
    
    // Loading states
    isLoading: false,
    isLoadingMore: false,
    hasMore: true,
    
    // Expansion tracking
    expandedJobs: {},  // Track which job groups are expanded
    expandedEvents: {},  // Track which events are expanded to show all frames
    
    // Filters
    filters: {
      plateText: '',
      minConfidence: 0,
      startTime: null,
      endTime: null,
      sourceIdentifier: null,
      sourceType: ''
    },
    
    // Fetch events (grouped detections) - PRIMARY METHOD with pagination
    async fetchEvents() {
      this.isLoading = true;
      
      try {
        const result = await api.getDetectionEvents({
          time_window: this.timeWindow,
          plate_text: this.filters.plateText || undefined,
          min_confidence: this.filters.minConfidence || undefined,
          start_time: this.filters.startTime || undefined,
          end_time: this.filters.endTime || undefined,
          source_identifier: this.filters.sourceIdentifier || undefined,
          limit: this.pageSize,
          offset: (this.page - 1) * this.pageSize
        });
        
        this.events = result.events || [];
        this.eventsTotal = result.total || 0;
        
      } catch (e) {
        console.error('Failed to fetch detection events', e);
        throw e;
      } finally {
        this.isLoading = false;
      }
    },
    
    // Go to next page of events
    nextEventsPage() {
      if (this.page * this.pageSize < this.eventsTotal) {
        this.page++;
        this.fetchEvents();
      }
    },
    
    // Go to previous page of events
    prevEventsPage() {
      if (this.page > 1) {
        this.page--;
        this.fetchEvents();
      }
    },
    
    // Fetch raw detections (for legacy views) - sorted by most recent from backend
    async fetch() {
      this.isLoading = true;
      try {
        const result = await api.getDetections({
          plate_text: this.filters.plateText,
          min_confidence: this.filters.minConfidence,
          start_time: this.filters.startTime,
          end_time: this.filters.endTime,
          source_identifier: this.filters.sourceIdentifier,
          limit: this.pageSize,
          offset: (this.page - 1) * this.pageSize
        });
        // Items are already sorted by timestamp DESC from the backend
        this.items = result.detections || [];
        this.total = result.total || 0;
        
        // Group detections by job_id
        this.groupByJob();
      } catch (e) {
        console.error('Failed to fetch detections', e);
        throw e;
      } finally {
        this.isLoading = false;
      }
    },
    
    // Get individual detections for an expanded event
    async getEventDetections(event) {
      try {
        const result = await api.getEventDetections(
          event.plate_text,
          event.first_seen,
          event.last_seen,
          100
        );
        return result.detections || [];
      } catch (e) {
        console.error('Failed to fetch event detections', e);
        return [];
      }
    },
    
    // Toggle event expansion (to show all frames)
    async toggleEventExpanded(eventKey) {
      if (this.expandedEvents[eventKey]) {
        delete this.expandedEvents[eventKey];
      } else {
        // Find the event and load its detections
        const event = this.events.find(e => 
          `${e.plate_text}-${e.first_seen}` === eventKey
        );
        if (event) {
          const detections = await this.getEventDetections(event);
          this.expandedEvents[eventKey] = detections;
        }
      }
    },
    
    // Check if event is expanded
    isEventExpanded(eventKey) {
      return !!this.expandedEvents[eventKey];
    },
    
    // Get expanded event detections
    getExpandedDetections(eventKey) {
      return this.expandedEvents[eventKey] || [];
    },
    
    // Group detections by job_id and sort by most recent (legacy)
    groupByJob() {
      const groups = {};
      
      this.items.forEach(detection => {
        const jobId = detection.job_id || 'ungrouped';
        if (!groups[jobId]) {
          groups[jobId] = {
            job_id: jobId,
            detections: [],
            source_type: detection.source_type,
            source_identifier: detection.source_identifier,
            earliest_timestamp: detection.timestamp,
            latest_timestamp: detection.timestamp,
            unique_plates: new Set(),
            total_count: 0
          };
        }
        
        groups[jobId].detections.push(detection);
        groups[jobId].unique_plates.add(detection.plate_text);
        groups[jobId].total_count++;
        
        // Track earliest and latest timestamps
        if (new Date(detection.timestamp) < new Date(groups[jobId].earliest_timestamp)) {
          groups[jobId].earliest_timestamp = detection.timestamp;
        }
        if (new Date(detection.timestamp) > new Date(groups[jobId].latest_timestamp)) {
          groups[jobId].latest_timestamp = detection.timestamp;
        }
      });
      
      // Convert to array and sort by latest timestamp (most recent first)
      this.groupedByJob = Object.values(groups)
        .map(group => ({
          ...group,
          unique_plates: group.unique_plates.size,
          // Sort detections within group by frame_number (for videos) or timestamp
          detections: group.detections.sort((a, b) => {
            if (a.frame_number && b.frame_number) {
              return a.frame_number - b.frame_number;
            }
            return new Date(a.timestamp) - new Date(b.timestamp);
          })
        }))
        .sort((a, b) => new Date(b.latest_timestamp) - new Date(a.latest_timestamp));
      
      // Auto-expand the first (most recent) job if not already set
      if (this.groupedByJob.length > 0 && Object.keys(this.expandedJobs).length === 0) {
        this.expandedJobs[this.groupedByJob[0].job_id] = true;
      }
    },
    
    // Toggle job group expansion
    toggleJobExpanded(jobId) {
      this.expandedJobs[jobId] = !this.expandedJobs[jobId];
    },
    
    // Check if job is expanded
    isJobExpanded(jobId) {
      return this.expandedJobs[jobId] || false;
    },
    
    // Expand all job groups
    expandAll() {
      this.groupedByJob.forEach(group => {
        this.expandedJobs[group.job_id] = true;
      });
    },
    
    // Collapse all job groups
    collapseAll() {
      this.expandedJobs = {};
    },
    
    setFilter(key, value) {
      this.filters[key] = value;
      this.page = 1;
      // Refresh both views
      this.fetchEvents();
    },
    
    async deleteDetection(id) {
      await api.deleteDetection(id);
      this.items = this.items.filter(d => d.id !== id);
      this.total--;
      // Re-group after deletion
      this.groupByJob();
      // Also refresh events
      this.fetchEvents();
    }
  });
});
