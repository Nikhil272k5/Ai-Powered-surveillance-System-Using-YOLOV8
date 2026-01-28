/**
 * AbnoGuard Command Center - Cyberpunk Dashboard
 * Real-time surveillance intelligence interface
 */

class CommandCenter {
    constructor() {
        this.ws = null;
        this.wsUrl = `ws://${window.location.host}/ws/stream`;
        this.apiUrl = `http://${window.location.host}/api`;
        this.alerts = [];
        this.startTime = Date.now();
        this.isConnected = false;

        this.init();
    }

    init() {
        this.updateTime();
        setInterval(() => this.updateTime(), 1000);
        setInterval(() => this.updateUptime(), 1000);

        this.bindEvents();
        this.connectWebSocket();
        this.loadInitialData();
        this.initProgressRing();

        // Animate on load
        this.animateDesks();
    }

    animateDesks() {
        const desks = document.querySelectorAll('.cyber-desk');
        desks.forEach((desk, i) => {
            desk.style.opacity = '0';
            desk.style.transform = 'translateY(20px)';
            setTimeout(() => {
                desk.style.transition = 'all 0.5s ease';
                desk.style.opacity = '1';
                desk.style.transform = 'translateY(0)';
            }, i * 100);
        });
    }

    bindEvents() {
        // Filter buttons
        document.querySelectorAll('.filter-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
                e.target.classList.add('active');
                this.filterAlerts(e.target.dataset.filter);
            });
        });

        // Control buttons
        document.getElementById('btn-export')?.addEventListener('click', () => this.exportData());
        document.getElementById('btn-reset')?.addEventListener('click', () => this.resetDashboard());
        document.getElementById('modal-close')?.addEventListener('click', () => this.closeModal());
        document.getElementById('btn-acknowledge')?.addEventListener('click', () => this.sendFeedback('acknowledged'));
        document.getElementById('btn-dismiss')?.addEventListener('click', () => this.sendFeedback('dismissed'));
    }

    connectWebSocket() {
        try {
            this.ws = new WebSocket(this.wsUrl);

            this.ws.onopen = () => {
                this.isConnected = true;
                this.updateStatus('SYSTEM ONLINE', 'online');
            };

            this.ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    this.handleMessage(data);
                } catch (e) {
                    console.error('Parse error:', e);
                }
            };

            this.ws.onclose = () => {
                this.isConnected = false;
                this.updateStatus('RECONNECTING...', 'offline');
                setTimeout(() => this.connectWebSocket(), 3000);
            };

            this.ws.onerror = () => {
                this.updateStatus('CONNECTION ERROR', 'error');
            };

        } catch (e) {
            console.error('WebSocket error:', e);
        }
    }

    handleMessage(message) {
        if (message.type === 'state_update') {
            const data = message.data;

            // Update frame info
            document.getElementById('fps-value').textContent = (data.fps || 0).toFixed(1);
            document.getElementById('frame-value').textContent = data.frame_count || 0;

            // Update normality
            if (data.normality) {
                this.updateNormality(data.normality);
            }

            // Update behaviors
            if (data.behaviors) {
                this.updateBehaviors(data.behaviors);
            }

            // Update alerts
            if (data.recent_alerts) {
                this.addAlerts(data.recent_alerts);
            }
        }
    }

    updateStatus(text, type) {
        const statusEl = document.getElementById('system-status');
        const indicator = document.querySelector('.status-indicator');

        statusEl.textContent = text;

        indicator.style.borderColor = type === 'online' ? 'var(--neon-green)' :
            type === 'error' ? 'var(--neon-red)' : 'var(--neon-orange)';
    }

    updateTime() {
        const now = new Date();
        const timeStr = now.toTimeString().split(' ')[0];
        document.getElementById('time-display').textContent = timeStr;
    }

    updateUptime() {
        const elapsed = Math.floor((Date.now() - this.startTime) / 1000);
        const mins = Math.floor(elapsed / 60).toString().padStart(2, '0');
        const secs = (elapsed % 60).toString().padStart(2, '0');
        document.getElementById('vital-uptime').textContent = `${mins}:${secs}`;
    }

    initProgressRing() {
        this.setNormalityProgress(0);
    }

    setNormalityProgress(percent) {
        const ring = document.getElementById('normality-ring');
        const circumference = 2 * Math.PI * 52;
        const offset = circumference - (percent / 100) * circumference;
        ring.style.strokeDashoffset = offset;
        document.getElementById('normality-percent').textContent = Math.round(percent);
    }

    updateNormality(data) {
        const progress = data.progress_percentage || 0;
        this.setNormalityProgress(progress);

        document.getElementById('norm-observations').textContent = data.observation_count || 0;

        if (data.baseline_stats) {
            document.getElementById('norm-speed').textContent =
                (data.baseline_stats.speed_mean || 0).toFixed(2);
            document.getElementById('norm-density').textContent =
                (data.baseline_stats.density_mean || 0).toFixed(2);
        }

        // Update status badge
        const statusBadge = document.querySelector('.desk-normality .desk-status');
        if (progress >= 100) {
            statusBadge.textContent = 'READY';
            statusBadge.classList.remove('learning');
            statusBadge.classList.add('online');
        }
    }

    updateBehaviors(behaviors) {
        const container = document.getElementById('behavior-container');
        const entries = Object.entries(behaviors);

        document.getElementById('behavior-count').textContent = entries.length;

        if (entries.length === 0) {
            container.innerHTML = `
                <div class="empty-state">
                    <span class="empty-icon">👁️</span>
                    <p>SCANNING FOR BEHAVIORS</p>
                </div>
            `;
            return;
        }

        container.innerHTML = entries.slice(0, 6).map(([trackId, b]) => {
            const intent = b.intent || 'unknown';
            const isAlert = ['panic_movement', 'evasive_behavior', 'suspicious_abandonment'].includes(intent);
            const isWarning = ['loitering'].includes(intent);

            return `
                <div class="behavior-item ${isAlert ? 'alert' : isWarning ? 'warning' : ''}">
                    <span class="behavior-label">T${trackId}: ${intent.replace(/_/g, ' ').toUpperCase()}</span>
                    <span class="behavior-conf">${((b.confidence || 0) * 100).toFixed(0)}%</span>
                </div>
            `;
        }).join('');
    }

    addAlerts(newAlerts) {
        for (const alert of newAlerts) {
            if (!this.alerts.find(a => a.alert_id === alert.alert_id)) {
                this.alerts.unshift(alert);
            }
        }

        this.alerts = this.alerts.slice(0, 100);
        document.getElementById('alert-count').textContent = this.alerts.length;

        this.renderAlerts();
        this.updateTrustMeter();
        this.updateVitals();
    }

    renderAlerts() {
        const container = document.getElementById('alerts-container');
        const filter = document.querySelector('.filter-btn.active')?.dataset.filter || 'all';

        let filtered = this.alerts;
        if (filter !== 'all') {
            const thresholds = { high: 70, medium: 50, low: 0 };
            filtered = this.alerts.filter(a => {
                const score = a.trust_score || 50;
                if (filter === 'high') return score >= 70;
                if (filter === 'medium') return score >= 50 && score < 70;
                if (filter === 'low') return score < 50;
                return true;
            });
        }

        if (filtered.length === 0) {
            container.innerHTML = `
                <div class="empty-state">
                    <span class="empty-icon">⚡</span>
                    <p>NO THREATS DETECTED</p>
                </div>
            `;
            return;
        }

        container.innerHTML = filtered.slice(0, 20).map(alert => {
            const score = alert.trust_score || 50;
            const severity = score >= 70 ? 'high' : score >= 50 ? 'medium' : 'low';
            const time = new Date((alert.timestamp || Date.now() / 1000) * 1000).toLocaleTimeString();

            return `
                <div class="alert-card ${severity}" data-id="${alert.alert_id || ''}">
                    <div class="alert-type">${(alert.type || 'ALERT').toUpperCase()}</div>
                    <div class="alert-desc">${alert.description || 'Anomaly detected'}</div>
                    <div class="alert-meta">
                        <span>${time}</span>
                        <span class="trust-badge ${severity}">${score.toFixed(0)}%</span>
                    </div>
                </div>
            `;
        }).join('');

        // Add click handlers
        container.querySelectorAll('.alert-card').forEach(card => {
            card.addEventListener('click', () => this.showAlertDetail(card.dataset.id));
        });
    }

    filterAlerts(type) {
        this.renderAlerts();
    }

    updateTrustMeter() {
        const scores = this.alerts.filter(a => typeof a.trust_score === 'number').map(a => a.trust_score);
        const avg = scores.length > 0 ? scores.reduce((a, b) => a + b, 0) / scores.length : 0;

        document.getElementById('trust-fill').style.width = `${avg}%`;
        document.getElementById('avg-trust').textContent = avg > 0 ? `${avg.toFixed(0)}%` : '--';
    }

    updateVitals() {
        document.getElementById('vital-tracks').textContent =
            Object.keys(this.behaviors || {}).length;
        document.getElementById('vital-events').textContent = this.alerts.length;
    }

    async loadInitialData() {
        try {
            // Load alerts
            const alertsRes = await fetch(`${this.apiUrl}/alerts`);
            if (alertsRes.ok) {
                const data = await alertsRes.json();
                this.addAlerts(data.alerts || []);
            }

            // Load analytics
            const analyticsRes = await fetch(`${this.apiUrl}/analytics`);
            if (analyticsRes.ok) {
                const data = await analyticsRes.json();
                this.updateFusionStats(data);
            }

            // Load improvement metrics
            const improvementRes = await fetch(`${this.apiUrl}/improvement`);
            if (improvementRes.ok) {
                const data = await improvementRes.json();
                this.updateImprovementStats(data);
            }
        } catch (e) {
            console.error('Failed to load data:', e);
        }
    }

    updateFusionStats(data) {
        document.getElementById('fusion-processed').textContent = data.processed || 0;
        document.getElementById('fusion-emitted').textContent = data.emitted || 0;
        document.getElementById('fusion-suppressed').textContent = data.suppressed || 0;
    }

    updateImprovementStats(data) {
        const fpr = data.current_metrics?.false_positive_rate;
        document.getElementById('fpr-value').textContent =
            typeof fpr === 'number' ? `${(fpr * 100).toFixed(1)}%` : '--';
        document.getElementById('adj-value').textContent = data.total_adjustments || 0;
    }

    showAlertDetail(alertId) {
        const alert = this.alerts.find(a => a.alert_id === alertId);
        if (!alert) return;

        this.selectedAlert = alert;

        const modal = document.getElementById('alert-modal');
        const body = document.getElementById('modal-body');

        body.innerHTML = `
            <div style="padding: 15px;">
                <div style="margin-bottom: 15px;">
                    <span style="color: var(--neon-cyan); font-family: Orbitron; font-size: 11px; letter-spacing: 2px;">THREAT TYPE</span>
                    <div style="font-size: 18px; margin-top: 5px;">${(alert.type || 'UNKNOWN').toUpperCase()}</div>
                </div>
                <div style="margin-bottom: 15px;">
                    <span style="color: var(--neon-cyan); font-family: Orbitron; font-size: 11px; letter-spacing: 2px;">DESCRIPTION</span>
                    <div style="font-size: 14px; color: var(--text-secondary); margin-top: 5px;">${alert.description || 'No details'}</div>
                </div>
                <div style="margin-bottom: 15px;">
                    <span style="color: var(--neon-cyan); font-family: Orbitron; font-size: 11px; letter-spacing: 2px;">TRUST SCORE</span>
                    <div style="font-size: 28px; color: var(--neon-green); font-family: Orbitron; margin-top: 5px;">${(alert.trust_score || 0).toFixed(0)}%</div>
                </div>
                <div style="margin-bottom: 15px;">
                    <span style="color: var(--neon-cyan); font-family: Orbitron; font-size: 11px; letter-spacing: 2px;">CAUSAL ANALYSIS</span>
                    <div style="font-size: 12px; color: var(--text-secondary); margin-top: 5px; line-height: 1.6;">${alert.causal_explanation || 'Direct detection - no causal chain'}</div>
                </div>
            </div>
        `;

        modal.classList.remove('hidden');

        // Update causal panel
        document.getElementById('causal-text').textContent =
            alert.causal_explanation || 'Direct detection - analyzing event chain...';
    }

    closeModal() {
        document.getElementById('alert-modal').classList.add('hidden');
        this.selectedAlert = null;
    }

    async sendFeedback(outcome) {
        if (!this.selectedAlert) return;

        try {
            await fetch(
                `${this.apiUrl}/alerts/${this.selectedAlert.alert_id}/feedback?outcome=${outcome}`,
                { method: 'POST' }
            );
            this.closeModal();
        } catch (e) {
            console.error('Feedback error:', e);
        }
    }

    async exportData() {
        try {
            const res = await fetch(`${this.apiUrl}/logs`);
            if (res.ok) {
                const data = await res.json();
                const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
                const url = URL.createObjectURL(blob);

                const a = document.createElement('a');
                a.href = url;
                a.download = `abnoguard_export_${Date.now()}.json`;
                a.click();

                URL.revokeObjectURL(url);
            }
        } catch (e) {
            console.error('Export error:', e);
        }
    }

    resetDashboard() {
        if (confirm('Reset all dashboard data?')) {
            this.alerts = [];
            this.renderAlerts();
            this.setNormalityProgress(0);
            document.getElementById('behavior-container').innerHTML = `
                <div class="empty-state">
                    <span class="empty-icon">👁️</span>
                    <p>SCANNING FOR BEHAVIORS</p>
                </div>
            `;
        }
    }
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    window.commandCenter = new CommandCenter();
});
