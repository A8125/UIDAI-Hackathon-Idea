document.addEventListener('DOMContentLoaded', () => {
    // Tab Switching Logic
    const navItems = document.querySelectorAll('.side-nav li');
    const tabContents = document.querySelectorAll('.tab-content');

    navItems.forEach(item => {
        item.addEventListener('click', () => {
            const targetId = item.getAttribute('data-tab');

            // Update Active Nav
            navItems.forEach(nav => nav.classList.remove('active'));
            item.classList.add('active');

            // Update Tabs
            tabContents.forEach(tab => {
                if (tab.id === targetId) {
                    tab.classList.add('active');
                } else {
                    tab.classList.remove('active');
                }
            });
        });
    });

    // Mock Data Ingestion
    loadMockData();
});

async function loadMockData() {
    // In a real app, this would fetch from a server serving the JSON/CSV files
    // Since we're running locally, we'll simulate the load with sample data 
    // that mirrors what's in the output/ directory.

    initAnomalies();
    initRecommendations();
    initClusters();
}

function initAnomalies() {
    const tableBody = document.querySelector('#anomaly-table tbody');
    const anomalies = [
        { severity: 'critical', district: 'Mumbai City', metric: 'Spike in Rejections', status: 'Reviewing' },
        { severity: 'high', district: 'Bengaluru Urban', metric: 'Saturation Drop', status: 'Assigned' },
        { severity: 'medium', district: 'Jaipur', metric: 'Enrollment Lag', status: 'Pending' },
        { severity: 'high', district: 'Chennai', metric: 'Biometric Failure %', status: 'Under Investigation' }
    ];

    tableBody.innerHTML = anomalies.map(a => `
        <tr>
            <td><span class="severity-pill severity-${a.severity}">${a.severity.toUpperCase()}</span></td>
            <td>${a.district}</td>
            <td>${a.metric}</td>
            <td>${a.status}</td>
        </tr>
    `).join('');
}

function initRecommendations() {
    const container = document.getElementById('recommendations-container');
    const recs = [
        { icon: '🚐', title: 'Deploy Mobile Van - District A', description: 'Saturation below 85% in rural blocks. Predicted recovery in 3 months with mobile support.' },
        { icon: '🏛️', title: 'Expand Permanent Center - District B', description: 'Consistently high waiting times. Recommended 2 additional counters.' },
        { icon: '🎓', title: 'Conduct Operator Training - District C', description: 'High rejection rates due to technical errors. Scheduled refresh course recommended.' }
    ];

    container.innerHTML = recs.map(r => `
        <div class="rec-card">
            <div class="rec-icon">${r.icon}</div>
            <div class="rec-content">
                <h4>${r.title}</h4>
                <p>${r.description}</p>
            </div>
        </div>
    `).join('');
}

function initClusters() {
    const container = document.getElementById('cluster-container');
    // Simplified cluster representation
    const clusters = [
        { id: 0, name: 'Saturated Megacities', count: 120, avg: '98%' },
        { id: 1, name: 'Growth Corridors', count: 215, avg: '92%' },
        { id: 2, name: 'Service Gaps', count: 85, avg: '81%' },
        { id: 3, name: 'Update Hubs', count: 150, avg: '95%' }
    ];

    // Placeholder for actual cluster breakdown
}
