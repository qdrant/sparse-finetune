/* qdrant-sparse-finetune Dashboard */

const API = '';
let activeJobId = null;
let pollInterval = null;
let lossChart = null;
let jobStartTime = null;
let elapsedInterval = null;

// ---------------------------------------------------------------------------
// Tab navigation
// ---------------------------------------------------------------------------

function switchTab(tabName) {
    document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('.nav-item').forEach(el => el.classList.remove('active'));

    const tab = document.getElementById('tab-' + tabName);
    const nav = document.querySelector('.nav-item[data-tab="' + tabName + '"]');
    if (tab) tab.classList.add('active');
    if (nav) nav.classList.add('active');

    if (tabName === 'collections') loadCollections();
    if (tabName === 'jobs') { loadJobs(); startJobPolling(); }
    else { stopJobPolling(); }
    if (tabName === 'settings') loadSettings();
}

// ---------------------------------------------------------------------------
// Status
// ---------------------------------------------------------------------------

async function checkStatus() {
    try {
        const res = await fetch(API + '/api/status');
        const data = await res.json();

        const qdrantDot = document.getElementById('qdrant-dot');
        const qdrantLabel = document.getElementById('qdrant-label');
        if (data.qdrant_connected) {
            qdrantDot.className = 'status-dot green';
            qdrantLabel.textContent = 'Qdrant: connected';
        } else {
            qdrantDot.className = 'status-dot red';
            qdrantLabel.textContent = 'Qdrant: disconnected';
        }

        const gpuDot = document.getElementById('gpu-dot');
        const gpuLabel = document.getElementById('gpu-label');
        if (data.gpu_available) {
            gpuDot.className = 'status-dot green';
            gpuLabel.textContent = 'GPU: ' + data.gpu_name;
        } else {
            gpuDot.className = 'status-dot yellow';
            gpuLabel.textContent = 'GPU: none (cloud only)';
        }

        if (data.version && data.version !== 'unknown') {
            document.getElementById('version-tag').textContent = 'v' + data.version;
        } else {
            document.getElementById('version-tag').textContent = 'dev';
        }
    } catch (e) {
        document.getElementById('qdrant-dot').className = 'status-dot red';
        document.getElementById('qdrant-label').textContent = 'Qdrant: error';
    }
}

// ---------------------------------------------------------------------------
// Training
// ---------------------------------------------------------------------------

async function startTraining() {
    const btn = document.getElementById('btn-train');
    btn.disabled = true;
    btn.textContent = 'Starting...';

    const body = {
        data_path: document.getElementById('train-data').value,
        queries_path: document.getElementById('train-queries').value || null,
        gpu_backend: document.getElementById('train-gpu').value,
        base_model: document.getElementById('train-model').value,
        ance_iterations: parseInt(document.getElementById('train-ance').value),
        batch_size: parseInt(document.getElementById('train-batch').value),
        learning_rate: parseFloat(document.getElementById('train-lr').value),
        num_epochs: parseInt(document.getElementById('train-epochs').value),
        run_name: document.getElementById('train-runname').value,
    };

    if (!body.data_path) {
        btn.disabled = false;
        btn.textContent = 'Start Training';
        return;
    }

    try {
        const res = await fetch(API + '/api/train', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        const data = await res.json();
        activeJobId = data.job_id;
        showActiveJob(data.job_id);
        startPolling();
    } catch (e) {
        btn.disabled = false;
        btn.textContent = 'Start Training';
    }
}

function showActiveJob(jobId) {
    const card = document.getElementById('active-job');
    card.style.display = 'block';
    document.getElementById('active-job-id').textContent = ' #' + jobId;
    document.getElementById('active-logs').innerHTML = '<div class="log-line">Waiting for logs...</div>';

    jobStartTime = Date.now();
    if (elapsedInterval) clearInterval(elapsedInterval);
    elapsedInterval = setInterval(updateElapsed, 1000);

    initLossChart();
}

function updateElapsed() {
    if (!jobStartTime) return;
    const elapsed = Math.floor((Date.now() - jobStartTime) / 1000);
    const mins = Math.floor(elapsed / 60);
    const secs = elapsed % 60;
    document.getElementById('active-elapsed').textContent =
        String(mins).padStart(2, '0') + ':' + String(secs).padStart(2, '0');
}

function initLossChart() {
    const ctx = document.getElementById('loss-chart');
    if (lossChart) lossChart.destroy();

    lossChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Training Loss',
                data: [],
                borderColor: '#00B4D8',
                backgroundColor: 'rgba(0, 180, 216, 0.1)',
                borderWidth: 2,
                fill: true,
                tension: 0.3,
                pointRadius: 0,
                pointHitRadius: 8,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: { color: '#8b949e', font: { size: 12 } }
                }
            },
            scales: {
                x: {
                    title: { display: true, text: 'Step', color: '#656d76' },
                    ticks: { color: '#656d76' },
                    grid: { color: 'rgba(48, 54, 61, 0.5)' },
                },
                y: {
                    title: { display: true, text: 'Loss', color: '#656d76' },
                    ticks: { color: '#656d76' },
                    grid: { color: 'rgba(48, 54, 61, 0.5)' },
                }
            }
        }
    });
}

function startPolling() {
    if (pollInterval) clearInterval(pollInterval);
    pollInterval = setInterval(pollJobStatus, 2000);
}

function stopPolling() {
    if (pollInterval) {
        clearInterval(pollInterval);
        pollInterval = null;
    }
    if (elapsedInterval) {
        clearInterval(elapsedInterval);
        elapsedInterval = null;
    }
}

async function pollJobStatus() {
    if (!activeJobId) return;

    try {
        const res = await fetch(API + '/api/jobs/' + activeJobId);
        const job = await res.json();

        // Update status badge
        const statusEl = document.getElementById('active-status');
        statusEl.textContent = job.status.toUpperCase();
        statusEl.className = 'job-status ' + job.status;

        // Update progress bar
        const progressEl = document.getElementById('active-progress');
        if (job.status === 'completed') {
            progressEl.className = 'progress-fill';
            progressEl.style.width = '100%';
        } else if (job.status === 'failed') {
            progressEl.className = 'progress-fill';
            progressEl.style.width = '100%';
            progressEl.style.background = 'var(--red)';
        } else {
            progressEl.className = 'progress-fill indeterminate';
        }

        // Update logs
        if (job.logs && job.logs.length > 0) {
            const logsEl = document.getElementById('active-logs');
            logsEl.innerHTML = job.logs.map(l => '<div class="log-line">' + escapeHtml(l) + '</div>').join('');
            logsEl.scrollTop = logsEl.scrollHeight;
        }

        // Update chart with mock data for demo (real metrics would come from job.metrics)
        if (job.metrics && job.metrics.length > 0 && lossChart) {
            lossChart.data.labels = job.metrics.map((_, i) => i + 1);
            lossChart.data.datasets[0].data = job.metrics;
            lossChart.update('none');
        }

        // Stop polling if done
        if (job.status === 'completed' || job.status === 'failed') {
            stopPolling();
            document.getElementById('btn-train').disabled = false;
            document.getElementById('btn-train').textContent = 'Start Training';
        }
    } catch (e) {
        // ignore transient errors
    }
}

// ---------------------------------------------------------------------------
// Evaluate
// ---------------------------------------------------------------------------

async function startEvaluation() {
    const btn = document.getElementById('btn-eval');
    btn.disabled = true;
    btn.textContent = 'Evaluating...';

    const body = {
        model_path: document.getElementById('eval-model').value,
        queries_path: document.getElementById('eval-queries').value,
        collection_name: document.getElementById('eval-collection').value || null,
    };

    if (!body.model_path || !body.queries_path) {
        btn.disabled = false;
        btn.textContent = 'Run Evaluation';
        return;
    }

    try {
        const res = await fetch(API + '/api/evaluate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        const data = await res.json();
        pollEvaluation(data.job_id);
    } catch (e) {
        showFeedback('eval-feedback', 'Error starting evaluation: ' + e.message, true);
        btn.disabled = false;
        btn.textContent = 'Run Evaluation';
    }
}

async function pollEvaluation(jobId) {
    const check = async () => {
        try {
            const res = await fetch(API + '/api/jobs/' + jobId);
            const job = await res.json();

            if (job.status === 'completed' && job.results) {
                displayMetrics(job.results);
                document.getElementById('btn-eval').disabled = false;
                document.getElementById('btn-eval').textContent = 'Run Evaluation';
                return;
            } else if (job.status === 'failed') {
                showFeedback('eval-feedback', 'Evaluation failed: ' + (job.error || 'Unknown error'), true);
                document.getElementById('btn-eval').disabled = false;
                document.getElementById('btn-eval').textContent = 'Run Evaluation';
                return;
            }

            setTimeout(check, 2000);
        } catch (e) {
            setTimeout(check, 3000);
        }
    };
    check();
}

function displayMetrics(metrics) {
    const container = document.getElementById('metrics-grid');
    container.innerHTML = '';

    const displayOrder = ['mrr@10', 'ndcg@10', 'recall@10', 'precision@10', 'ndcg@50', 'recall@50', 'ndcg@100', 'recall@100'];
    const keys = displayOrder.filter(k => k in metrics);
    // Add any remaining keys
    Object.keys(metrics).forEach(k => {
        if (!keys.includes(k)) keys.push(k);
    });

    keys.forEach(key => {
        const card = document.createElement('div');
        card.className = 'metric-card';
        card.innerHTML =
            '<div class="metric-value">' + metrics[key].toFixed(4) + '</div>' +
            '<div class="metric-label">' + key + '</div>';
        container.appendChild(card);
    });

    document.getElementById('eval-results').style.display = 'block';
    document.getElementById('eval-feedback').className = 'feedback';
    document.getElementById('eval-feedback').style.display = 'none';
}

// ---------------------------------------------------------------------------
// Collections
// ---------------------------------------------------------------------------

async function loadCollections() {
    try {
        const res = await fetch(API + '/api/collections');
        const data = await res.json();

        const tbody = document.getElementById('collections-body');
        const select = document.getElementById('search-collection');

        if (!data.collections || data.collections.length === 0) {
            tbody.innerHTML = '<tr><td colspan="4" style="text-align: center; color: var(--text-muted); padding: 24px;">No collections found</td></tr>';
            select.innerHTML = '<option value="">No collections</option>';
            return;
        }

        tbody.innerHTML = data.collections.map(c =>
            '<tr>' +
            '<td style="font-weight: 600;">' + escapeHtml(c.name) + '</td>' +
            '<td>' + (c.points_count != null ? c.points_count.toLocaleString() : '--') + '</td>' +
            '<td>' + (c.vectors_count != null ? c.vectors_count.toLocaleString() : '--') + '</td>' +
            '<td><span class="collection-status"><span class="status-dot green"></span> ' + escapeHtml(c.status) + '</span></td>' +
            '</tr>'
        ).join('');

        select.innerHTML = data.collections.map(c =>
            '<option value="' + escapeHtml(c.name) + '">' + escapeHtml(c.name) + '</option>'
        ).join('');

    } catch (e) {
        document.getElementById('collections-body').innerHTML =
            '<tr><td colspan="4" style="text-align: center; color: var(--red); padding: 24px;">Failed to load collections</td></tr>';
    }
}

async function runSearch() {
    const collection = document.getElementById('search-collection').value;
    const query = document.getElementById('search-query').value;
    const topK = parseInt(document.getElementById('search-topk').value) || 10;

    if (!collection || !query) return;

    const container = document.getElementById('search-results');
    container.innerHTML = '<div style="padding: 12px; color: var(--text-muted);">Searching...</div>';

    try {
        const res = await fetch(API + '/api/search', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ collection_name: collection, query: query, top_k: topK }),
        });
        const data = await res.json();

        if (!data.results || data.results.length === 0) {
            container.innerHTML = '<div style="padding: 12px; color: var(--text-muted);">No results found</div>';
            return;
        }

        container.innerHTML = data.results.map((r, i) =>
            '<div class="search-result-item">' +
            '<div class="result-id">#' + (i + 1) + ' | ID: ' + escapeHtml(String(r.id)) + (r.score != null ? ' | Score: ' + r.score : '') + '</div>' +
            '<div class="result-payload">' + escapeHtml(JSON.stringify(r.payload, null, 2).substring(0, 500)) + '</div>' +
            '</div>'
        ).join('');
    } catch (e) {
        container.innerHTML = '<div style="padding: 12px; color: var(--red);">Search failed: ' + escapeHtml(e.message) + '</div>';
    }
}

// ---------------------------------------------------------------------------
// Publish
// ---------------------------------------------------------------------------

async function publishModel() {
    const btn = document.getElementById('btn-publish');
    btn.disabled = true;
    btn.textContent = 'Publishing...';

    const body = {
        model_path: document.getElementById('pub-model').value,
        repo_name: document.getElementById('pub-repo').value,
        hf_token: document.getElementById('pub-token').value,
        private: document.getElementById('pub-private').checked,
    };

    if (!body.model_path || !body.repo_name || !body.hf_token) {
        showFeedback('pub-feedback', 'All fields are required.', true);
        btn.disabled = false;
        btn.textContent = 'Publish';
        return;
    }

    try {
        const res = await fetch(API + '/api/publish', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        const data = await res.json();
        pollPublish(data.job_id);
    } catch (e) {
        showFeedback('pub-feedback', 'Error: ' + e.message, true);
        btn.disabled = false;
        btn.textContent = 'Publish';
    }
}

async function pollPublish(jobId) {
    const check = async () => {
        try {
            const res = await fetch(API + '/api/jobs/' + jobId);
            const job = await res.json();

            if (job.status === 'completed') {
                const lastLog = job.logs[job.logs.length - 1] || 'Published successfully';
                showFeedback('pub-feedback', lastLog, false);
                document.getElementById('btn-publish').disabled = false;
                document.getElementById('btn-publish').textContent = 'Publish';
                return;
            } else if (job.status === 'failed') {
                showFeedback('pub-feedback', 'Failed: ' + (job.error || 'Unknown error'), true);
                document.getElementById('btn-publish').disabled = false;
                document.getElementById('btn-publish').textContent = 'Publish';
                return;
            }

            setTimeout(check, 2000);
        } catch (e) {
            setTimeout(check, 3000);
        }
    };
    check();
}

// ---------------------------------------------------------------------------
// Jobs
// ---------------------------------------------------------------------------

async function loadJobs() {
    try {
        const res = await fetch(API + '/api/jobs');
        const data = await res.json();

        const container = document.getElementById('jobs-list');

        if (!data.jobs || data.jobs.length === 0) {
            container.innerHTML = '<div class="empty-state"><h3>No jobs yet</h3><p>Start a training job to see it here</p></div>';
            return;
        }

        const sorted = data.jobs.sort((a, b) => (b.created_at || '').localeCompare(a.created_at || ''));
        container.innerHTML = sorted.map(job =>
            '<div class="job-list-item" onclick="showJobDetail(\'' + job.id + '\')">' +
            '<div class="job-info">' +
            '<span class="job-type">' + escapeHtml(job.type) + '</span>' +
            (job.source ? '<span class="job-source ' + job.source + '">' + job.source + '</span>' : '') +
            '<span class="job-id">#' + escapeHtml(job.id) + '</span>' +
            '<span class="job-time">' + formatJobTime(job) + '</span>' +
            '</div>' +
            '<span class="job-status ' + job.status + '">' + job.status.toUpperCase() + '</span>' +
            '</div>'
        ).join('');
    } catch (e) {
        // ignore
    }
}

async function showJobDetail(jobId) {
    try {
        const res = await fetch(API + '/api/jobs/' + jobId);
        const job = await res.json();

        const detail = document.getElementById('job-detail');
        detail.style.display = 'block';

        const statusEl = document.getElementById('detail-status');
        statusEl.textContent = job.status.toUpperCase();
        statusEl.className = 'job-status ' + job.status;

        // Source + type header with elapsed time
        var headerInfo = '<span style="font-size:13px;color:var(--text-secondary);">' +
            escapeHtml(job.type) + ' job' +
            (job.source ? ' <span class="job-source ' + job.source + '">' + job.source + '</span>' : '') +
            ' &mdash; ' + formatJobTime(job) +
            '</span>';
        document.getElementById('detail-header-info').innerHTML = headerInfo;

        document.getElementById('detail-config').innerHTML =
            '<pre style="margin: 0; font-family: var(--font-mono); font-size: 12px; color: var(--text-secondary);">' +
            escapeHtml(JSON.stringify(job.config, null, 2)) + '</pre>';

        // Results section
        const resultsEl = document.getElementById('detail-results');
        if (job.results && typeof job.results === 'object') {
            let html = '';
            // Summary info
            if (job.results.count != null) {
                html += '<div style="margin-bottom:12px;color:var(--text-secondary);font-size:13px;">' +
                    'Generated <strong style="color:var(--text-primary);">' + job.results.count + '</strong> items';
                if (job.results.output_file) html += ' &rarr; <code>' + escapeHtml(job.results.output_file) + '</code>';
                html += '</div>';
            }
            // Queries table
            if (job.results.queries && job.results.queries.length > 0) {
                html += '<table class="data-table"><thead><tr><th>#</th><th>Query</th><th>Matched Product</th></tr></thead><tbody>';
                job.results.queries.forEach(function(q, i) {
                    html += '<tr><td>' + (i+1) + '</td>' +
                        '<td style="font-weight:600;">' + escapeHtml(q.query) + '</td>' +
                        '<td style="font-size:12px;color:var(--text-secondary);max-width:400px;overflow:hidden;text-overflow:ellipsis;">' + escapeHtml(q.positive_text || '') + '</td></tr>';
                });
                html += '</tbody></table>';
            }
            // Generic metrics (for evaluate jobs)
            if (!job.results.queries) {
                html += '<pre style="margin:0;font-family:var(--font-mono);font-size:12px;color:var(--text-secondary);">' +
                    escapeHtml(JSON.stringify(job.results, null, 2)) + '</pre>';
            }
            resultsEl.innerHTML = html;
            resultsEl.style.display = 'block';
        } else {
            resultsEl.style.display = 'none';
        }

        const logsEl = document.getElementById('detail-logs');
        if (job.logs && job.logs.length > 0) {
            logsEl.innerHTML = job.logs.map(l => '<div class="log-line">' + escapeHtml(l) + '</div>').join('');
        } else {
            logsEl.innerHTML = '<div class="log-line">No logs available</div>';
        }
    } catch (e) {
        // ignore
    }
}

let jobPollInterval = null;

function startJobPolling() {
    stopJobPolling();
    jobPollInterval = setInterval(loadJobs, 3000);
}

function stopJobPolling() {
    if (jobPollInterval) { clearInterval(jobPollInterval); jobPollInterval = null; }
}

// ---------------------------------------------------------------------------
// Settings
// ---------------------------------------------------------------------------

const SETTINGS_KEYS = ['QDRANT_URL', 'QDRANT_API_KEY', 'OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'OPENROUTER_API_KEY', 'VULTR_API_KEY'];

async function loadSettings() {
    try {
        const res = await fetch(API + '/api/settings');
        const data = await res.json();
        SETTINGS_KEYS.forEach(key => {
            const input = document.getElementById('setting-' + key);
            if (input) {
                input.value = data.settings[key] || '';
                input.dataset.masked = data.settings[key] || '';
            }
        });
    } catch (e) {
        // ignore
    }
}

async function saveSettings() {
    const body = {};
    SETTINGS_KEYS.forEach(key => {
        const input = document.getElementById('setting-' + key);
        if (!input) return;
        const val = input.value;
        if (val === '' || val === input.dataset.masked) {
            body[key] = '__UNCHANGED__';
        } else {
            body[key] = val;
        }
    });

    try {
        const res = await fetch(API + '/api/settings', {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        const data = await res.json();
        if (data.status === 'ok') {
            showFeedback('settings-feedback', 'Settings saved.', false);
            loadSettings();
            checkStatus();
        } else {
            showFeedback('settings-feedback', 'Save failed.', true);
        }
    } catch (e) {
        showFeedback('settings-feedback', 'Error: ' + e.message, true);
    }
}

function toggleReveal(btn) {
    const input = btn.parentElement.querySelector('input');
    if (input.type === 'password') {
        input.type = 'text';
        btn.textContent = 'Hide';
    } else {
        input.type = 'password';
        btn.textContent = 'Show';
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function formatJobTime(job) {
    var parts = [];
    if (job.created_at) {
        var d = new Date(job.created_at + 'Z');
        parts.push(d.toLocaleString());
    }
    if (job.started_at && job.completed_at) {
        var start = new Date(job.started_at + 'Z').getTime();
        var end = new Date(job.completed_at + 'Z').getTime();
        var secs = Math.round((end - start) / 1000);
        if (secs < 60) parts.push(secs + 's');
        else parts.push(Math.floor(secs / 60) + 'm ' + (secs % 60) + 's');
    } else if (job.started_at && job.status === 'running') {
        var start = new Date(job.started_at + 'Z').getTime();
        var secs = Math.round((Date.now() - start) / 1000);
        if (secs < 60) parts.push(secs + 's (running)');
        else parts.push(Math.floor(secs / 60) + 'm ' + (secs % 60) + 's (running)');
    }
    return parts.join(' — ');
}

function escapeHtml(str) {
    if (!str) return '';
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

function showFeedback(elementId, message, isError) {
    const el = document.getElementById(elementId);
    el.textContent = message;
    el.className = 'feedback ' + (isError ? 'error' : 'success');
    el.style.display = 'block';
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

document.addEventListener('DOMContentLoaded', function() {
    checkStatus();
    setInterval(checkStatus, 15000);
});
