/**
 * CloudShield AI Internet Firewall - Interactive Dashboard Controller
 * Connects directly to Flask Backend API trained on log2.csv
 */

document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    loadSystemHealth();
    loadDatasetStats();
    loadModelMetrics();
    loadRealDatasetSamples();
    initInspectionForm();
});

// Cache for real dataset preset scenarios
let presetSamples = {};

/* --------------------------------------------------------------------------
   Tabs Switching
   -------------------------------------------------------------------------- */
function initTabs() {
    const tabs = document.querySelectorAll('.nav-tab');
    const panes = document.querySelectorAll('.tab-pane');

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            const targetId = tab.getAttribute('data-tab');

            tabs.forEach(t => t.classList.remove('active'));
            panes.forEach(p => p.classList.remove('active'));

            tab.classList.add('active');

            if (targetId === 'console') {
                document.getElementById('paneConsole').classList.add('active');
            } else if (targetId === 'analytics') {
                document.getElementById('paneAnalytics').classList.add('active');
            } else if (targetId === 'dataset') {
                document.getElementById('paneDataset').classList.add('active');
            }
        });
    });
}

/* --------------------------------------------------------------------------
   System Health
   -------------------------------------------------------------------------- */
async function loadSystemHealth() {
    try {
        const res = await fetch('/api/health');
        const data = await res.json();

        const statusPill = document.getElementById('statusPill');
        const statusText = document.getElementById('statusText');

        if (data.status === 'healthy') {
            statusPill.className = 'status-pill status-online';
            statusText.textContent = 'Model Active & Synchronized';
        } else {
            statusPill.className = 'status-pill status-degraded';
            statusText.textContent = 'System Degraded: Model/Dataset Check Needed';
            showToast('Warning: Check dataset or model status', 'error');
        }
    } catch (err) {
        console.error('Failed to query health endpoint:', err);
    }
}

/* --------------------------------------------------------------------------
   Dataset Statistics
   -------------------------------------------------------------------------- */
async function loadDatasetStats() {
    try {
        const res = await fetch('/api/dataset/stats');
        const data = await res.json();

        if (data.total_records) {
            const formattedTotal = data.total_records.toLocaleString();
            document.getElementById('kpiTotalRecords').textContent = formattedTotal;
            document.getElementById('headerRecordCount').textContent = formattedTotal;

            // Render Class Chips
            const chipsContainer = document.getElementById('classChips');
            if (chipsContainer && data.class_distribution) {
                chipsContainer.innerHTML = '';
                const colors = {
                    'allow': 'emerald',
                    'deny': 'red',
                    'drop': 'amber',
                    'reset-both': 'purple'
                };

                for (const [cls, count] of Object.entries(data.class_distribution)) {
                    const pct = data.class_percentages[cls] || 0;
                    const colorClass = colors[cls] || 'emerald';
                    const chip = document.createElement('div');
                    chip.className = 'class-chip';
                    chip.innerHTML = `
                        <span class="pill-dot ${colorClass}"></span>
                        <span>${cls.toUpperCase()}:</span>
                        <strong>${count.toLocaleString()} (${pct}%)</strong>
                    `;
                    chipsContainer.appendChild(chip);
                }
            }
        }
    } catch (err) {
        console.error('Failed to load dataset stats:', err);
    }
}

/* --------------------------------------------------------------------------
   Model Metrics & Confusion Matrix
   -------------------------------------------------------------------------- */
async function loadModelMetrics() {
    try {
        const res = await fetch('/api/model/metrics');
        const data = await res.json();

        // 1. KPI Cards
        if (data.metrics) {
            const accPct = (data.metrics.accuracy * 100).toFixed(2);
            const f1Pct = (data.metrics.f1_weighted * 100).toFixed(2);
            document.getElementById('kpiAccuracy').textContent = `${accPct}%`;
            document.getElementById('kpiWeightedF1').textContent = `${f1Pct}%`;
        }

        // 2. Per-class metrics table
        const tbody = document.getElementById('metricsTableBody');
        if (tbody && data.per_class_metrics) {
            tbody.innerHTML = '';
            for (const [cls, m] of Object.entries(data.per_class_metrics)) {
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td><strong>${cls.toUpperCase()}</strong></td>
                    <td>${(m.precision * 100).toFixed(2)}%</td>
                    <td>${(m.recall * 100).toFixed(2)}%</td>
                    <td>${(m.f1_score * 100).toFixed(2)}%</td>
                    <td>${m.support.toLocaleString()}</td>
                `;
                tbody.appendChild(tr);
            }
        }

        // 3. Confusion Matrix Grid
        const cmGrid = document.getElementById('confusionMatrixGrid');
        if (cmGrid && data.confusion_matrix) {
            const labels = data.confusion_matrix.labels;
            const matrix = data.confusion_matrix.matrix;

            cmGrid.innerHTML = '';

            // Top-left blank cell
            const cornerCell = document.createElement('div');
            cornerCell.className = 'cm-cell cm-header-cell';
            cornerCell.textContent = 'ACT \\ PRED';
            cmGrid.appendChild(cornerCell);

            // Column Headers
            labels.forEach(lbl => {
                const hCell = document.createElement('div');
                hCell.className = 'cm-cell cm-header-cell';
                hCell.textContent = lbl.toUpperCase();
                cmGrid.appendChild(hCell);
            });

            // Rows
            labels.forEach((actualLabel, rIdx) => {
                // Row header
                const rHeader = document.createElement('div');
                rHeader.className = 'cm-cell cm-header-cell';
                rHeader.textContent = actualLabel.toUpperCase();
                cmGrid.appendChild(rHeader);

                // Matrix values
                labels.forEach((predLabel, cIdx) => {
                    const count = matrix[rIdx][cIdx];
                    const cell = document.createElement('div');
                    cell.className = 'cm-cell';

                    if (rIdx === cIdx) {
                        cell.classList.add('cm-diag-cell');
                    } else if (count > 0) {
                        cell.classList.add('cm-error-cell');
                    } else {
                        cell.classList.add('cm-off-cell');
                    }

                    cell.innerHTML = `<span>${count.toLocaleString()}</span>`;
                    cmGrid.appendChild(cell);
                });
            });
        }

        // 4. Feature Importance List
        const fiList = document.getElementById('featureImportanceList');
        if (fiList && data.feature_importance) {
            fiList.innerHTML = '';
            data.feature_importance.forEach(item => {
                const row = document.createElement('div');
                row.className = 'fi-row';
                row.innerHTML = `
                    <div class="fi-name" title="${item.feature}">${item.feature}</div>
                    <div class="fi-bar-container">
                        <div class="fi-bar-fill" style="width: ${Math.max(item.percentage * 3.5, 3)}%;"></div>
                    </div>
                    <div class="fi-pct">${item.percentage}%</div>
                `;
                fiList.appendChild(row);
            });
        }

    } catch (err) {
        console.error('Failed to load model metrics:', err);
    }
}

/* --------------------------------------------------------------------------
   Real Dataset Samples & Presets
   -------------------------------------------------------------------------- */
async function loadRealDatasetSamples() {
    try {
        // 1. Fetch class representative samples
        const repRes = await fetch('/api/dataset/sample?mode=class_reps');
        const repData = await repRes.json();

        if (repData.samples) {
            repData.samples.forEach(s => {
                presetSamples[s.Action] = s;
            });
            setupPresetButtons();
        }

        // 2. Fetch browsable rows for the dataset table
        const tableRes = await fetch('/api/dataset/sample?mode=browse&limit=15');
        const tableData = await tableRes.json();

        const tableBody = document.getElementById('sampleTableBody');
        if (tableBody && tableData.samples) {
            tableBody.innerHTML = '';
            tableData.samples.forEach(row => {
                const tr = document.createElement('tr');
                const actionBadgeClass = getBadgeColorClass(row.Action);
                tr.innerHTML = `
                    <td>#${row.row_id}</td>
                    <td>${row['Source Port']}</td>
                    <td>${row['Destination Port']}</td>
                    <td>${row['NAT Source Port']}</td>
                    <td>${row['NAT Destination Port']}</td>
                    <td>${row['Bytes'].toLocaleString()}</td>
                    <td>${row['Packets']}</td>
                    <td>${row['Elapsed Time (sec)']}s</td>
                    <td><span class="badge-pill ${actionBadgeClass}">${row.Action.toUpperCase()}</span></td>
                    <td>
                        <button class="btn-table-action" onclick="loadRowIntoConsole(${JSON.stringify(row).replace(/"/g, '&quot;')})">
                            Inspect
                        </button>
                    </td>
                `;
                tableBody.appendChild(tr);
            });
        }

    } catch (err) {
        console.error('Failed to load samples:', err);
    }
}

function getBadgeColorClass(action) {
    if (action === 'allow') return 'emerald-badge';
    if (action === 'deny') return 'badge-pill red-text';
    if (action === 'drop') return 'badge-pill amber-text';
    return 'badge-pill purple-text';
}

function setupPresetButtons() {
    const buttons = document.querySelectorAll('.btn-preset');
    buttons.forEach(btn => {
        btn.addEventListener('click', () => {
            const presetKey = btn.getAttribute('data-preset');
            const sample = presetSamples[presetKey];
            if (sample) {
                fillFormWithRecord(sample);
                showToast(`Loaded ${presetKey.toUpperCase()} sample into console`, 'success');
                // Auto trigger inspection
                triggerInspection();
            }
        });
    });
}

window.loadRowIntoConsole = function(row) {
    fillFormWithRecord(row);
    // Switch to console tab
    document.getElementById('tabBtnConsole').click();
    showToast(`Loaded row #${row.row_id} into inspection console`, 'success');
    triggerInspection();
};

function fillFormWithRecord(record) {
    const fieldMap = {
        'srcPort': 'Source Port',
        'dstPort': 'Destination Port',
        'natSrcPort': 'NAT Source Port',
        'natDstPort': 'NAT Destination Port',
        'bytes': 'Bytes',
        'bytesSent': 'Bytes Sent',
        'bytesRecv': 'Bytes Received',
        'packets': 'Packets',
        'elapsedTime': 'Elapsed Time (sec)',
        'pktsSent': 'pkts_sent',
        'pktsRecv': 'pkts_received'
    };

    for (const [elemId, colName] of Object.entries(fieldMap)) {
        const input = document.getElementById(elemId);
        if (input && record[colName] !== undefined) {
            input.value = record[colName];
        }
    }
}

/* --------------------------------------------------------------------------
   Inspection Form & Prediction Execution
   -------------------------------------------------------------------------- */
function initInspectionForm() {
    const form = document.getElementById('inspectionForm');
    const btnReset = document.getElementById('btnResetForm');

    form.addEventListener('submit', (e) => {
        e.preventDefault();
        triggerInspection();
    });

    btnReset.addEventListener('click', () => {
        form.reset();
        showToast('Form reset to defaults', 'success');
    });

    // Run initial prediction with default values
    setTimeout(triggerInspection, 300);
}

async function triggerInspection() {
    const form = document.getElementById('inspectionForm');
    const formData = new FormData(form);

    const payload = {
        "Source Port": parseInt(formData.get('Source Port'), 10),
        "Destination Port": parseInt(formData.get('Destination Port'), 10),
        "NAT Source Port": parseInt(formData.get('NAT Source Port'), 10),
        "NAT Destination Port": parseInt(formData.get('NAT Destination Port'), 10),
        "Bytes": parseInt(formData.get('Bytes'), 10),
        "Bytes Sent": parseInt(formData.get('Bytes Sent'), 10),
        "Bytes Received": parseInt(formData.get('Bytes Received'), 10),
        "Packets": parseInt(formData.get('Packets'), 10),
        "Elapsed Time (sec)": parseFloat(formData.get('Elapsed Time (sec)')),
        "pkts_sent": parseInt(formData.get('pkts_sent'), 10),
        "pkts_received": parseInt(formData.get('pkts_received'), 10)
    };

    const startTime = performance.now();

    try {
        const res = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const elapsed = (performance.now() - startTime).toFixed(1);
        document.getElementById('inferenceTime').textContent = `${elapsed} ms`;

        if (!res.ok) {
            const err = await res.json();
            showToast(`Inspection failed: ${err.error}`, 'error');
            return;
        }

        const data = await res.json();
        renderPredictionVerdict(data);

    } catch (err) {
        console.error('Inspection request error:', err);
        showToast('Network error while inspecting traffic', 'error');
    }
}

function renderPredictionVerdict(data) {
    const action = data.predicted_action;
    const confPct = data.confidence_percentage.toFixed(1);

    // 1. Hero Badge
    const badge = document.getElementById('verdictBadge');
    const badgeText = document.getElementById('verdictText');
    const badgeIcon = document.getElementById('badgeIcon');

    badge.className = `action-badge-hero ${action}`;
    badgeText.textContent = action.toUpperCase();

    const icons = {
        'allow': '✓',
        'deny': '✕',
        'drop': '⚠',
        'reset-both': '⚡'
    };
    badgeIcon.textContent = icons[action] || '●';

    // 2. Confidence Meter
    document.getElementById('confidenceValue').textContent = `${confPct}%`;
    document.getElementById('confidenceBar').style.width = `${confPct}%`;

    // 3. Advice Box
    const adviceBox = document.getElementById('adviceBox');
    const adviceTitle = document.getElementById('adviceTitle');
    const adviceDesc = document.getElementById('adviceDesc');

    if (data.analysis) {
        adviceTitle.textContent = `${data.analysis.verdict} (${data.analysis.risk_level})`;
        adviceDesc.textContent = data.analysis.action_advice;

        const borderColors = {
            'allow': 'var(--accent-emerald)',
            'deny': 'var(--accent-crimson)',
            'drop': 'var(--accent-amber)',
            'reset-both': 'var(--accent-purple)'
        };
        adviceBox.style.borderLeftColor = borderColors[action] || 'var(--accent-emerald)';
    }

    // 4. Softmax Probability Bars
    const probs = data.probabilities || {};
    updateProbBar('Allow', probs['allow']);
    updateProbBar('Deny', probs['deny']);
    updateProbBar('Drop', probs['drop']);
    updateProbBar('ResetBoth', probs['reset-both']);
}

function updateProbBar(clsKey, rawProb) {
    const probValElem = document.getElementById(`probVal${clsKey}`);
    const probFillElem = document.getElementById(`probFill${clsKey}`);

    const pct = rawProb !== undefined ? (rawProb * 100).toFixed(1) : '0.0';
    if (probValElem) probValElem.textContent = `${pct}%`;
    if (probFillElem) probFillElem.style.width = `${pct}%`;
}

/* --------------------------------------------------------------------------
   Toast Utility
   -------------------------------------------------------------------------- */
function showToast(message, type = 'success') {
    const container = document.getElementById('toastContainer');
    if (!container) return;

    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `
        <span>${type === 'success' ? '✓' : '⚠'}</span>
        <span>${message}</span>
    `;

    container.appendChild(toast);

    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateY(10px)';
        setTimeout(() => toast.remove(), 300);
    }, 3200);
}
