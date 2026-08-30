// Global State Object
const state = {
    // Current questionnaire selections
    questionnaire: {
        intensity: 'Medium',
        stability: 'Medium',
        body_position: 'Standing',
        rotation: 'Moderate',
        movement_pattern: 'Regular'
    },
    // Activity details loaded from backend
    activities: {},
    // Flag to prevent double submission
    isPredicting: false
};

// Server API URL Base helper (for GitHub Pages -> Render backend integration)
function getStoredServerUrl() {
    return localStorage.getItem('har_render_server_url') || '';
}

function setStoredServerUrl(url) {
    if (url && url.trim().length > 0) {
        let formatted = url.trim().replace(/\/+$/, '');
        if (!formatted.startsWith('http://') && !formatted.startsWith('https://')) {
            formatted = 'https://' + formatted;
        }
        localStorage.setItem('har_render_server_url', formatted);
    } else {
        localStorage.removeItem('har_render_server_url');
    }
}

function getApiUrl(path) {
    const stored = getStoredServerUrl();
    if (stored) {
        return `${stored}${path.startsWith('/') ? path : '/' + path}`;
    }
    
    // Auto-detect when running directly from local filesystem (file:// protocol)
    if (window.location.protocol === 'file:' || !window.location.hostname) {
        const localDefault = 'http://127.0.0.1:5000';
        return `${localDefault}${path.startsWith('/') ? path : '/' + path}`;
    }
    
    // Auto-detect when hosted on GitHub Pages or custom domain
    const host = window.location.hostname;
    if (host.includes('github.io') || host.includes('invytra.in')) {
        const defaultRenderUrl = 'https://human-activity-recognition-for-aysha.onrender.com';
        return `${defaultRenderUrl}${path.startsWith('/') ? path : '/' + path}`;
    }
    
    return path;
}

// Activity Emojis Map
const activityEmojis = {
    'WALKING': '🚶',
    'WALKING_UPSTAIRS': '⬆️',
    'WALKING_DOWNSTAIRS': '⬇️',
    'SITTING': '🪑',
    'STANDING': '🧍',
    'LAYING': '🛏️'
};

// ========================================================
// LIVE SENSOR MODE
// ========================================================

const liveSensor = {
    WINDOW_SIZE: 128,          // 128 samples at 50 Hz = 2.56 seconds
    accBuffer: [],             // [{x, y, z, t}]
    gyrBuffer: [],             // [{x, y, z, t}]
    
    active: false,
    paused: false,
    predictionInterval: null,
    predictionIntervalMs: 1000,
    
    rotationMatrix: null,      // 3x3 array if calibrated
    predictionHistory: [],     // Rolling buffer of last 5 prediction objects for temporal smoothing
    activityHistory: [],       // Log of recent activities with timestamps

    elements: {},
};

function initLiveSensor() {
    liveSensor.elements = {
        supportNote: document.getElementById('live-support-note'),
        supportText: document.getElementById('live-support-text'),
        permState: document.getElementById('live-permission-state'),
        activeState: document.getElementById('live-active-state'),
        requestBtn: document.getElementById('btn-request-sensor'),
        errorMsg: document.getElementById('live-sensor-error'),
        toggleBtn: document.getElementById('btn-live-toggle'),
        calibrateBtn: document.getElementById('btn-calibrate'),
        debugBtn: document.getElementById('btn-toggle-debug'),
        sampleCount: document.getElementById('live-sample-count'),
        windowBar: document.getElementById('live-window-bar'),
        
        statusAcc: document.getElementById('status-acc'),
        statusGyro: document.getElementById('status-gyro'),
        statusSampling: document.getElementById('status-sampling'),
        statusWindowCount: document.getElementById('status-window-count'),
        
        accX: document.getElementById('live-acc-x'),
        accY: document.getElementById('live-acc-y'),
        accZ: document.getElementById('live-acc-z'),
        accBarX: document.getElementById('live-acc-bar-x'),
        accBarY: document.getElementById('live-acc-bar-y'),
        accBarZ: document.getElementById('live-acc-bar-z'),
        
        gyrX: document.getElementById('live-gyr-x'),
        gyrY: document.getElementById('live-gyr-y'),
        gyrZ: document.getElementById('live-gyr-z'),
        gyrBarX: document.getElementById('live-gyr-bar-x'),
        gyrBarY: document.getElementById('live-gyr-bar-y'),
        gyrBarZ: document.getElementById('live-gyr-bar-z'),
        
        resultPlaceholder: document.getElementById('live-result-placeholder'),
        resultDisplay: document.getElementById('live-result-display'),
        activityName: document.getElementById('live-activity-name'),
        confidenceBar: document.getElementById('live-confidence-bar'),
        confidenceText: document.getElementById('live-confidence-text'),
        probChart: document.getElementById('live-probabilities-chart'),
        modelName: document.getElementById('live-model-name'),
        historyList: document.getElementById('live-history-list'),
        
        // Debug Card
        debugCard: document.getElementById('debug-info-card'),
        dbgSamples: document.getElementById('dbg-samples'),
        dbgRate: document.getElementById('dbg-rate'),
        dbgWindow: document.getElementById('dbg-window'),
        dbgFeatCount: document.getElementById('dbg-feat-count'),
        dbgFeatExpected: document.getElementById('dbg-feat-expected'),
        dbgModel: document.getElementById('dbg-model'),
        dbgScaler: document.getElementById('dbg-scaler'),
        dbgCalibration: document.getElementById('dbg-calibration'),
    };

    if (window.DeviceMotionEvent) {
        liveSensor.elements.supportNote.className = 'sensor-support-note supported';
        liveSensor.elements.supportText.textContent = '✓ DeviceMotion sensor API supported on this device/browser.';
    } else {
        liveSensor.elements.supportNote.className = 'sensor-support-note not-supported';
        liveSensor.elements.supportText.textContent = '✗ DeviceMotion API is not available. Use a mobile browser (Chrome/Safari on phone).';
        liveSensor.elements.requestBtn.disabled = true;
    }

    liveSensor.elements.requestBtn.addEventListener('click', requestSensorAccess);
    if (liveSensor.elements.toggleBtn) liveSensor.elements.toggleBtn.addEventListener('click', toggleSensorPause);
    if (liveSensor.elements.calibrateBtn) liveSensor.elements.calibrateBtn.addEventListener('click', triggerCalibration);
    if (liveSensor.elements.debugBtn) liveSensor.elements.debugBtn.addEventListener('click', toggleDebugMode);
}

async function requestSensorAccess() {
    const btn = liveSensor.elements.requestBtn;
    btn.disabled = true;
    btn.textContent = 'Requesting...';
    
    try {
        if (typeof DeviceMotionEvent !== 'undefined' && typeof DeviceMotionEvent.requestPermission === 'function') {
            const motionPermission = await DeviceMotionEvent.requestPermission();
            if (motionPermission !== 'granted') {
                showLiveError('Sensor permission denied. Please allow motion sensor access in browser settings.');
                btn.disabled = false;
                btn.textContent = 'REQUEST SENSOR ACCESS';
                return;
            }
        }
        
        window.addEventListener('devicemotion', handleDeviceMotion, true);
        
        liveSensor.elements.permState.classList.add('hidden');
        liveSensor.elements.activeState.classList.remove('hidden');
        liveSensor.active = true;
        
        liveSensor.predictionInterval = setInterval(submitLivePrediction, liveSensor.predictionIntervalMs);
        
    } catch (e) {
        console.error(e);
        showLiveError('Failed to access sensors: ' + e.message + '. Try opening on your phone.');
        btn.disabled = false;
        btn.textContent = 'REQUEST SENSOR ACCESS';
    }
}

function handleDeviceMotion(event) {
    if (liveSensor.paused) return;
    
    const acc = event.accelerationIncludingGravity || event.acceleration;
    const rot = event.rotationRate;
    const now = performance.now();
    
    if (!acc) return;
    
    const ax = acc.x || 0;
    const ay = acc.y || 0;
    const az = acc.z || 0;
    
    liveSensor.accBuffer.push({ x: ax, y: ay, z: az, t: now });
    if (liveSensor.accBuffer.length > liveSensor.WINDOW_SIZE * 2) {
        liveSensor.accBuffer.shift();
    }
    
    let gx = 0, gy = 0, gz = 0;
    if (rot) {
        gx = (rot.alpha || 0) * Math.PI / 180;
        gy = (rot.beta  || 0) * Math.PI / 180;
        gz = (rot.gamma || 0) * Math.PI / 180;
    }
    liveSensor.gyrBuffer.push({ x: gx, y: gy, z: gz, t: now });
    if (liveSensor.gyrBuffer.length > liveSensor.WINDOW_SIZE * 2) {
        liveSensor.gyrBuffer.shift();
    }
    
    updateAxisDisplay(ax, ay, az, 'acc');
    updateAxisDisplay(gx, gy, gz, 'gyr');
    
    const currentSamples = Math.min(liveSensor.WINDOW_SIZE, liveSensor.accBuffer.length);
    const pct = Math.min(100, (currentSamples / liveSensor.WINDOW_SIZE) * 100);
    
    if (liveSensor.elements.windowBar) liveSensor.elements.windowBar.style.width = pct + '%';
    if (liveSensor.elements.sampleCount) liveSensor.elements.sampleCount.textContent = `${currentSamples} / ${liveSensor.WINDOW_SIZE} samples`;
    if (liveSensor.elements.statusWindowCount) liveSensor.elements.statusWindowCount.textContent = `${currentSamples} / ${liveSensor.WINDOW_SIZE}`;
}

function updateAxisDisplay(x, y, z, type) {
    const maxVal = type === 'acc' ? 15 : 5;
    const toBar = (v) => Math.max(0, Math.min(100, ((v + maxVal) / (2 * maxVal)) * 100));
    
    const el = liveSensor.elements;
    if (type === 'acc' && el.accX) {
        el.accX.textContent = x.toFixed(2);
        el.accY.textContent = y.toFixed(2);
        el.accZ.textContent = z.toFixed(2);
        el.accBarX.style.width = toBar(x) + '%';
        el.accBarY.style.width = toBar(y) + '%';
        el.accBarZ.style.width = toBar(z) + '%';
    } else if (type === 'gyr' && el.gyrX) {
        el.gyrX.textContent = x.toFixed(3);
        el.gyrY.textContent = y.toFixed(3);
        el.gyrZ.textContent = z.toFixed(3);
        el.gyrBarX.style.width = toBar(x) + '%';
        el.gyrBarY.style.width = toBar(y) + '%';
        el.gyrBarZ.style.width = toBar(z) + '%';
    }
}

function toggleSensorPause() {
    liveSensor.paused = !liveSensor.paused;
    const btn = liveSensor.elements.toggleBtn;
    const statusSampling = liveSensor.elements.statusSampling;
    
    if (liveSensor.paused) {
        btn.textContent = '▶ Start Detection';
        btn.classList.remove('btn-secondary');
        btn.classList.add('btn-primary');
        if (statusSampling) {
            statusSampling.textContent = 'Paused';
            statusSampling.className = 'chip-val badge-paused';
            statusSampling.style.color = '#eab308';
        }
        clearInterval(liveSensor.predictionInterval);
    } else {
        btn.textContent = '⏹ Stop Detection';
        btn.classList.remove('btn-primary');
        btn.classList.add('btn-secondary');
        if (statusSampling) {
            statusSampling.textContent = 'Active (50 Hz)';
            statusSampling.className = 'chip-val badge-active';
            statusSampling.style.color = '#2563eb';
        }
        liveSensor.predictionInterval = setInterval(submitLivePrediction, liveSensor.predictionIntervalMs);
    }
}

async function triggerCalibration() {
    if (liveSensor.accBuffer.length < 20) {
        alert("Please wait a moment for sensor samples to accumulate before calibrating.");
        return;
    }
    
    const btn = liveSensor.elements.calibrateBtn;
    btn.disabled = true;
    btn.textContent = 'Calibrating...';
    
    const recent = liveSensor.accBuffer.slice(-50);
    const mx = recent.reduce((s, v) => s + v.x, 0) / recent.length;
    const my = recent.reduce((s, v) => s + v.y, 0) / recent.length;
    const mz = recent.reduce((s, v) => s + v.z, 0) / recent.length;
    
    try {
        const response = await fetch(getApiUrl('/api/calibrate'), {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ x: mx, y: my, z: mz })
        });
        
        const data = await response.json();
        if (data.status === 'success') {
            liveSensor.rotationMatrix = data.rotation_matrix;
            alert("✓ Phone orientation calibrated successfully! Sensor axes aligned with training reference.");
            if (liveSensor.elements.dbgCalibration) {
                liveSensor.elements.dbgCalibration.textContent = 'Active (3D Rodrigues Rotation Matrix)';
                liveSensor.elements.dbgCalibration.style.color = '#16a34a';
            }
        }
    } catch (e) {
        console.error("Calibration error:", e);
        alert("Failed to calibrate orientation.");
    } finally {
        btn.disabled = false;
        btn.textContent = '📐 Auto-Calibrate';
    }
}

function toggleDebugMode() {
    const card = liveSensor.elements.debugCard;
    if (card) {
        card.classList.toggle('hidden');
    }
}

async function submitLivePrediction() {
    if (liveSensor.accBuffer.length < 30) return;
    
    const accSamples = liveSensor.accBuffer.slice(-128);
    const gyrSamples = liveSensor.gyrBuffer.slice(-128);
    
    const payload = {
        acc: accSamples.map(s => ({ x: s.x, y: s.y, z: s.z })),
        gyro: gyrSamples.map(s => ({ x: s.x, y: s.y, z: s.z })),
        rotation_matrix: liveSensor.rotationMatrix
    };
    
    try {
        const response = await fetch(getApiUrl('/api/live-predict'), {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        
        const data = await response.json();
        if (response.ok && data.activity) {
            
            liveSensor.predictionHistory.push(data);
            if (liveSensor.predictionHistory.length > 5) {
                liveSensor.predictionHistory.shift();
            }
            
            const smoothedResult = applyTemporalSmoothing(liveSensor.predictionHistory);
            
            liveSensor.elements.resultPlaceholder.classList.add('hidden');
            liveSensor.elements.resultDisplay.classList.remove('hidden');
            
            const isLowConfidence = smoothedResult.confidence < 0.70;
            
            if (isLowConfidence) {
                liveSensor.elements.activityName.textContent = '🔍 Detecting... (Low Confidence)';
                liveSensor.elements.activityName.style.color = '#eab308';
            } else {
                const emoji = activityEmojis[smoothedResult.activity] || '🏃';
                liveSensor.elements.activityName.textContent = `${emoji} ${smoothedResult.activity.replace(/_/g, ' ')}`;
                liveSensor.elements.activityName.style.color = 'var(--text-primary)';
                
                addActivityHistoryEntry(smoothedResult.activity, smoothedResult.confidence);
            }
            
            const confPct = (smoothedResult.confidence * 100).toFixed(1) + '%';
            liveSensor.elements.confidenceBar.style.width = confPct;
            liveSensor.elements.confidenceText.textContent = confPct;
            
            renderProbabilityChart(liveSensor.elements.probChart, smoothedResult.top_predictions, smoothedResult.activity);
            if (liveSensor.elements.modelName) liveSensor.elements.modelName.textContent = data.model;
            
            if (liveSensor.elements.dbgSamples) liveSensor.elements.dbgSamples.textContent = accSamples.length;
            if (liveSensor.elements.dbgFeatCount) liveSensor.elements.dbgFeatCount.textContent = data.features_extracted_count || 138;
            if (liveSensor.elements.dbgModel) liveSensor.elements.dbgModel.textContent = data.model;
        }
    } catch (e) {
        console.warn('Live prediction submit error:', e);
    }
}

function applyTemporalSmoothing(history) {
    if (history.length === 1) return history[0];
    
    const probSums = {};
    const count = history.length;
    
    history.forEach(item => {
        item.top_predictions.forEach(p => {
            probSums[p.activity] = (probSums[p.activity] || 0) + p.probability;
        });
    });
    
    const smoothedPredictions = [];
    let bestActivity = '';
    let maxProb = -1;
    
    for (const [act, sum] of Object.entries(probSums)) {
        const avg = sum / count;
        smoothedPredictions.push({ activity: act, probability: avg });
        if (avg > maxProb) {
            maxProb = avg;
            bestActivity = act;
        }
    }
    smoothedPredictions.sort((a, b) => b.probability - a.probability);
    
    return {
        activity: bestActivity,
        confidence: maxProb,
        top_predictions: smoothedPredictions
    };
}

function addActivityHistoryEntry(activity, confidence) {
    const list = liveSensor.elements.historyList;
    if (!list) return;
    
    const now = new Date();
    const timeStr = now.toTimeString().split(' ')[0];
    const emoji = activityEmojis[activity] || '🏃';
    
    if (liveSensor.activityHistory.length > 0) {
        const last = liveSensor.activityHistory[0];
        if (last.activity === activity && (now - last.time) < 3000) {
            return;
        }
    }
    
    liveSensor.activityHistory.unshift({ activity, time: now });
    if (liveSensor.activityHistory.length > 10) liveSensor.activityHistory.pop();
    
    const firstChild = list.querySelector('li');
    if (firstChild && firstChild.style.fontStyle === 'italic') {
        list.innerHTML = '';
    }
    
    const li = document.createElement('li');
    li.style.cssText = 'display:flex; justify-content:space-between; padding:0.3rem 0; border-bottom:1px solid #f1f5f9;';
    li.innerHTML = `
        <span><span style="font-family:monospace; color:#64748b; margin-right:0.5rem;">${timeStr}</span> <strong>${emoji} ${activity.replace(/_/g, ' ')}</strong></span>
        <span style="color:#2563eb; font-weight:600;">${(confidence * 100).toFixed(1)}%</span>
    `;
    
    list.insertBefore(li, list.firstChild);
}

function showLiveError(msg) {
    const el = liveSensor.elements.errorMsg;
    if (el) {
        el.textContent = msg;
        el.classList.remove('hidden');
    }
}

// Initialize Application on Page Load
document.addEventListener('DOMContentLoaded', () => {
    initApp();
    setupEventHandlers();
    initLiveSensor();
});

// Main Initialization Sequence
async function initApp() {
    const currentApi = getApiUrl('/api/status');
    updateLoadingStep('step-backend', 'running', `Connecting to backend (${currentApi})...`);
    
    try {
        const response = await fetch(currentApi);
        const data = await response.json();
        
        // Step 1: Connection Check
        updateLoadingStep('step-backend', 'success', 'Connected to backend server.');
        updateServerStatusUI(true, 'Server Connected');
        
        // Step 2: Dataset Check
        if (data.dataset_verified) {
            updateLoadingStep('step-dataset', 'success', 'UCI HAR Dataset found and verified.');
        } else {
            updateLoadingStep('step-dataset', 'error', 'UCI HAR Dataset folder is missing.');
            showLoadingError('UCI HAR Dataset directory not found on backend. Please verify backend dataset directory.');
            return;
        }

        // Step 3: Model Check
        if (data.model_loaded && data.status === 'ready') {
            updateLoadingStep('step-model', 'success', 'Trained model files loaded successfully.');
            
            // Short delay to show the checkmarks before transition
            setTimeout(async () => {
                // Fade out overlay
                const overlay = document.getElementById('loading-overlay');
                overlay.style.opacity = '0';
                setTimeout(() => overlay.classList.add('hidden'), 500);
                
                // Load secondary app data
                await loadAppData();
            }, 800);
        } else {
            updateLoadingStep('step-model', 'error', 'Trained model files not found.');
            showLoadingError('ML Model file not found. You need to train the model first.', true);
        }
        
    } catch (error) {
        console.error("Initialization error:", error);
        updateLoadingStep('step-backend', 'error', 'Failed to connect to backend.');
        updateServerStatusUI(false, 'Disconnected');
        showLoadingError(`Could not reach backend at ${currentApi}. Please verify your Render Backend URL.`);
    }
}

// Load secondary data from API
async function loadAppData() {
    try {
        // Fetch activities descriptions
        const actRes = await fetch(getApiUrl('/api/activities'));
        const actData = await actRes.json();
        
        // Map activities to a lookup object
        const container = document.getElementById('activities-cards-container');
        if (container) container.innerHTML = '';
        
        actData.forEach(act => {
            state.activities[act.id] = act;
            
            // Build activity card
            if (container) {
                const card = document.createElement('div');
                card.className = 'activity-card';
                card.innerHTML = `
                    <div class="activity-icon">${act.icon}</div>
                    <h3 class="activity-name">${act.name}</h3>
                    <p class="activity-description">${act.description}</p>
                `;
                container.appendChild(card);
            }
        });
        
        // Load model metadata and results dashboard
        await refreshDashboard();
        
    } catch (e) {
        console.error("Error loading app data:", e);
    }
}

// Refresh Model Metadata and Visualization Charts
async function refreshDashboard() {
    try {
        // Load model metadata
        const modelRes = await fetch(getApiUrl('/api/model'));
        if (modelRes.ok) {
            const meta = await modelRes.json();
            
            document.getElementById('meta-train-samples').textContent = meta.training_samples.toLocaleString();
            document.getElementById('meta-test-samples').textContent = meta.testing_samples.toLocaleString();
            document.getElementById('meta-features').textContent = meta.features_count;
            document.getElementById('meta-activities').textContent = meta.activities_count;
            document.getElementById('meta-best-model').textContent = meta.best_model_name;
            document.getElementById('meta-accuracy').textContent = (meta.accuracy * 100).toFixed(2) + '%';
            document.getElementById('meta-precision').textContent = (meta.precision * 100).toFixed(2) + '%';
            document.getElementById('meta-recall').textContent = (meta.recall * 100).toFixed(2) + '%';
            document.getElementById('meta-f1').textContent = (meta.f1_score * 100).toFixed(2) + '%';
            document.getElementById('meta-train-time').textContent = meta.training_time_s.toFixed(2) + ' seconds';
            document.getElementById('meta-pred-time').textContent = (meta.prediction_time_s * 1000).toFixed(2) + ' ms';
        }

        // Load visual plots
        const resultsRes = await fetch(getApiUrl('/api/results'));
        if (resultsRes.ok) {
            const results = await resultsRes.json();
            
            // Set image sources with cache buster to force refresh
            const cb = '?t=' + new Date().getTime();
            document.getElementById('img-confusion-matrix').src = getApiUrl(results.plots.confusion_matrix) + cb;
            document.getElementById('img-model-comparison').src = getApiUrl(results.plots.model_comparison) + cb;
            document.getElementById('img-activity-distribution').src = getApiUrl(results.plots.activity_distribution) + cb;
        }
    } catch (e) {
        console.error("Error refreshing dashboard:", e);
    }
}

function updateServerStatusUI(isConnected, labelText) {
    const dot = document.getElementById('server-status-dot');
    const text = document.getElementById('server-status-text');
    if (dot) dot.style.backgroundColor = isConnected ? '#22c55e' : '#ef4444';
    if (text) text.textContent = labelText || (isConnected ? 'Server Connected' : 'Server Off');
}

function initServerConfigHandlers() {
    const openBtn = document.getElementById('btn-open-server-modal');
    const closeBtn = document.getElementById('btn-close-server-modal');
    const modal = document.getElementById('server-modal');
    const inputUrl = document.getElementById('input-server-url');
    const saveBtn = document.getElementById('btn-save-server-url');
    const resetBtn = document.getElementById('btn-reset-server-url');
    const saveUrlInitBtn = document.getElementById('btn-save-url-init');
    const inputUrlInit = document.getElementById('input-render-url-init');

    if (openBtn && modal) {
        openBtn.addEventListener('click', () => {
            if (inputUrl) inputUrl.value = getStoredServerUrl();
            modal.classList.remove('hidden');
        });
    }

    if (closeBtn && modal) {
        closeBtn.addEventListener('click', () => modal.classList.add('hidden'));
    }

    if (saveBtn) {
        saveBtn.addEventListener('click', () => {
            const val = inputUrl ? inputUrl.value : '';
            setStoredServerUrl(val);
            if (modal) modal.classList.add('hidden');
            hideLoadingError();
            const overlay = document.getElementById('loading-overlay');
            if (overlay) {
                overlay.classList.remove('hidden');
                overlay.style.opacity = '1';
            }
            initApp();
        });
    }

    if (resetBtn) {
        resetBtn.addEventListener('click', () => {
            setStoredServerUrl('');
            if (inputUrl) inputUrl.value = '';
            if (modal) modal.classList.add('hidden');
            hideLoadingError();
            const overlay = document.getElementById('loading-overlay');
            if (overlay) {
                overlay.classList.remove('hidden');
                overlay.style.opacity = '1';
            }
            initApp();
        });
    }

    if (saveUrlInitBtn) {
        saveUrlInitBtn.addEventListener('click', () => {
            const val = inputUrlInit ? inputUrlInit.value : '';
            if (val) {
                setStoredServerUrl(val);
                hideLoadingError();
                initApp();
            } else {
                alert("Please enter a valid Render backend URL (e.g. https://your-app.onrender.com).");
            }
        });
    }
}

// Event Handlers Setup
function setupEventHandlers() {
    initServerConfigHandlers();

    // Retry initialization button
    document.getElementById('btn-retry-init').addEventListener('click', () => {
        hideLoadingError();
        initApp();
    });

    // Train Model directly from Loading overlay
    document.getElementById('btn-train-init').addEventListener('click', async () => {
        hideLoadingError();
        updateLoadingStep('step-model', 'running', 'Training ML pipeline (Logistic, DT, RF, SVM)...');
        await triggerTraining(true);
    });

    // Navigation links scrolling and active state
    const navLinks = document.querySelectorAll('.nav-link');
    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const targetId = link.getAttribute('href');
            const targetSection = document.querySelector(targetId);
            
            if (targetSection) {
                // Remove active class from all
                navLinks.forEach(l => l.classList.remove('active'));
                // Add to clicked
                link.classList.add('active');
                
                // Smooth scroll
                window.scrollTo({
                    top: targetSection.offsetTop - 70,
                    behavior: 'smooth'
                });
            }
        });
    });

    // Hero section "Start Prediction" scroll handler
    const heroBtn = document.querySelector('.hero-actions .btn-primary');
    if (heroBtn) {
        heroBtn.addEventListener('click', (e) => {
            e.preventDefault();
            const predictSection = document.getElementById('predict');
            if (predictSection) {
                window.scrollTo({
                    top: predictSection.offsetTop - 70,
                    behavior: 'smooth'
                });
                
                // Update nav link highlight
                document.querySelectorAll('.nav-link').forEach(l => {
                    l.classList.toggle('active', l.getAttribute('href') === '#predict');
                });
            }
        });
    }

    // Questionnaire segmented control buttons
    const segControls = [
        { id: 'ctrl-intensity', key: 'intensity' },
        { id: 'ctrl-stability', key: 'stability' },
        { id: 'ctrl-position', key: 'body_position' },
        { id: 'ctrl-rotation', key: 'rotation' },
        { id: 'ctrl-pattern', key: 'movement_pattern' }
    ];

    segControls.forEach(ctrl => {
        const container = document.getElementById(ctrl.id);
        if (container) {
            const buttons = container.querySelectorAll('.segment-btn');
            buttons.forEach(btn => {
                btn.addEventListener('click', () => {
                    // Remove active from peers
                    buttons.forEach(b => b.classList.remove('active'));
                    // Add to clicked
                    btn.classList.add('active');
                    // Update state value
                    state.questionnaire[ctrl.key] = btn.getAttribute('data-value');
                });
            });
        }
    });

    // Run prediction button
    document.getElementById('btn-predict').addEventListener('click', runQuestionnairePrediction);

    // Run sample validation button
    document.getElementById('btn-predict-sample').addEventListener('click', runSamplePrediction);

    // Random sample button
    document.getElementById('btn-random-sample').addEventListener('click', () => {
        const randomId = Math.floor(Math.random() * 2947) + 1; // 1-indexed for user
        document.getElementById('input-sample-id').value = randomId;
        runSamplePrediction();
    });

    // Dashboard tab buttons
    const tabBtns = document.querySelectorAll('.tab-btn');
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const tabId = btn.getAttribute('data-tab');
            
            // Deactivate all buttons & content
            tabBtns.forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            
            // Activate selected
            btn.classList.add('active');
            document.getElementById(tabId).classList.add('active');
        });
    });

    // Retrain button on dashboard
    document.getElementById('btn-retrain').addEventListener('click', () => {
        triggerTraining(false);
    });
}

// Perform prediction from Questionnaire selections
async function runQuestionnairePrediction() {
    if (state.isPredicting) return;
    
    const btn = document.getElementById('btn-predict');
    btn.disabled = true;
    btn.textContent = 'ANALYZING SENSOR DATA...';
    state.isPredicting = true;
    
    // Hide previous results
    document.getElementById('result-placeholder').classList.add('hidden');
    document.getElementById('result-success').classList.add('hidden');
    document.getElementById('result-no-match').classList.add('hidden');

    try {
        const response = await fetch(getApiUrl('/api/predict'), {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(state.questionnaire)
        });

        const data = await response.json();
        
        if (data.error === 'INSUFFICIENT_MATCHES') {
            // Show no-match panel
            document.getElementById('result-no-match').classList.remove('hidden');
        } else if (response.ok) {
            // Populate results card
            const activityLabel = data.activity;
            const emoji = activityEmojis[activityLabel] || '🏃';
            
            document.getElementById('result-activity-name').textContent = `${emoji} ${activityLabel.replace(/_/g, ' ')}`;
            
            const confPercent = (data.confidence * 100).toFixed(1) + '%';
            document.getElementById('result-confidence-text').textContent = confPercent;
            document.getElementById('result-confidence-bar').style.width = confPercent;
            
            // Populate matching parameters preview
            document.getElementById('detail-intensity').textContent = state.questionnaire.intensity;
            document.getElementById('detail-stability').textContent = state.questionnaire.stability;
            document.getElementById('detail-position').textContent = state.questionnaire.body_position;
            document.getElementById('detail-rotation').textContent = state.questionnaire.rotation;
            document.getElementById('detail-pattern').textContent = state.questionnaire.movement_pattern;
            document.getElementById('detail-matches').textContent = data.matched_samples;
            document.getElementById('result-model-used').textContent = data.model;
            
            // Render probabilities bars
            renderProbabilityChart(document.getElementById('probabilities-chart'), data.top_predictions, activityLabel);
            
            // Display success card
            document.getElementById('result-success').classList.remove('hidden');
        } else {
            alert(`Prediction error: ${data.error || 'Server error'}`);
            document.getElementById('result-placeholder').classList.remove('hidden');
        }
    } catch (e) {
        console.error(e);
        alert('Network connection error when submitting prediction request.');
        document.getElementById('result-placeholder').classList.remove('hidden');
    } finally {
        btn.disabled = false;
        btn.textContent = 'ANALYZE ACTIVITY';
        state.isPredicting = false;
    }
}

// Perform prediction on selected test index
async function runSamplePrediction() {
    const input = document.getElementById('input-sample-id');
    const userVal = parseInt(input.value, 10);
    
    if (isNaN(userVal) || userVal < 1 || userVal > 2947) {
        alert("Please enter a valid sample index between 1 and 2,947.");
        return;
    }
    
    const sampleId = userVal - 1; // Convert 1-indexed user input to 0-indexed API ID
    
    const btn = document.getElementById('btn-predict-sample');
    btn.disabled = true;
    btn.textContent = 'LOADING SAMPLE...';
    
    document.getElementById('sample-placeholder').classList.add('hidden');
    document.getElementById('sample-result-display').classList.add('hidden');
    
    try {
        const response = await fetch(getApiUrl(`/api/sample/${sampleId}`));
        const data = await response.json();
        
        if (response.ok) {
            document.getElementById('lbl-sample-id').textContent = userVal;
            document.getElementById('lbl-features-count').textContent = data.features_count;
            
            document.getElementById('lbl-predicted-activity').textContent = data.predicted.replace(/_/g, ' ');
            document.getElementById('lbl-sample-confidence').textContent = (data.confidence * 100).toFixed(2) + '%';
            document.getElementById('lbl-actual-activity').textContent = data.actual.replace(/_/g, ' ');
            
            // Update validation status checkmark/cross badge
            const badge = document.getElementById('lbl-validation-status');
            if (data.correct) {
                badge.className = 'validation-status status-correct';
                badge.textContent = '✓ CORRECT';
            } else {
                badge.className = 'validation-status status-incorrect';
                badge.textContent = '✗ INCORRECT';
            }
            
            // Render probability chart
            renderProbabilityChart(document.getElementById('sample-probabilities'), data.top_predictions, data.predicted);
            
            // Show result block
            document.getElementById('sample-result-display').classList.remove('hidden');
        } else {
            alert(`Sample error: ${data.error || 'Server error'}`);
            document.getElementById('sample-placeholder').classList.remove('hidden');
        }
    } catch (e) {
        console.error(e);
        alert("Failed to analyze sample. Please check connection.");
        document.getElementById('sample-placeholder').classList.remove('hidden');
    } finally {
        btn.disabled = false;
        btn.textContent = 'ANALYZE SENSOR RECORDING';
    }
}

// Render the Probability Bars in the result sections
function renderProbabilityChart(containerElement, predictions, winningClassLabel) {
    containerElement.innerHTML = '';
    
    predictions.forEach(pred => {
        const isWinner = pred.activity === winningClassLabel;
        const emoji = activityEmojis[pred.activity] || '';
        const pct = (pred.probability * 100).toFixed(1) + '%';
        
        const row = document.createElement('div');
        row.className = 'prob-row';
        row.innerHTML = `
            <span class="prob-name">${emoji} ${pred.activity.replace(/_/g, ' ')}</span>
            <div class="prob-bar-outer">
                <div class="prob-bar-inner ${isWinner ? 'highlight' : ''}" style="width: ${pct}"></div>
            </div>
            <span class="prob-val">${pct}</span>
        `;
        
        containerElement.appendChild(row);
    });
}

// Trigger Pipeline Training
async function triggerTraining(isFromLoadingOverlay = false) {
    const btn = document.getElementById(isFromLoadingOverlay ? 'btn-train-init' : 'btn-retrain');
    const statusDiv = document.getElementById(isFromLoadingOverlay ? 'loading-error' : 'retrain-status');
    
    btn.disabled = true;
    if (isFromLoadingOverlay) {
        btn.textContent = 'TRAINING MODEL...';
    } else {
        btn.classList.add('hidden');
        statusDiv.classList.remove('hidden');
    }

    try {
        const response = await fetch(getApiUrl('/api/train'), { method: 'POST' });
        const data = await response.json();
        
        if (response.ok && data.status === 'success') {
            if (isFromLoadingOverlay) {
                // Successfully trained from loading overlay, restart check
                initApp();
            } else {
                // Successfully trained from dashboard, update metadata & visual charts
                await refreshDashboard();
                alert("Model pipeline retrained successfully! Selected the new best classifier based on test accuracy.");
            }
        } else {
            alert(`Training error: ${data.message || 'Error occurred'}`);
        }
    } catch (e) {
        console.error(e);
        alert("Training request failed. Please check backend connection.");
    } finally {
        btn.disabled = false;
        if (isFromLoadingOverlay) {
            btn.textContent = 'Train Model Now';
        } else {
            btn.classList.remove('hidden');
            statusDiv.classList.add('hidden');
        }
    }
}

// Loading overlay step display helper
function updateLoadingStep(elementId, status, text) {
    const el = document.getElementById(elementId);
    if (!el) return;
    
    el.className = status; // success, error, pending, running
    if (text) el.textContent = text;
}

// Display error block in loading screen
function showLoadingError(message, showTrainBtn = false) {
    document.getElementById('loading-error').classList.remove('hidden');
    document.getElementById('error-message').textContent = message;
    
    // Toggle the dynamic train button
    const trainBtn = document.getElementById('btn-train-init');
    trainBtn.classList.toggle('hidden', !showTrainBtn);
}

// Hide error block in loading screen
function hideLoadingError() {
    document.getElementById('loading-error').classList.add('hidden');
}
