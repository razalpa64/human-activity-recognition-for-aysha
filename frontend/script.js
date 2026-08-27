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
    // Raw sensor reading buffers (sliding window of up to WINDOW_SIZE samples)
    WINDOW_SIZE: 50,
    accBuffer: [],     // [{x, y, z}]
    gyrBuffer: [],     // [{x, y, z}]
    gravBuffer: [],    // [{x, y, z}]  (from deviceorientation)
    
    active: false,
    paused: false,
    predictionInterval: null,
    predictionIntervalMs: 1500,
    
    // DOM elements cache
    elements: {},
};

function initLiveSensor() {
    // Cache element references
    liveSensor.elements = {
        supportNote: document.getElementById('live-support-note'),
        supportText: document.getElementById('live-support-text'),
        permState: document.getElementById('live-permission-state'),
        activeState: document.getElementById('live-active-state'),
        requestBtn: document.getElementById('btn-request-sensor'),
        errorMsg: document.getElementById('live-sensor-error'),
        toggleBtn: document.getElementById('btn-live-toggle'),
        sampleCount: document.getElementById('live-sample-count'),
        windowBar: document.getElementById('live-window-bar'),
        
        // Acc readings
        accX: document.getElementById('live-acc-x'),
        accY: document.getElementById('live-acc-y'),
        accZ: document.getElementById('live-acc-z'),
        accBarX: document.getElementById('live-acc-bar-x'),
        accBarY: document.getElementById('live-acc-bar-y'),
        accBarZ: document.getElementById('live-acc-bar-z'),
        
        // Gyro readings
        gyrX: document.getElementById('live-gyr-x'),
        gyrY: document.getElementById('live-gyr-y'),
        gyrZ: document.getElementById('live-gyr-z'),
        gyrBarX: document.getElementById('live-gyr-bar-x'),
        gyrBarY: document.getElementById('live-gyr-bar-y'),
        gyrBarZ: document.getElementById('live-gyr-bar-z'),
        
        // Prediction result
        resultPlaceholder: document.getElementById('live-result-placeholder'),
        resultDisplay: document.getElementById('live-result-display'),
        activityName: document.getElementById('live-activity-name'),
        confidenceBar: document.getElementById('live-confidence-bar'),
        confidenceText: document.getElementById('live-confidence-text'),
        probChart: document.getElementById('live-probabilities-chart'),
        matchedCount: document.getElementById('live-matched'),
        modelName: document.getElementById('live-model-name'),
    };

    // Check if DeviceMotion is supported
    if (window.DeviceMotionEvent) {
        const noteEl = liveSensor.elements.supportNote;
        const textEl = liveSensor.elements.supportText;
        noteEl.className = 'sensor-support-note supported';
        textEl.textContent = '✓ DeviceMotion sensor API supported on this device/browser.';
    } else {
        const noteEl = liveSensor.elements.supportNote;
        const textEl = liveSensor.elements.supportText;
        noteEl.className = 'sensor-support-note not-supported';
        textEl.textContent = '✗ DeviceMotion API is not available. Use a mobile browser (Chrome/Safari on phone).';
        liveSensor.elements.requestBtn.disabled = true;
    }

    // Request permission button
    liveSensor.elements.requestBtn.addEventListener('click', requestSensorAccess);
    liveSensor.elements.toggleBtn.addEventListener('click', toggleSensorPause);
}

async function requestSensorAccess() {
    const btn = liveSensor.elements.requestBtn;
    btn.disabled = true;
    btn.textContent = 'Requesting...';
    
    try {
        // iOS 13+ requires explicit permission request
        if (typeof DeviceMotionEvent !== 'undefined' && typeof DeviceMotionEvent.requestPermission === 'function') {
            const motionPermission = await DeviceMotionEvent.requestPermission();
            if (motionPermission !== 'granted') {
                showLiveError('Sensor permission denied. Please allow motion sensor access in browser settings.');
                btn.disabled = false;
                btn.textContent = 'REQUEST SENSOR ACCESS';
                return;
            }
        }
        
        // Attach event listeners
        window.addEventListener('devicemotion', handleDeviceMotion, true);
        window.addEventListener('deviceorientation', handleDeviceOrientation, true);
        
        // Show active state
        liveSensor.elements.permState.classList.add('hidden');
        liveSensor.elements.activeState.classList.remove('hidden');
        liveSensor.active = true;
        
        // Start prediction loop
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
    
    if (!acc) return;
    
    const x = acc.x || 0;
    const y = acc.y || 0;
    const z = acc.z || 0;
    
    // Add to buffer
    liveSensor.accBuffer.push({ x, y, z });
    if (liveSensor.accBuffer.length > liveSensor.WINDOW_SIZE) {
        liveSensor.accBuffer.shift();
    }
    
    // Gyroscope
    if (rot) {
        const gx = (rot.alpha || 0) * Math.PI / 180;
        const gy = (rot.beta  || 0) * Math.PI / 180;
        const gz = (rot.gamma || 0) * Math.PI / 180;
        liveSensor.gyrBuffer.push({ x: gx, y: gy, z: gz });
        if (liveSensor.gyrBuffer.length > liveSensor.WINDOW_SIZE) {
            liveSensor.gyrBuffer.shift();
        }
    }
    
    // Update UI axis displays
    updateAxisDisplay(x, y, z, 'acc');
    
    // Update window bar progress
    const pct = Math.min(100, (liveSensor.accBuffer.length / liveSensor.WINDOW_SIZE) * 100);
    liveSensor.elements.windowBar.style.width = pct + '%';
    liveSensor.elements.sampleCount.textContent = `${liveSensor.accBuffer.length} / ${liveSensor.WINDOW_SIZE} samples`;
}

function handleDeviceOrientation(event) {
    if (liveSensor.paused) return;
    
    // Use gamma/beta as proxies for gravity direction
    const beta  = (event.beta  || 0) * Math.PI / 180;  // front-back tilt
    const gamma = (event.gamma || 0) * Math.PI / 180;  // left-right tilt
    
    // Approximate gravity components (phone acceleration due to gravity)
    // These are analogous to tGravityAcc-mean()-X and Y in the UCI feature set
    const gx = Math.sin(gamma);
    const gy = -Math.sin(beta) * Math.cos(gamma);
    const gz = -Math.cos(beta) * Math.cos(gamma);
    
    liveSensor.gravBuffer.push({ x: gx, y: gy, z: gz });
    if (liveSensor.gravBuffer.length > liveSensor.WINDOW_SIZE) {
        liveSensor.gravBuffer.shift();
    }
    
    // Update gyro display if we have gyro data from DeviceMotion
    if (liveSensor.gyrBuffer.length > 0) {
        const last = liveSensor.gyrBuffer[liveSensor.gyrBuffer.length - 1];
        updateAxisDisplay(last.x, last.y, last.z, 'gyr');
    }
}

function updateAxisDisplay(x, y, z, type) {
    // Map value range to bar width. 
    // Accelerometer includes gravity ~9.8 m/s^2. We map [-15, 15] → [0%, 100%]
    // Gyroscope in rad/s, map [-5, 5] → [0%, 100%]
    const maxVal = type === 'acc' ? 15 : 5;
    
    const toBar = (v) => {
        const pct = ((v + maxVal) / (2 * maxVal)) * 100;
        return Math.max(0, Math.min(100, pct));
    };
    
    const el = liveSensor.elements;
    if (type === 'acc') {
        el.accX.textContent = x.toFixed(2);
        el.accY.textContent = y.toFixed(2);
        el.accZ.textContent = z.toFixed(2);
        el.accBarX.style.width = toBar(x) + '%';
        el.accBarY.style.width = toBar(y) + '%';
        el.accBarZ.style.width = toBar(z) + '%';
    } else {
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
    
    if (liveSensor.paused) {
        btn.textContent = '▶ Resume Sensor';
        btn.classList.remove('btn-secondary');
        btn.classList.add('btn-primary');
        clearInterval(liveSensor.predictionInterval);
    } else {
        btn.textContent = '⏸ Pause Sensor';
        btn.classList.remove('btn-primary');
        btn.classList.add('btn-secondary');
        liveSensor.predictionInterval = setInterval(submitLivePrediction, liveSensor.predictionIntervalMs);
    }
}

// Compute aggregate statistics from the sliding window buffers
function computeWindowStats() {
    const accBuf = liveSensor.accBuffer;
    const gyrBuf = liveSensor.gyrBuffer;
    const gravBuf = liveSensor.gravBuffer;
    
    if (accBuf.length < 10) return null;  // need at least 10 samples
    
    // Compute means
    const accMeanX = accBuf.reduce((s, v) => s + v.x, 0) / accBuf.length;
    const accMeanY = accBuf.reduce((s, v) => s + v.y, 0) / accBuf.length;
    const accMeanZ = accBuf.reduce((s, v) => s + v.z, 0) / accBuf.length;
    
    // Compute standard deviations
    const accStdX = Math.sqrt(accBuf.reduce((s, v) => s + (v.x - accMeanX) ** 2, 0) / accBuf.length);
    const accStdY = Math.sqrt(accBuf.reduce((s, v) => s + (v.y - accMeanY) ** 2, 0) / accBuf.length);
    const accStdZ = Math.sqrt(accBuf.reduce((s, v) => s + (v.z - accMeanZ) ** 2, 0) / accBuf.length);
    
    // Signal Magnitude Area (SMA): mean of |acc| across window
    // UCI tBodyAcc-sma() is the integral of absolute values over the 3 axes, normalised to [-1, 1].
    // We compute an equivalent by averaging the per-sample magnitudes and then mapping to [-1, 1] range.
    // A completely still phone has ~9.8 m/s^2 on one axis. Walking typically adds ±3 m/s^2 variations.
    // We map the acceleration SMA to UCI's normalised range by subtracting the gravity component estimate.
    const gravMeanX = gravBuf.length > 0 ? gravBuf.reduce((s, v) => s + v.x, 0) / gravBuf.length : 0;
    const gravMeanY = gravBuf.length > 0 ? gravBuf.reduce((s, v) => s + v.y, 0) / gravBuf.length : 0;
    const gravMeanZ = gravBuf.length > 0 ? gravBuf.reduce((s, v) => s + v.z, 0) / gravBuf.length : -1;
    
    // Body acceleration (remove approximate gravity)
    // g ≈ 9.81 m/s^2 - we use the gravity vector components to approximate
    const G = 9.81;
    const bodyBuf = accBuf.map(v => ({
        x: v.x - gravMeanX * G,
        y: v.y - gravMeanY * G,
        z: v.z + gravMeanZ * G  // gravity sign convention
    }));
    
    // SMA of body acceleration (normalised approximation)
    const bodySmaRaw = bodyBuf.reduce((s, v) => s + (Math.abs(v.x) + Math.abs(v.y) + Math.abs(v.z)) / 3, 0) / bodyBuf.length;
    
    // Map raw SMA (0 = still, ~5+ = very active) to UCI [-1, 1] range
    // UCI acc_sma: still ≈ -0.95, walking ≈ -0.15, vigorous ≈ +0.15
    // Linear mapping: raw 0 → -1.0, raw 5.0 → +0.5
    const accSma = Math.max(-1, Math.min(0.5, (bodySmaRaw / 5.0) * 1.5 - 1.0));
    
    // Gyroscope SMA
    let gyrSma = -0.95;  // default: still
    if (gyrBuf.length > 0) {
        const gyrSmaRaw = gyrBuf.reduce((s, v) => s + (Math.abs(v.x) + Math.abs(v.y) + Math.abs(v.z)) / 3, 0) / gyrBuf.length;
        // Map raw rad/s SMA (0=still, 1=walking) to UCI range. UCI gyro_sma: still≈-0.95, walking≈-0.25, vigorous≈-0.1
        gyrSma = Math.max(-1, Math.min(0.0, (gyrSmaRaw * 1.5) - 1.0));
    }
    
    // Compute standard deviations normalised to UCI range.
    // Walking acc std ≈ 1–3 m/s^2. UCI acc-std: still≈-0.95, walking≈-0.2 to +0.2
    const normStd = (s, maxRaw) => Math.max(-1, Math.min(1, (s / maxRaw) * 2 - 1));
    
    return {
        acc_sma:   accSma,
        gyr_sma:   gyrSma,
        gravity_x: Math.max(-1, Math.min(1, gravMeanX)),
        gravity_y: Math.max(-1, Math.min(1, gravMeanY)),
        acc_std_x: normStd(accStdX, 4.0),
        acc_std_y: normStd(accStdY, 4.0),
        acc_std_z: normStd(accStdZ, 4.0),
        // Additional stats (for future use)
        acc_mean_x: accMeanX,
        acc_mean_y: accMeanY,
        acc_mean_z: accMeanZ,
    };
}

// Send window statistics to backend for prediction
async function submitLivePrediction() {
    const stats = computeWindowStats();
    if (!stats) return;  // Not enough samples yet
    
    try {
        const response = await fetch('/api/live-predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(stats)
        });
        
        const data = await response.json();
        
        if (data.error === 'INSUFFICIENT_MATCHES') {
            // Keep previous result visible, just don't update
            return;
        }
        
        if (response.ok && data.activity) {
            // Show result display
            liveSensor.elements.resultPlaceholder.classList.add('hidden');
            liveSensor.elements.resultDisplay.classList.remove('hidden');
            
            // Update activity
            const emoji = activityEmojis[data.activity] || '🏃';
            liveSensor.elements.activityName.textContent = `${emoji} ${data.activity.replace(/_/g, ' ')}`;
            
            // Update confidence
            const confPct = (data.confidence * 100).toFixed(1) + '%';
            liveSensor.elements.confidenceBar.style.width = confPct;
            liveSensor.elements.confidenceText.textContent = confPct;
            
            // Update probability chart
            renderProbabilityChart(liveSensor.elements.probChart, data.top_predictions, data.activity);
            
            // Update meta
            liveSensor.elements.matchedCount.textContent = data.matched_samples;
            liveSensor.elements.modelName.textContent = data.model;
        }
        
    } catch (e) {
        // Silent fail - keep trying on next interval
        console.warn('Live prediction error:', e);
    }
}

function showLiveError(msg) {
    const el = liveSensor.elements.errorMsg;
    el.textContent = msg;
    el.classList.remove('hidden');
}

// Initialize Application on Page Load
document.addEventListener('DOMContentLoaded', () => {
    initApp();
    setupEventHandlers();
    initLiveSensor();
});

// Main Initialization Sequence
async function initApp() {
    updateLoadingStep('step-backend', 'running', 'Connecting to backend...');
    
    try {
        const response = await fetch('/api/status');
        const data = await response.json();
        
        // Step 1: Connection Check
        updateLoadingStep('step-backend', 'success', 'Connected to backend server.');
        
        // Step 2: Dataset Check
        if (data.dataset_verified) {
            updateLoadingStep('step-dataset', 'success', 'UCI HAR Dataset found and verified.');
        } else {
            updateLoadingStep('step-dataset', 'error', 'UCI HAR Dataset folder is missing.');
            showLoadingError('UCI HAR Dataset directory not found. Please verify "dataset/UCI HAR Dataset/" exists.');
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
        showLoadingError('Could not reach Flask server. Is backend/app.py running on port 5000?');
    }
}

// Load secondary data from API
async function loadAppData() {
    try {
        // Fetch activities descriptions
        const actRes = await fetch('/api/activities');
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
        const modelRes = await fetch('/api/model');
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
        const resultsRes = await fetch('/api/results');
        if (resultsRes.ok) {
            const results = await resultsRes.json();
            
            // Set image sources with cache buster to force refresh
            const cb = '?t=' + new Date().getTime();
            document.getElementById('img-confusion-matrix').src = results.plots.confusion_matrix + cb;
            document.getElementById('img-model-comparison').src = results.plots.model_comparison + cb;
            document.getElementById('img-activity-distribution').src = results.plots.activity_distribution + cb;
        }
    } catch (e) {
        console.error("Error refreshing dashboard:", e);
    }
}

// Event Handlers Setup
function setupEventHandlers() {
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
        const response = await fetch('/api/predict', {
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
        const response = await fetch(`/api/sample/${sampleId}`);
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
        const response = await fetch('/api/train', { method: 'POST' });
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
