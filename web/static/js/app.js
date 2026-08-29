/**
 * Action Recognition Visualization System - frontend interaction logic
 */

// Global state
let currentState = {
    initialized: false,
    currentIndex: 0,
    total: 0,
    correct: 0,
    incorrect: 0,
    currentData: null,
    currentResults: null,
    originalImages: [],
    hosImages: []
};

/**
 * Switch between image tabs
 */
function showImageTab(tabName) {
    const originalGrid = document.getElementById('imageGridOriginal');
    const hosGrid = document.getElementById('imageGridHos');
    const buttons = document.querySelectorAll('.tab-btn');
    
    if (tabName === 'original') {
        originalGrid.style.display = 'grid';
        hosGrid.style.display = 'none';
        buttons[0].classList.add('active');
        buttons[1].classList.remove('active');
    } else {
        originalGrid.style.display = 'none';
        hosGrid.style.display = 'grid';
        buttons[0].classList.remove('active');
        buttons[1].classList.add('active');
    }
}

// Base API URL
const API_BASE = '';

/**
 * Show the loading overlay
 */
function showLoading(text = 'Processing...') {
    document.getElementById('loadingText').textContent = text;
    document.getElementById('loadingOverlay').style.display = 'flex';
}

/**
 * Hide the loading overlay
 */
function hideLoading() {
    document.getElementById('loadingOverlay').style.display = 'none';
}

/**
 * Update the status bar
 */
function updateStatus(message) {
    document.getElementById('statusBar').textContent = message;
}

/**
 * Initialize the system
 */
async function initSystem() {
    try {
        showLoading('Initializing the system...');
        updateStatus('Initializing...');
        
        const response = await fetch(`${API_BASE}/api/init`, { method: 'POST' });
        const data = await response.json();
        
        if (data.success) {
            currentState.initialized = true;
            currentState.total = data.total_count;
            currentState.currentIndex = 0;
            
            document.getElementById('initBtn').disabled = true;
            document.getElementById('nextBtn').disabled = false;
            document.getElementById('prevBtn').disabled = false;
            document.getElementById('runBtn').disabled = false;
            
            updateProgress();
            await loadCurrentData();
            
            updateStatus(`Initialized, ${data.total_count} samples in total`);
            alert(`System initialized successfully!\nLoaded ${data.total_count} samples`);
        } else {
            throw new Error(data.error);
        }
    } catch (error) {
        console.error('Initialization failed:', error);
        updateStatus('Initialization failed');
        alert('Initialization failed: ' + error.message);
    } finally {
        hideLoading();
    }
}

/**
 * Load the current sample
 */
async function loadCurrentData() {
    try {
        showLoading('Loading data...');
        
        const response = await fetch(`${API_BASE}/api/data/current`);
        const data = await response.json();
        
        if (data.success) {
            currentState.currentData = data.data;
            displayVideoInfo(data.data);
            displayGroundTruth(data.data);
            displayImages(data.images);
            updateProgress(data.progress);
            updateStats(data.stats);
            
            // Clear the agent results
            clearAgentResults();
            
            // Enable the buttons
            document.getElementById('runBtn').disabled = false;
            
            updateStatus(`Loaded: ${data.data.narration_id}`);
        } else if (data.completed) {
            handleCompletion();
            return;
        } else {
            throw new Error(data.error);
        }
    } catch (error) {
        console.error('Failed to load data:', error);
        updateStatus('Failed to load');
        alert('Failed to load data: ' + error.message);
    } finally {
        hideLoading();
    }
}

/**
 * Display the video information
 */
function displayVideoInfo(data) {
    const infoHtml = `
        <p><span class="info-label">Narration ID:</span> <span class="info-value">${data.narration_id}</span></p>
        <p><span class="info-label">Participant:</span> <span class="info-value">${data.participant_id}</span></p>
        <p><span class="info-label">Video:</span> <span class="info-value">${data.video_id}</span></p>
        <p><span class="info-label">Narration:</span> <span class="info-value">${data.narration}</span></p>
        <p><span class="info-label">Time:</span> <span class="info-value">${data.start_timestamp} - ${data.stop_timestamp}</span></p>
        <p><span class="info-label">Frames:</span> <span class="info-value">${data.start_frame} - ${data.stop_frame}</span></p>
    `;
    document.getElementById('videoInfo').innerHTML = infoHtml;
}

/**
 * Display the ground truth
 */
function displayGroundTruth(data) {
    const gtHtml = `
        <p class="gt-text">🎯 ${data.ground_truth}</p>
        <p><span class="info-label">Verb Class:</span> <span class="info-value">${data.verb_class}</span></p>
        <p><span class="info-label">Noun Class:</span> <span class="info-value">${data.noun_class}</span></p>
    `;
    document.getElementById('groundTruth').innerHTML = gtHtml;
}

/**
 * Display the images (raw frames and HOS frames)
 */
function displayImages(images) {
    const originalGrid = document.getElementById('imageGridOriginal');
    const hosGrid = document.getElementById('imageGridHos');
    originalGrid.innerHTML = '';
    hosGrid.innerHTML = '';
    
    // The first 16 images are raw frames, the last 16 are HOS frames
    const originalImages = images.slice(0, 16);
    const hosImages = images.slice(16, 32);
    
    currentState.originalImages = originalImages;
    currentState.hosImages = hosImages;
    
    // Display the raw frames
    originalImages.forEach(img => {
        const frame = document.createElement('div');
        frame.className = 'image-frame';
        
        if (img.data && img.exists) {
            frame.innerHTML = `
                <img src="${img.data}" alt="Frame ${img.index}">
            `;
        } else {
            frame.innerHTML = `
                <div class="placeholder">
                    Frame ${img.index}<br>
                    ${img.exists ? 'Failed to load' : 'Not found'}
                </div>
            `;
        }
        
        originalGrid.appendChild(frame);
    });
    
    // Display the HOS frames
    hosImages.forEach(img => {
        const frame = document.createElement('div');
        frame.className = 'image-frame';
        
        if (img.data && img.exists) {
            frame.innerHTML = `
                <img src="${img.data}" alt="HOS Frame ${img.index}">
            `;
        } else {
            frame.innerHTML = `
                <div class="placeholder">
                    HOS Frame ${img.index}<br>
                    ${img.exists ? 'Failed to load' : 'Not found'}
                </div>
            `;
        }
        
        hosGrid.appendChild(frame);
    });
}

/**
 * Clear the agent results and the semantic context
 */
function clearAgentResults() {
    // Clear the results
    ['agent1', 'agent2', 'agent3'].forEach(agent => {
        const element = document.getElementById(`${agent}Result`);
        if (element) {
            element.innerHTML = '<p class="placeholder">Click "Run Recognition" to see the results</p>';
        }
    });

    // Clear the semantic context
    const semanticElement = document.getElementById('semanticContext');
    if (semanticElement) {
        semanticElement.innerHTML = '<p class="placeholder">Click "Run Recognition" to see the semantic context</p>';
    }
}

/**
 * Run recognition
 */
async function runRecognition() {
    try {
        showLoading('Running recognition...\nThis may take several minutes');
        updateStatus('Running recognition...');
        
        // Disable the buttons
        document.getElementById('runBtn').disabled = true;
        
        const response = await fetch(`${API_BASE}/api/recognize`, { method: 'POST' });
        const data = await response.json();
        
        if (data.success) {
            currentState.currentResults = data.results;
            displayAgentResults(data.results);
            
            // Enable the evaluation button
            document.getElementById('correctBtn').disabled = false;
            document.getElementById('incorrectBtn').disabled = false;
            
            updateStatus('Recognition finished, please evaluate the result');
        } else {
            throw new Error(data.error);
        }
    } catch (error) {
        console.error('Recognition failed:', error);
        updateStatus('Recognition failed');
        alert('Recognition failed: ' + error.message);
    } finally {
        hideLoading();
        document.getElementById('runBtn').disabled = false;
    }
}

/**
 * Display the agent results
 */
function displayAgentResults(results) {
    console.log('=== Recognition result received ===');
    console.log('Full results object:', JSON.stringify(results, null, 2));
    console.log('agent1:', results?.agent1);
    console.log('agent2:', results?.agent2);
    console.log('agent3:', results?.agent3);

    // Debug: check whether the DOM elements exist
    const _agent1Container = document.getElementById('agent1Result');
    const _agent2Container = document.getElementById('agent2Result');
    const _agent3Container = document.getElementById('agent3Result');
    console.log('DOM element check:', {
        agent1Container: !!_agent1Container,
        agent2Container: !!_agent2Container,
        agent3Container: !!_agent3Container
    });

    // Display the semantic context
    displaySemanticContext(results);

    // Agent 1: noun selection
    const agent1Container = document.getElementById('agent1Result');
    console.log('Before setting the Agent 1 result:', agent1Container.innerHTML);
    if (results.agent1 && results.agent1.success) {
        let html = `<p class="nouns-text" style="font-size: 1.2em; font-weight: bold;">
            📦 Selected nouns: ${JSON.stringify(results.agent1.nouns)}
        </p>`;
        if (results.agent1.raw_result) {
            html += `<p class="nouns-text">Raw result: ${JSON.stringify(results.agent1.raw_result)}</p>`;
        }
        console.log('Agent 1 HTML:', html);
        agent1Container.innerHTML = html;
        console.log('After setting the Agent 1 result:', agent1Container.innerHTML);
    } else if (results.agent1) {
        agent1Container.innerHTML = `<p class="error-message">❌ ${results.agent1.error || 'Recognition failed'}</p>`;
    } else {
        agent1Container.innerHTML = '<p class="placeholder">No data</p>';
    }

    // Agent 2: action selection
    const agent2Container = document.getElementById('agent2Result');
    if (results.agent2 && results.agent2.success) {
        let html = '<p class="nouns-text">';
        html += `📦 Input nouns: ${JSON.stringify(results.agent2.nouns)}<br>`;

        const actions = results.agent2.actions || [];
        if (actions.length > 0) {
            html += `</p><ul class="action-list">`;
            actions.forEach((action, idx) => {
                const actionText = typeof action === 'string' ? action :
                                   (action.action || `${action.verb || ''} ${action.noun || ''}`);
                html += `<li class="action-item">
                    <div class="action-text">${idx + 1}. ${actionText}</div>
                </li>`;
            });
            html += '</ul>';
        } else {
            html += 'No action result</p>';
        }
        agent2Container.innerHTML = html;
    } else if (results.agent2) {
        agent2Container.innerHTML = `<p class="error-message">❌ ${results.agent2.error || 'Recognition failed'}</p>`;
    } else {
        agent2Container.innerHTML = '<p class="placeholder">No data</p>';
    }

    // Agent 3: action scoring
    const agent3Container = document.getElementById('agent3Result');
    if (results.agent3 && results.agent3.success) {
        const actions = results.agent3.actions || [];

        if (actions.length === 0) {
            agent3Container.innerHTML = '<p class="placeholder">No scoring result</p>';
        } else {
            let html = '<ul class="action-list">';
            actions.forEach((action, idx) => {
                const confidence = parseFloat(action.confidence) || 0;
                const confidencePercent = (confidence * 100).toFixed(0);
                const actionText = action.action || `${action.verb || ''} ${action.noun || ''}`;

                html += `
                    <li class="action-item">
                        <div class="action-text">${idx + 1}. ${actionText}</div>
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: ${confidencePercent}%"></div>
                        </div>
                        <div class="confidence-text">Confidence: ${(confidence * 100).toFixed(1)}%</div>
                    </li>
                `;
            });
            html += '</ul>';
            agent3Container.innerHTML = html;
        }
    } else if (results.agent3) {
        agent3Container.innerHTML = `<p class="error-message">❌ ${results.agent3.error || 'Recognition failed'}</p>`;
    } else {
        agent3Container.innerHTML = '<p class="placeholder">No data</p>';
    }

    // Automatically scroll to the result section
    setTimeout(() => {
        const actionButtonsContainer = document.querySelector('.action-buttons-container');
        if (actionButtonsContainer) {
            actionButtonsContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    }, 100);
}

/**
 * Display the semantic context (pre_nouns)
 */
function displaySemanticContext(results) {
    console.log('=== displaySemanticContext called ===');
    console.log('results object:', results);
    console.log('results.semantic_context:', results?.semantic_context);
    
    const semanticContextContainer = document.getElementById('semanticContext');

    // Return directly if the element does not exist
    if (!semanticContextContainer) {
        console.warn('The semanticContext element does not exist');
        return;
    }

    // Check whether semantic_context data is available
    if (results.semantic_context && results.semantic_context.success) {
        const preNouns = results.semantic_context.pre_nouns || {};
        console.log('pre_nouns data:', preNouns);

        if (Object.keys(preNouns).length === 0) {
            semanticContextContainer.innerHTML = '<p class="placeholder">No semantic context data</p>';
            return;
        }

        // Build the HTML for each noun and its semantic description
        let html = '<div class="semantic-context-list">';

        for (const [noun, description] of Object.entries(preNouns)) {
            html += `
                <div class="semantic-context-item">
                    <div class="semantic-noun">📦 ${escapeHtml(noun)}</div>
                    <div class="semantic-description">${escapeHtml(description || 'No description')}</div>
                </div>
            `;
        }

        html += '</div>';
        console.log('Generated HTML:', html);
        semanticContextContainer.innerHTML = html;
    } else if (results.semantic_context && results.semantic_context.success === false) {
        semanticContextContainer.innerHTML = `<p class="error-message">❌ ${results.semantic_context.error || 'Failed to load'}</p>`;
    } else {
        console.log('No semantic_context data, show the empty state');
        semanticContextContainer.innerHTML = '<p class="placeholder">No data</p>';
    }
}

/**
 * Display the agent prompts (deprecated, kept for compatibility)
 */
function displayAgentPrompts(results) {
    // Get the prompts from the context
    const context = results.context || {};

    // Agent 1 prompt
    const agent1PromptContainer = document.getElementById('agent1Prompt');
    if (context.agent1 && context.agent1.strategies_text) {
        agent1PromptContainer.innerHTML = formatPromptText(context.agent1.strategies_text);
    } else {
        agent1PromptContainer.innerHTML = '<p class="prompt-text" style="color: #999;">No strategy prompt</p>';
    }

    // Agent 2 prompt
    const agent2PromptContainer = document.getElementById('agent2Prompt');
    if (context.agent2 && context.agent2.strategies_text) {
        agent2PromptContainer.innerHTML = formatPromptText(context.agent2.strategies_text);
    } else {
        agent2PromptContainer.innerHTML = '<p class="prompt-text" style="color: #999;">No strategy prompt</p>';
    }

    // Agent 3 prompt
    const agent3PromptContainer = document.getElementById('agent3Prompt');
    if (context.agent3 && context.agent3.strategies_text) {
        agent3PromptContainer.innerHTML = formatPromptText(context.agent3.strategies_text);
    } else {
        agent3PromptContainer.innerHTML = '<p class="prompt-text" style="color: #999;">No strategy prompt</p>';
    }
}

/**
 * Format the prompt text
 */
function formatPromptText(text) {
    if (!text || text === 'No strategy') {
        return '<p class="prompt-text" style="color: #999;">No strategy prompt</p>';
    }
    
    // Split the text by lines and format it
    const lines = text.split('\n');
    let html = '<div class="prompt-strategy-list">';
    
    lines.forEach(line => {
        line = line.trim();
        if (line) {
            // Check whether there is a category prefix (e.g. "- Category: content")
            if (line.startsWith('- ') && line.includes(':')) {
                const parts = line.substring(2).split(':');
                const category = parts[0].trim();
                const content = parts.slice(1).join(':').trim();
                html += `
                    <div class="prompt-strategy-item">
                        <div class="prompt-category">${escapeHtml(category)}</div>
                        <div class="prompt-content-text">${escapeHtml(content)}</div>
                    </div>
                `;
            } else {
                html += `<div class="prompt-strategy-item">${escapeHtml(line)}</div>`;
            }
        }
    });
    
    html += '</div>';
    return html;
}

/**
 * HTML escape helper
 */
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

/**
 * Mark the result
 */
async function markResult(isCorrect) {
    if (!currentState.currentResults) {
        alert('Please run recognition first');
        return;
    }
    
    try {
        // Use the result of Agent 3 (action scoring)
        const agentResult = currentState.currentResults.agent3;
        
        // Get the first predicted action
        let pred_verb = 'no result';
        let pred_noun = 'no result';
        
        if (agentResult && agentResult.success && agentResult.actions && agentResult.actions.length > 0) {
            const firstAction = agentResult.actions[0];
            pred_verb = firstAction.verb || 'N/A';
            pred_noun = firstAction.noun || 'N/A';
        }
        
        const response = await fetch(`${API_BASE}/api/evaluate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                is_correct: isCorrect,
                agent_result: {
                    actions: agentResult && agentResult.success ? agentResult.actions : [],
                    verbs: pred_verb,
                    nouns: pred_noun
                }
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            // Update the statistics
            currentState.correct = data.stats.correct;
            currentState.incorrect = data.stats.incorrect;
            updateStats(data.stats);
            
            // Add a log entry
            addLogEntry(data.log_entry);
            
            updateStatus(isCorrect ? 'Marked as correct ✓' : 'Marked as incorrect ✗');
            
            // Automatically jump to the next sample
            setTimeout(() => nextItem(), 500);
        } else {
            throw new Error(data.error);
        }
    } catch (error) {
        console.error('Evaluation failed:', error);
        alert('Evaluation failed: ' + error.message);
    }
}

/**
 * Add a log entry
 */
function addLogEntry(entry) {
    const logContent = document.getElementById('logContent');
    const logEntry = document.createElement('div');
    logEntry.className = `log-entry ${entry.is_correct ? 'correct' : 'incorrect'}`;
    
    const symbol = entry.is_correct ? '✓' : '✗';
    logEntry.innerHTML = `
        [${symbol}] ${entry.narration_id} | 
        GT: ${entry.gt_verb}+${entry.gt_noun} | 
        Pred: ${entry.pred_verb}+${entry.pred_noun}
    `;
    
    logContent.insertBefore(logEntry, logContent.firstChild);
    
    // Limit the number of log entries
    while (logContent.children.length > 100) {
        logContent.removeChild(logContent.lastChild);
    }
}

/**
 * Move to the next sample
 */
async function nextItem() {
    try {
        showLoading('Loading...');
        
        const response = await fetch(`${API_BASE}/api/next`, { method: 'POST' });
        const data = await response.json();
        
        if (data.success) {
            currentState.currentIndex = data.progress.current - 1;
            updateProgress(data.progress);
            await loadCurrentData();
        } else if (data.completed) {
            handleCompletion(data.final_stats);
            return;
        } else {
            throw new Error(data.error);
        }
    } catch (error) {
        console.error('Failed to jump:', error);
        alert('Failed to jump: ' + error.message);
    } finally {
        hideLoading();
    }
}

/**
 * Move to the previous sample
 */
async function prevItem() {
    if (currentState.currentIndex <= 0) {
        return;
    }
    
    currentState.currentIndex--;
    
    try {
        // Requires backend support to return a given index; simplified here
        showLoading('Loading...');
        
        // Temporary solution: reload the current sample
        await loadCurrentData();
        
        updateStatus(`Jumped to sample ${currentState.currentIndex + 1}`);
    } catch (error) {
        console.error('Failed to jump:', error);
    } finally {
        hideLoading();
    }
}

/**
 * Save the data
 */
async function saveData() {
    try {
        showLoading('Saving data...');
        
        const response = await fetch(`${API_BASE}/api/save`, { method: 'POST' });
        const data = await response.json();
        
        if (data.success) {
            updateStatus('Data saved');
            alert(`Data saved successfully!\n\nStatistics:\n` +
                  `Total: ${data.stats.total}\n` +
                  `Correct: ${data.stats.correct}\n` +
                  `Incorrect: ${data.stats.incorrect}\n` +
                  `Accuracy: ${(data.stats.accuracy * 100).toFixed(2)}%\n\n` +
                  `The file has been saved on the server`);
        } else {
            throw new Error(data.error);
        }
    } catch (error) {
        console.error('Failed to save:', error);
        alert('Failed to save: ' + error.message);
    } finally {
        hideLoading();
    }
}

/**
 * Update the progress display
 */
function updateProgress(progress) {
    const progressText = document.getElementById('progressText');
    if (progress) {
        progressText.textContent = `Progress: ${progress.current} / ${progress.total}`;
    } else {
        progressText.textContent = `Progress: ${currentState.currentIndex + 1} / ${currentState.total}`;
    }
}

/**
 * Update the statistics display
 */
function updateStats(stats) {
    const statsText = document.getElementById('statsText');
    const accuracy = stats.accuracy ? (stats.accuracy * 100).toFixed(1) : 0;
    statsText.textContent = `Correct: ${stats.correct} | Incorrect: ${stats.incorrect} | Accuracy: ${accuracy}%`;
}

/**
 * All samples processed
 */
function handleCompletion(finalStats) {
    hideLoading();
    
    let message = '🎉 All samples have been processed!\n\n';
    if (finalStats) {
        message += `Statistics:\n` +
                   `Total: ${finalStats.total}\n` +
                   `Correct: ${finalStats.correct}\n` +
                   `Incorrect: ${finalStats.incorrect}\n` +
                   `Accuracy: ${(finalStats.accuracy * 100).toFixed(2)}%\n`;
    }
    
    message += '\nPlease click the "Save Data" button to save the results.';
    
    alert(message);
    updateStatus('All samples have been processed');
    
    // Disable the buttons
    document.getElementById('runBtn').disabled = true;
    document.getElementById('correctBtn').disabled = true;
    document.getElementById('incorrectBtn').disabled = true;
    document.getElementById('nextBtn').disabled = true;
}

/**
 * Keyboard shortcuts
 */
document.addEventListener('keydown', (e) => {
    // Ignore the shortcuts while typing
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
        return;
    }
    
    switch(e.key) {
        case 'Enter':
            if (!e.ctrlKey && !e.shiftKey) {
                e.preventDefault();
                if (!document.getElementById('runBtn').disabled) {
                    runRecognition();
                }
            }
            break;
        case 'c':
        case 'C':
            if (e.ctrlKey) {
                e.preventDefault();
                if (!document.getElementById('correctBtn').disabled) {
                    markResult(true);
                }
            }
            break;
        case 'x':
        case 'X':
            if (e.ctrlKey) {
                e.preventDefault();
                if (!document.getElementById('incorrectBtn').disabled) {
                    markResult(false);
                }
            }
            break;
        case 'ArrowRight':
            e.preventDefault();
            if (!document.getElementById('nextBtn').disabled) {
                nextItem();
            }
            break;
        case 'ArrowLeft':
            e.preventDefault();
            if (!document.getElementById('prevBtn').disabled) {
                prevItem();
            }
            break;
        case 's':
        case 'S':
            if (e.ctrlKey) {
                e.preventDefault();
                saveData();
            }
            break;
    }
});

// Prompt for initialization when the page is loaded
window.addEventListener('load', () => {
    updateStatus('Please click the "Initialize System" button to start');
});
