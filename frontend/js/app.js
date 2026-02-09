// Image Forgery Detection - Frontend JavaScript
const API_URL = 'http://localhost:8000/api';
let selectedFile = null;
let authToken = localStorage.getItem('token');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    setupDropZone();
    if (authToken) loadDashboard();
});

// Drop Zone Setup
function setupDropZone() {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');

    dropZone.addEventListener('click', () => fileInput.click());
    
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });
    
    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });
    
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        const file = e.dataTransfer.files[0];
        if (file && file.type.startsWith('image/')) handleFile(file);
    });
    
    fileInput.addEventListener('change', (e) => {
        if (e.target.files[0]) handleFile(e.target.files[0]);
    });
}

function handleFile(file) {
    selectedFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
        document.getElementById('preview-image').src = e.target.result;
        document.getElementById('preview-container').classList.remove('hidden');
        document.getElementById('results').classList.add('hidden');
    };
    reader.readAsDataURL(file);
}

function clearPreview() {
    selectedFile = null;
    document.getElementById('preview-container').classList.add('hidden');
    document.getElementById('results').classList.add('hidden');
    document.getElementById('file-input').value = '';
}

// Analyze Image
async function analyzeImage() {
    if (!selectedFile) return;
    
    document.getElementById('loading').classList.remove('hidden');
    document.getElementById('results').classList.add('hidden');
    
    const formData = new FormData();
    formData.append('file', selectedFile);
    
    try {
        const headers = {};
        if (authToken) headers['Authorization'] = `Bearer ${authToken}`;
        
        const response = await fetch(`${API_URL}/analyze`, {
            method: 'POST',
            headers,
            body: formData
        });
        
        const data = await response.json();
        displayResults(data);
    } catch (error) {
        alert('Error analyzing image: ' + error.message);
    } finally {
        document.getElementById('loading').classList.add('hidden');
    }
}

function displayResults(data) {
    document.getElementById('results').classList.remove('hidden');
    
    // Verdict badge
    const verdictBadge = document.getElementById('verdict-badge');
    if (data.verdict === 'FORGED') {
        verdictBadge.innerHTML = `
            <span class="inline-block bg-red-600 text-white px-6 py-3 rounded-full text-xl font-bold">
                <i class="fas fa-exclamation-triangle mr-2"></i>FORGED
            </span>
        `;
    } else {
        verdictBadge.innerHTML = `
            <span class="inline-block bg-green-600 text-white px-6 py-3 rounded-full text-xl font-bold">
                <i class="fas fa-check-circle mr-2"></i>AUTHENTIC
            </span>
        `;
    }
    
    // Confidence and score
    document.getElementById('confidence').textContent = data.confidence.toUpperCase();
    document.getElementById('score').textContent = (data.score * 100).toFixed(1) + '%';
    
    // Model scores
    const scoresDiv = document.getElementById('model-scores');
    scoresDiv.innerHTML = '';
    
    if (data.details) {
        for (const [model, score] of Object.entries(data.details)) {
            const percentage = (score * 100).toFixed(1);
            scoresDiv.innerHTML += `
                <div class="flex items-center">
                    <span class="w-32 text-gray-400">${model.replace('_', ' ')}</span>
                    <div class="flex-1 bg-gray-700 rounded-full h-3 mx-2">
                        <div class="bg-indigo-500 h-3 rounded-full" style="width: ${percentage}%"></div>
                    </div>
                    <span class="w-16 text-right">${percentage}%</span>
                </div>
            `;
        }
    }
    
    // Heatmap
    if (data.id) {
        document.getElementById('heatmap-container').classList.remove('hidden');
        document.getElementById('heatmap-image').src = `${API_URL}/analyze/${data.id}/heatmap`;
    }
}

// Section Navigation
function showSection(section) {
    document.getElementById('upload-section').classList.add('hidden');
    document.getElementById('history-section').classList.add('hidden');
    document.getElementById('dashboard-section').classList.add('hidden');
    document.getElementById(`${section}-section`).classList.remove('hidden');
    
    if (section === 'history') loadHistory();
    if (section === 'dashboard') loadDashboard();
}

// Load History
async function loadHistory() {
    if (!authToken) {
        document.getElementById('history-list').innerHTML = '<p class="text-gray-400">Please login to view history</p>';
        return;
    }
    
    try {
        const response = await fetch(`${API_URL}/history`, {
            headers: { 'Authorization': `Bearer ${authToken}` }
        });
        const data = await response.json();
        
        const historyList = document.getElementById('history-list');
        historyList.innerHTML = '';
        
        data.forEach(item => {
            const verdictClass = item.verdict === 'FORGED' ? 'text-red-400' : 'text-green-400';
            historyList.innerHTML += `
                <div class="bg-gray-800 rounded-lg p-4 flex justify-between items-center">
                    <div>
                        <p class="font-semibold">${item.filename}</p>
                        <p class="text-sm text-gray-400">${new Date(item.created_at).toLocaleString()}</p>
                    </div>
                    <div class="text-right">
                        <p class="${verdictClass} font-bold">${item.verdict}</p>
                        <p class="text-sm text-gray-400">${(item.score * 100).toFixed(1)}%</p>
                    </div>
                </div>
            `;
        });
    } catch (error) {
        console.error('Error loading history:', error);
    }
}

// Load Dashboard
async function loadDashboard() {
    if (!authToken) return;
    
    try {
        const response = await fetch(`${API_URL}/history/stats/summary`, {
            headers: { 'Authorization': `Bearer ${authToken}` }
        });
        const data = await response.json();
        
        document.getElementById('total-analyses').textContent = data.total_analyses;
        document.getElementById('authentic-count').textContent = data.authentic_count;
        document.getElementById('forged-count').textContent = data.forged_count;
        document.getElementById('forgery-rate').textContent = (data.forgery_rate * 100).toFixed(1) + '%';
    } catch (error) {
        console.error('Error loading dashboard:', error);
    }
}
