/**
 * Beer Counter Frontend Application
 * Handles video upload, processing, and result display
 */

// Configuration
const API_BASE_URL = 'http://localhost:8000';

// State
let selectedFile = null;
let currentVideoId = null;
let pollingInterval = null;

// DOM Elements
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const selectBtn = document.getElementById('select-btn');
const fileInfo = document.getElementById('file-info');
const fileName = document.getElementById('file-name');
const fileSize = document.getElementById('file-size');
const clearFile = document.getElementById('clear-file');
const actionButtons = document.getElementById('action-buttons');
const uploadBtn = document.getElementById('upload-btn');
const processingSection = document.getElementById('processing-section');
const statusSpinner = document.getElementById('status-spinner');
const statusText = document.getElementById('status-text');
const statusDetail = document.getElementById('status-detail');
const statusContainer = document.getElementById('status-container');
const resultsSection = document.getElementById('results-section');
const tapACount = document.getElementById('tap-a-count');
const tapBCount = document.getElementById('tap-b-count');
const totalCount = document.getElementById('total-count');
const timelineContainer = document.getElementById('timeline-container');
const historyBody = document.getElementById('history-body');
const noHistory = document.getElementById('no-history');
const refreshHistory = document.getElementById('refresh-history');
const toast = document.getElementById('toast');
const toastMessage = document.getElementById('toast-message');
const toastIcon = document.getElementById('toast-icon');

// Utility Functions
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function formatDate(dateString) {
    const date = new Date(dateString);
    return date.toLocaleDateString('es-ES', {
        day: '2-digit',
        month: '2-digit',
        year: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    });
}

function formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
}

function showToast(message, type = 'info') {
    const icons = {
        success: '✅',
        error: '❌',
        info: 'ℹ️',
        warning: '⚠️'
    };
    
    toastIcon.textContent = icons[type] || icons.info;
    toastMessage.textContent = message;
    
    toast.classList.remove('translate-y-full', 'opacity-0');
    
    setTimeout(() => {
        toast.classList.add('translate-y-full', 'opacity-0');
    }, 3000);
}

function getStatusBadge(status) {
    const styles = {
        pending: 'bg-gray-100 text-gray-800',
        processing: 'bg-blue-100 text-blue-800',
        completed: 'bg-green-100 text-green-800',
        error: 'bg-red-100 text-red-800'
    };
    
    const labels = {
        pending: 'Pendiente',
        processing: 'Procesando',
        completed: 'Completado',
        error: 'Error'
    };
    
    return `<span class="px-2 py-1 rounded-full text-xs font-medium ${styles[status] || styles.pending}">${labels[status] || status}</span>`;
}

// File Handling
function handleFileSelect(file) {
    if (!file) return;
    
    const allowedTypes = ['.mp4', '.mov', '.avi', '.mkv'];
    const ext = '.' + file.name.split('.').pop().toLowerCase();
    
    if (!allowedTypes.includes(ext)) {
        showToast('Tipo de archivo no permitido. Use MP4, MOV, AVI o MKV.', 'error');
        return;
    }
    
    selectedFile = file;
    fileName.textContent = file.name;
    fileSize.textContent = formatFileSize(file.size);
    fileInfo.classList.remove('hidden');
    actionButtons.classList.remove('hidden');
    dropZone.classList.add('hidden');
}

function clearSelectedFile() {
    selectedFile = null;
    fileInput.value = '';
    fileInfo.classList.add('hidden');
    actionButtons.classList.add('hidden');
    dropZone.classList.remove('hidden');
}

// API Functions
async function uploadVideo(file) {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await fetch(`${API_BASE_URL}/api/videos/upload`, {
        method: 'POST',
        body: formData
    });
    
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Error al subir el vídeo');
    }
    
    return response.json();
}

async function processVideo(videoId) {
    const response = await fetch(`${API_BASE_URL}/api/videos/${videoId}/process`, {
        method: 'POST'
    });
    
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Error al procesar el vídeo');
    }
    
    return response.json();
}

async function getVideoStatus(videoId) {
    const response = await fetch(`${API_BASE_URL}/api/videos/${videoId}/status`);
    
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Error al obtener el estado');
    }
    
    return response.json();
}

async function getVideoDetails(videoId) {
    const response = await fetch(`${API_BASE_URL}/api/videos/${videoId}`);
    
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Error al obtener detalles');
    }
    
    return response.json();
}

async function getVideoHistory() {
    const response = await fetch(`${API_BASE_URL}/api/videos`);
    
    if (!response.ok) {
        throw new Error('Error al cargar historial');
    }
    
    return response.json();
}

async function deleteVideo(videoId) {
    const response = await fetch(`${API_BASE_URL}/api/videos/${videoId}`, {
        method: 'DELETE'
    });
    
    if (!response.ok) {
        throw new Error('Error al eliminar vídeo');
    }
    
    return response.json();
}

// UI Update Functions
function updateProcessingStatus(status, detail = '') {
    statusText.textContent = status;
    statusDetail.textContent = detail;
}

function showProcessingState(state) {
    const states = {
        processing: {
            bgClass: 'bg-blue-50',
            textClass: 'text-blue-800',
            detailClass: 'text-blue-600',
            showSpinner: true
        },
        completed: {
            bgClass: 'bg-green-50',
            textClass: 'text-green-800',
            detailClass: 'text-green-600',
            showSpinner: false
        },
        error: {
            bgClass: 'bg-red-50',
            textClass: 'text-red-800',
            detailClass: 'text-red-600',
            showSpinner: false
        }
    };
    
    const config = states[state] || states.processing;
    
    statusContainer.className = `flex items-center space-x-4 p-4 rounded-lg ${config.bgClass}`;
    statusText.className = `font-medium ${config.textClass}`;
    statusDetail.className = `text-sm ${config.detailClass}`;
    statusSpinner.style.display = config.showSpinner ? 'block' : 'none';
}

function displayResults(data) {
    tapACount.textContent = data.tap_a_count;
    tapBCount.textContent = data.tap_b_count;
    totalCount.textContent = data.total_count;
    
    // Display video timestamp if available
    if (data.video_timestamp) {
        const timestampEl = document.getElementById('videoTimestamp');
        if (timestampEl) {
            timestampEl.textContent = data.video_timestamp;
        }
    }
    
    resultsSection.classList.remove('hidden');
    
    // Animate count numbers
    animateCount(tapACount, data.tap_a_count);
    animateCount(tapBCount, data.tap_b_count);
    animateCount(totalCount, data.total_count);
}

function animateCount(element, target) {
    let current = 0;
    const duration = 500;
    const step = target / (duration / 16);
    
    const animate = () => {
        current += step;
        if (current < target) {
            element.textContent = Math.floor(current);
            requestAnimationFrame(animate);
        } else {
            element.textContent = target;
        }
    };
    
    if (target > 0) {
        animate();
    }
}

function displayTimeline(pourEvents) {
    timelineContainer.innerHTML = '';
    
    if (!pourEvents || pourEvents.length === 0) {
        timelineContainer.innerHTML = '<p class="text-gray-500 text-center py-4">No se detectaron eventos</p>';
        return;
    }
    
    // Sort events by start time
    const sortedEvents = [...pourEvents].sort((a, b) => a.start_time_seconds - b.start_time_seconds);
    
    sortedEvents.forEach((event, index) => {
        const tapColor = event.tap === 'A' ? 'blue' : 'green';
        const eventEl = document.createElement('div');
        eventEl.className = `flex items-center space-x-3 p-3 rounded-lg bg-${tapColor}-50 border-l-4 border-${tapColor}-500`;
        eventEl.innerHTML = `
            <span class="text-${tapColor}-600 font-mono font-medium">${formatTime(event.start_time_seconds)}</span>
            <span class="text-gray-400">→</span>
            <span class="text-${tapColor}-600 font-mono font-medium">${formatTime(event.end_time_seconds)}</span>
            <span class="flex-1 text-gray-700">Grifo ${event.tap}</span>
            <span class="text-gray-500 text-sm">#${index + 1}</span>
        `;
        timelineContainer.appendChild(eventEl);
    });
}

function updateHistoryTable(videos) {
    historyBody.innerHTML = '';
    
    if (!videos || videos.length === 0) {
        noHistory.classList.remove('hidden');
        return;
    }
    
    noHistory.classList.add('hidden');
    
    videos.forEach(video => {
        const row = document.createElement('tr');
        row.className = 'hover:bg-gray-50';
        row.innerHTML = `
            <td class="px-4 py-3">
                <span class="font-medium text-gray-800">${video.original_filename}</span>
            </td>
            <td class="px-4 py-3">${getStatusBadge(video.status)}</td>
            <td class="px-4 py-3 text-center font-semibold text-blue-600">${video.tap_a_count}</td>
            <td class="px-4 py-3 text-center font-semibold text-green-600">${video.tap_b_count}</td>
            <td class="px-4 py-3 text-center font-bold text-amber-600">${video.total_count}</td>
            <td class="px-4 py-3 text-gray-500">${formatDate(video.created_at)}</td>
            <td class="px-4 py-3 text-center">
                <button class="view-btn text-amber-600 hover:text-amber-700 mr-2" data-id="${video.id}" title="Ver detalles">
                    <svg class="w-5 h-5 inline" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"/>
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"/>
                    </svg>
                </button>
                <button class="delete-btn text-red-500 hover:text-red-700" data-id="${video.id}" title="Eliminar">
                    <svg class="w-5 h-5 inline" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"/>
                    </svg>
                </button>
            </td>
        `;
        historyBody.appendChild(row);
    });
    
    // Add event listeners
    document.querySelectorAll('.view-btn').forEach(btn => {
        btn.addEventListener('click', async (e) => {
            const videoId = e.currentTarget.dataset.id;
            await viewVideoDetails(videoId);
        });
    });
    
    document.querySelectorAll('.delete-btn').forEach(btn => {
        btn.addEventListener('click', async (e) => {
            const videoId = e.currentTarget.dataset.id;
            if (confirm('¿Está seguro de eliminar este vídeo?')) {
                await handleDeleteVideo(videoId);
            }
        });
    });
}

async function viewVideoDetails(videoId) {
    try {
        const video = await getVideoDetails(videoId);
        displayResults(video);
        displayTimeline(video.pour_events);
        resultsSection.scrollIntoView({ behavior: 'smooth' });
    } catch (error) {
        showToast(error.message, 'error');
    }
}

async function handleDeleteVideo(videoId) {
    try {
        await deleteVideo(videoId);
        showToast('Vídeo eliminado correctamente', 'success');
        loadHistory();
    } catch (error) {
        showToast(error.message, 'error');
    }
}

// Main Upload and Process Flow
async function handleUploadAndProcess() {
    if (!selectedFile) return;
    
    uploadBtn.disabled = true;
    uploadBtn.innerHTML = '<div class="spinner mr-2"></div><span>Subiendo...</span>';
    
    try {
        // Upload video
        const uploadResult = await uploadVideo(selectedFile);
        currentVideoId = uploadResult.id;
        
        showToast('Vídeo subido correctamente', 'success');
        
        // Show processing section
        processingSection.classList.remove('hidden');
        showProcessingState('processing');
        updateProcessingStatus('Iniciando procesamiento...', 'Preparando análisis de vídeo');
        
        // Start processing
        await processVideo(currentVideoId);
        updateProcessingStatus('Procesando vídeo...', 'Analizando fotogramas para detectar tiradas');
        
        // Poll for status
        startStatusPolling(currentVideoId);
        
    } catch (error) {
        showToast(error.message, 'error');
        processingSection.classList.add('hidden');
    } finally {
        uploadBtn.disabled = false;
        uploadBtn.innerHTML = `
            <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12"/>
            </svg>
            <span>Subir y Procesar</span>
        `;
        clearSelectedFile();
    }
}

function startStatusPolling(videoId) {
    if (pollingInterval) {
        clearInterval(pollingInterval);
    }
    
    pollingInterval = setInterval(async () => {
        try {
            const status = await getVideoStatus(videoId);
            
            if (status.status === 'completed') {
                clearInterval(pollingInterval);
                pollingInterval = null;
                
                showProcessingState('completed');
                updateProcessingStatus('¡Procesamiento completado!', `Se detectaron ${status.total_count} cervezas`);
                
                // Get full details and display
                const details = await getVideoDetails(videoId);
                displayResults(details);
                displayTimeline(details.pour_events);
                
                showToast('Procesamiento completado', 'success');
                loadHistory();
                
            } else if (status.status === 'error') {
                clearInterval(pollingInterval);
                pollingInterval = null;
                
                showProcessingState('error');
                updateProcessingStatus('Error en el procesamiento', status.error_message || 'Error desconocido');
                
                showToast('Error al procesar el vídeo', 'error');
                loadHistory();
            }
            
        } catch (error) {
            console.error('Error polling status:', error);
        }
    }, 2000); // Poll every 2 seconds
}

async function loadHistory() {
    try {
        const videos = await getVideoHistory();
        updateHistoryTable(videos);
    } catch (error) {
        console.error('Error loading history:', error);
    }
}

// Event Listeners
dropZone.addEventListener('click', () => fileInput.click());
selectBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    fileInput.click();
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFileSelect(e.target.files[0]);
    }
});

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('drag-over');
});

dropZone.addEventListener('dragleave', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    if (e.dataTransfer.files.length > 0) {
        handleFileSelect(e.dataTransfer.files[0]);
    }
});

clearFile.addEventListener('click', clearSelectedFile);
uploadBtn.addEventListener('click', handleUploadAndProcess);
refreshHistory.addEventListener('click', loadHistory);

// Check API health on load
async function checkApiHealth() {
    const apiStatus = document.getElementById('api-status');
    try {
        const response = await fetch(`${API_BASE_URL}/api/health`);
        if (response.ok) {
            apiStatus.innerHTML = `
                <span class="w-3 h-3 bg-green-400 rounded-full animate-pulse"></span>
                <span class="text-sm">API Conectada</span>
            `;
        } else {
            throw new Error('API not healthy');
        }
    } catch (error) {
        apiStatus.innerHTML = `
            <span class="w-3 h-3 bg-red-400 rounded-full"></span>
            <span class="text-sm">API Desconectada</span>
        `;
    }
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    checkApiHealth();
    loadHistory();
    
    // Check API health periodically
    setInterval(checkApiHealth, 30000);
});
