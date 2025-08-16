class QuranReciteAI {
    constructor() {
        this.mediaRecorder = null;
        this.audioChunks = [];
        this.recordingTimer = null;
        this.recordingStartTime = null;
        this.currentAudioFile = null;
        this.currentRecordedBlob = null;
        
        this.initializeElements();
        this.bindEvents();
    }
    
    // Utility function to safely create blob URL
    createSafeBlobURL(blob) {
        try {
            return URL.createObjectURL(blob);
        } catch (error) {
            console.error('Error creating blob URL:', error);
            throw new Error('Gagal membuat URL preview. Silakan coba lagi.');
        }
    }
    
    // Utility function to safely revoke blob URL
    revokeSafeBlobURL(url) {
        try {
            if (url && url.startsWith('blob:')) {
                URL.revokeObjectURL(url);
                return true;
            }
        } catch (error) {
            console.error('Error revoking blob URL:', error);
        }
        return false;
    }

    initializeElements() {
        // Tab elements
        this.tabButtons = document.querySelectorAll('.tab-btn');
        this.tabContents = document.querySelectorAll('.tab-content');
        
        // Upload elements
        this.uploadArea = document.getElementById('upload-area');
        this.audioFileInput = document.getElementById('audio-file');
        this.filePreview = document.getElementById('file-preview');
        this.fileName = document.getElementById('file-name');
        this.fileSize = document.getElementById('file-size');
        this.audioPreview = document.getElementById('audio-preview');
        this.removeFileBtn = document.getElementById('remove-file');
        this.processUploadBtn = document.getElementById('process-upload');
        
        // Record elements
        this.startRecordBtn = document.getElementById('start-record');
        this.stopRecordBtn = document.getElementById('stop-record');
        this.recordTimer = document.getElementById('record-timer');
        this.recordIndicator = document.getElementById('record-indicator');
        this.recordedAudio = document.getElementById('recorded-audio');
        this.recordedPreview = document.getElementById('recorded-preview');
        this.reRecordBtn = document.getElementById('re-record');
        this.processRecordBtn = document.getElementById('process-record');
        
        // Processing elements
        this.processingSection = document.getElementById('processing-section');
        this.processingSteps = {
            step1: document.getElementById('step-1'),
            step2: document.getElementById('step-2'),
            step3: document.getElementById('step-3')
        };
        
        // Results elements
        this.resultsSection = document.getElementById('results-section');
        // Removed matches count element
        this.transcriptionText = document.getElementById('transcription-text');
        this.allMatchesContainer = document.getElementById('all-matches-container');
        this.newSearchBtn = document.getElementById('new-search');
        
        // Error elements
        this.errorSection = document.getElementById('error-section');
        this.errorMessage = document.getElementById('error-message');
        this.errorTranscription = document.getElementById('error-transcription');
        this.tryAgainBtn = document.getElementById('try-again');
    }

    bindEvents() {
        // Tab switching
        this.tabButtons.forEach(btn => {
            btn.addEventListener('click', () => this.switchTab(btn.dataset.tab));
        });

        // Upload events
        this.uploadArea.addEventListener('click', () => this.audioFileInput.click());
        this.uploadArea.addEventListener('dragover', (e) => this.handleDragOver(e));
        this.uploadArea.addEventListener('dragleave', (e) => this.handleDragLeave(e));
        this.uploadArea.addEventListener('drop', (e) => this.handleDrop(e));
        this.audioFileInput.addEventListener('change', (e) => this.handleFileSelect(e));
        this.removeFileBtn.addEventListener('click', () => this.removeFile());
        this.processUploadBtn.addEventListener('click', () => this.processAudio('upload'));

        // Record events
        this.startRecordBtn.addEventListener('click', () => this.startRecording());
        this.stopRecordBtn.addEventListener('click', () => this.stopRecording());
        this.reRecordBtn.addEventListener('click', () => this.resetRecording());
        this.processRecordBtn.addEventListener('click', () => this.processAudio('record'));

        // Navigation events
        this.newSearchBtn.addEventListener('click', () => this.resetApp());
        this.tryAgainBtn.addEventListener('click', () => this.resetApp());
    }

    switchTab(tabName) {
        // Update tab buttons
        this.tabButtons.forEach(btn => {
            btn.classList.toggle('active', btn.dataset.tab === tabName);
        });

        // Update tab contents
        this.tabContents.forEach(content => {
            content.classList.toggle('active', content.id === `${tabName}-tab`);
        });

        // Reset states when switching tabs
        this.resetStates();
    }

    resetStates() {
        this.hideAllSections();
        this.removeFile();
        this.resetRecording();
    }

    hideAllSections() {
        this.processingSection.style.display = 'none';
        this.resultsSection.style.display = 'none';
        this.errorSection.style.display = 'none';
    }

    // Upload functionality
    handleDragOver(e) {
        e.preventDefault();
        this.uploadArea.classList.add('dragover');
    }

    handleDragLeave(e) {
        e.preventDefault();
        this.uploadArea.classList.remove('dragover');
    }

    handleDrop(e) {
        e.preventDefault();
        this.uploadArea.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            this.handleFile(files[0]);
        }
    }

    handleFileSelect(e) {
        const file = e.target.files[0];
        if (file) {
            this.handleFile(file);
        }
    }

    handleFile(file) {
        // Validate file type
        const allowedTypes = ['audio/mp3', 'audio/wav', 'audio/m4a', 'audio/mpeg', 'audio/x-wav'];
        if (!allowedTypes.includes(file.type)) {
            alert('Format file tidak didukung. Gunakan MP3, WAV, atau M4A.');
            return;
        }

        // Validate file size (50MB)
        if (file.size > 50 * 1024 * 1024) {
            alert('Ukuran file terlalu besar. Maksimal 50MB.');
            return;
        }

        this.currentAudioFile = file;
        this.showFilePreview(file);
    }

    showFilePreview(file) {
        this.fileName.textContent = file.name;
        this.fileSize.textContent = this.formatFileSize(file.size);
        
        try {
            // Revoke previous blob URL if exists
            this.revokeSafeBlobURL(this.audioPreview.src);
            
            const url = this.createSafeBlobURL(file);
            this.audioPreview.src = url;
            
            // Add error handler for audio element
            this.audioPreview.onerror = (e) => {
                console.error('Audio preview error:', e);
                // Try to recreate the blob URL
                setTimeout(() => {
                    try {
                        const newUrl = this.createSafeBlobURL(file);
                        this.audioPreview.src = newUrl;
                    } catch (retryError) {
                        console.error('Failed to recreate blob URL:', retryError);
                        alert('Gagal memuat ulang preview audio.');
                    }
                }, 100);
            };
            
            this.uploadArea.style.display = 'none';
            this.filePreview.style.display = 'block';
        } catch (error) {
            console.error('Error in showFilePreview:', error);
            alert(error.message || 'Terjadi kesalahan saat memuat preview audio. Silakan coba lagi.');
        }
    }

    removeFile() {
        this.currentAudioFile = null;
        this.audioFileInput.value = '';
        this.uploadArea.style.display = 'block';
        this.filePreview.style.display = 'none';
        
        this.revokeSafeBlobURL(this.audioPreview.src);
        this.audioPreview.src = '';
        this.audioPreview.onerror = null; // Remove error handler
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    // Recording functionality
    async startRecording() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            
            this.mediaRecorder = new MediaRecorder(stream);
            this.audioChunks = [];
            
            this.mediaRecorder.ondataavailable = (event) => {
                this.audioChunks.push(event.data);
            };
            
            this.mediaRecorder.onstop = () => {
                try {
                    const audioBlob = new Blob(this.audioChunks, { type: 'audio/wav' });
                    this.currentRecordedBlob = audioBlob;
                    
                    // Revoke previous blob URL if exists
                    this.revokeSafeBlobURL(this.recordedPreview.src);
                    
                    const url = this.createSafeBlobURL(audioBlob);
                    this.recordedPreview.src = url;
                    
                    // Add error handler for recorded audio element
                    this.recordedPreview.onerror = (e) => {
                        console.error('Recorded audio preview error:', e);
                        // Try to recreate the blob URL
                        setTimeout(() => {
                            try {
                                const newUrl = this.createSafeBlobURL(audioBlob);
                                this.recordedPreview.src = newUrl;
                            } catch (retryError) {
                                console.error('Failed to recreate recorded blob URL:', retryError);
                                alert('Gagal memuat ulang preview rekaman.');
                            }
                        }, 100);
                    };
                    
                    this.recordedAudio.style.display = 'block';
                } catch (error) {
                    console.error('Error in recording onstop:', error);
                    alert(error.message || 'Terjadi kesalahan saat memuat preview rekaman. Silakan coba merekam ulang.');
                }
                
                // Stop all tracks to release microphone
                stream.getTracks().forEach(track => track.stop());
            };
            
            this.mediaRecorder.start();
            this.startRecordingUI();
            
        } catch (error) {
            console.error('Error accessing microphone:', error);
            alert('Tidak dapat mengakses mikrofon. Pastikan izin mikrofon telah diberikan.');
        }
    }

    startRecordingUI() {
        this.startRecordBtn.disabled = true;
        this.stopRecordBtn.disabled = false;
        this.recordIndicator.classList.add('recording');
        
        this.recordingStartTime = Date.now();
        this.recordingTimer = setInterval(() => {
            const elapsed = Date.now() - this.recordingStartTime;
            const minutes = Math.floor(elapsed / 60000);
            const seconds = Math.floor((elapsed % 60000) / 1000);
            this.recordTimer.textContent = `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
        }, 1000);
    }

    stopRecording() {
        if (this.mediaRecorder && this.mediaRecorder.state === 'recording') {
            this.mediaRecorder.stop();
        }
        this.stopRecordingUI();
    }

    stopRecordingUI() {
        this.startRecordBtn.disabled = false;
        this.stopRecordBtn.disabled = true;
        this.recordIndicator.classList.remove('recording');
        
        if (this.recordingTimer) {
            clearInterval(this.recordingTimer);
            this.recordingTimer = null;
        }
    }

    resetRecording() {
        this.stopRecording();
        this.recordTimer.textContent = '00:00';
        this.recordedAudio.style.display = 'none';
        this.currentRecordedBlob = null;
        
        this.revokeSafeBlobURL(this.recordedPreview.src);
        this.recordedPreview.src = '';
        this.recordedPreview.onerror = null; // Remove error handler
    }

    // Audio processing
    async processAudio(source) {
        let audioData;
        
        if (source === 'upload' && this.currentAudioFile) {
            audioData = this.currentAudioFile;
        } else if (source === 'record' && this.currentRecordedBlob) {
            audioData = this.currentRecordedBlob;
        } else {
            alert('Tidak ada audio untuk diproses.');
            return;
        }

        this.showProcessing();
        
        try {
            const formData = new FormData();
            formData.append('audio', audioData, 'audio.wav');
            
            // Update processing steps
            this.updateProcessingStep(1);
            
            const response = await fetch('/api/identify-verse', {
                method: 'POST',
                body: formData
            });
            
            this.updateProcessingStep(2);
            
            const result = await response.json();
            
            this.updateProcessingStep(3);
            
            // Wait a bit for better UX
            setTimeout(() => {
                if (result.success) {
                    this.showResults(result);
                } else {
                    this.showError(result);
                }
            }, 1000);
            
        } catch (error) {
            console.error('Error processing audio:', error);
            this.showError({
                message: 'Terjadi kesalahan saat memproses audio. Silakan coba lagi.',
                transcription: ''
            });
        }
    }

    showProcessing() {
        this.hideAllSections();
        this.processingSection.style.display = 'block';
        
        // Reset processing steps
        Object.values(this.processingSteps).forEach(step => {
            step.classList.remove('active');
        });
    }

    updateProcessingStep(stepNumber) {
        const stepKey = `step${stepNumber}`;
        if (this.processingSteps[stepKey]) {
            this.processingSteps[stepKey].classList.add('active');
        }
    }

    showResults(result) {
        this.hideAllSections();
        this.resultsSection.style.display = 'block';
        
        // Update transcription
        this.transcriptionText.textContent = result.transcription;
        
        // Clear previous matches
        this.allMatchesContainer.innerHTML = '';
        
        // Display single result with all matches listed
        this.createSingleResultWithAllMatches(result);
    }

    createSingleResultWithAllMatches(result) {
        const verseType = result.verse_type || 'single'; // Get verse type from backend
        
        const resultContainer = document.createElement('div');
        resultContainer.className = 'unified-result-container';
        
        let displayArabicText, displayTranslation, matchesList;
        
        // Use display_text from backend if available, otherwise fallback to old method
        if (result.display_text) {
            displayArabicText = result.display_text.arabic;
            displayTranslation = result.display_text.translation;
        } else {
            // Fallback for backward compatibility
            const bestMatch = result.all_matches[0];
            displayArabicText = bestMatch.verse_text.replace(/^\(\d+\)\s*/, '').trim();
            displayTranslation = bestMatch.translation.replace(/^\(\d+\)\s*/, '').trim();
        }
        
        // Generate matches list based on verse type
        if (verseType === 'consecutive') {
            // For consecutive verses: only show surah name with verse range
            const match = result.all_matches[0];
            const verseRange = match.verse_range || `${match.verse_number}${match.end_verse_number ? '-' + match.end_verse_number : ''}`;
            matchesList = `<div class="match-list-item best-score">
                            Surat ${match.surah_name_latin} Ayat ${verseRange}
                            <span class="match-score">(${Math.round(match.similarity_score)}%)</span>
                        </div>`;
        } else {
            // For single/identical verses: show all verse locations
            matchesList = result.all_matches.map((match, index) => {
                const scoreClass = index === 0 ? 'best-score' : 'other-score';
                const displayText = `Surat ${match.surah_name_latin} Ayat ${match.verse_number}`;
                
                return `<div class="match-list-item ${scoreClass}">
                            ${displayText}
                            <span class="match-score">(${Math.round(match.similarity_score)}%)</span>
                        </div>`;
            }).join('');
        }
        
        resultContainer.innerHTML = `
            <div class="unified-arabic-text">${displayArabicText}</div>
            <div class="unified-translation">${displayTranslation}</div>
            <div class="unified-matches-section">
                <div class="unified-matches-title">Surat dan Ayat yang Teridentifikasi:</div>
                <div class="unified-matches-list">
                    ${matchesList}
                </div>
            </div>

        `;
        
        this.allMatchesContainer.appendChild(resultContainer);
    }

    createImprovedMatchItem(match, isBestMatch = false) {
        const matchItem = document.createElement('div');
        matchItem.className = `improved-match-item ${isBestMatch ? 'best-match' : ''}`;
        
        matchItem.innerHTML = `
            ${isBestMatch ? '<div class="best-match-badge"><i class="fas fa-star"></i> Hasil Terbaik</div>' : ''}
            <div class="improved-arabic-text">${match.verse_text}</div>
            <div class="improved-translation">${match.translation}</div>
            <div class="improved-location">
                <span class="improved-surah-info">Surat ${match.surah_name_latin}</span>
                <span class="improved-similarity-score">(${Math.round(match.similarity_score)}%)</span>
            </div>

        `;
        
        this.allMatchesContainer.appendChild(matchItem);
    }

    createSingleMatchResult(result) {
        const matchItem = document.createElement('div');
        matchItem.className = 'single-match-result';
        
        matchItem.innerHTML = `
            <div class="result-arabic-text">${result.verse_text}</div>
            <div class="result-translation">${result.translation}</div>
            <div class="result-location">
                <span class="surah-info">Surat ${result.surah_name_latin}</span>
                <span class="similarity-score">(${Math.round(result.similarity_score)}%)</span>
            </div>

        `;
        
        this.allMatchesContainer.appendChild(matchItem);
    }

    createMatchItem(match, isBestMatch = false) {
        const matchItem = document.createElement('div');
        matchItem.className = `match-item ${isBestMatch ? 'best-match' : ''}`;
        
        matchItem.innerHTML = `
            <div class="match-header">
                <div class="match-location">
                    <div class="surah-name-arabic">${match.surah_name_arabic}</div>
                    <div class="surah-name-latin">${match.surah_name} (${match.surah_number})</div>
                    <div class="verse-location"></div>
                </div>
                <div class="similarity-score">
                    <span class="score-label">Skor:</span>
                    <span class="score-value">${Math.round(match.similarity_score)}%</span>
                </div>
            </div>
            
            <div class="match-content">
                <div class="match-arabic-text">${match.verse_text}</div>
                <div class="match-translation">${match.translation}</div>
            </div>
            
        `;
        
        this.allMatchesContainer.appendChild(matchItem);
    }











    showError(result) {
        this.hideAllSections();
        this.errorSection.style.display = 'block';
        
        this.errorMessage.textContent = result.message || 'Maaf, ayat tidak dapat diidentifikasi. Coba rekam dengan lebih jelas.';
        this.errorTranscription.textContent = result.transcription || 'Tidak ada transkripsi tersedia.';
    }

    resetApp() {
        this.hideAllSections();
        this.resetStates();
        
        // Reset to upload tab
        this.switchTab('upload');
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new QuranReciteAI();
});

// Add some utility functions for better UX
document.addEventListener('DOMContentLoaded', () => {
    // Add smooth scrolling for better navigation
    const smoothScroll = (target) => {
        target.scrollIntoView({
            behavior: 'smooth',
            block: 'start'
        });
    };

    // Add loading states for buttons
    const addLoadingState = (button, text = 'Memproses...') => {
        const originalText = button.innerHTML;
        button.innerHTML = `<i class="fas fa-spinner fa-spin"></i> ${text}`;
        button.disabled = true;
        
        return () => {
            button.innerHTML = originalText;
            button.disabled = false;
        };
    };

    // Add error handling for network issues
    window.addEventListener('online', () => {
        console.log('Connection restored');
    });

    window.addEventListener('offline', () => {
        alert('Koneksi internet terputus. Beberapa fitur mungkin tidak berfungsi.');
    });
    
    // Add global error handler for blob URL issues
    window.addEventListener('error', (e) => {
        if (e.target && e.target.tagName === 'AUDIO' && e.target.src && e.target.src.startsWith('blob:')) {
            console.error('Audio blob URL error:', e);
            // Try to reload the audio element
            const audioElement = e.target;
            const currentSrc = audioElement.src;
            audioElement.src = '';
            setTimeout(() => {
                audioElement.src = currentSrc;
            }, 100);
        }
    }, true);
    
    // Add unhandled promise rejection handler
    window.addEventListener('unhandledrejection', (e) => {
        if (e.reason && e.reason.message && e.reason.message.includes('blob')) {
            console.error('Unhandled blob URL promise rejection:', e.reason);
            e.preventDefault(); // Prevent the error from being logged to console
        }
    });
});

// Add keyboard shortcuts for better accessibility
document.addEventListener('keydown', (e) => {
    // Ctrl/Cmd + U for upload tab
    if ((e.ctrlKey || e.metaKey) && e.key === 'u') {
        e.preventDefault();
        document.querySelector('[data-tab="upload"]').click();
    }
    
    // Ctrl/Cmd + R for record tab
    if ((e.ctrlKey || e.metaKey) && e.key === 'r') {
        e.preventDefault();
        document.querySelector('[data-tab="record"]').click();
    }
    
    // Space to start/stop recording (when in record tab)
    if (e.code === 'Space' && document.querySelector('#record-tab').classList.contains('active')) {
        e.preventDefault();
        const startBtn = document.getElementById('start-record');
        const stopBtn = document.getElementById('stop-record');
        
        if (!startBtn.disabled) {
            startBtn.click();
        } else if (!stopBtn.disabled) {
            stopBtn.click();
        }
    }
});