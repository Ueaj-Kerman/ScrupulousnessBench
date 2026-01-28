const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const preview = document.getElementById('preview');
const pasteBtn = document.getElementById('paste-image');
const tweetInput = document.getElementById('tweet');
const candidateBox = document.getElementById('candidate');
const questionInput = document.getElementById('question');
const answerInput = document.getElementById('answer');
const nameInput = document.getElementById('name');
const useCandidateBtn = document.getElementById('use-candidate');
const regenerateNameBtn = document.getElementById('regenerate-name');
const clearBtn = document.getElementById('clear');
const saveBtn = document.getElementById('save');
const statusDiv = document.getElementById('status');
const examplesList = document.getElementById('examples-list');

let currentImage = null;
let currentImageExt = 'png';
let editingName = null;
let questionDebounceTimer = null;
let nameDebounceTimer = null;
let nameGenerated = false;

async function loadExamples() {
    try {
        const resp = await fetch('/api/examples');
        const examples = await resp.json();

        examplesList.innerHTML = '';
        examples.forEach(ex => {
            const item = document.createElement('div');
            item.className = 'example-item' + (editingName === ex.name ? ' active' : '');
            item.innerHTML = `
                <div class="example-content">
                    <div class="name">${ex.name}</div>
                    <div class="question">${ex.question || '(no question)'}</div>
                </div>
                <button class="delete-btn" title="Delete">×</button>
            `;
            item.querySelector('.example-content').addEventListener('click', () => loadExample(ex));
            item.querySelector('.delete-btn').addEventListener('click', async (e) => {
                e.stopPropagation();
                if (confirm(`Delete "${ex.name}"?`)) {
                    await fetch(`/api/example/${ex.name}`, { method: 'DELETE' });
                    if (editingName === ex.name) clearForm();
                    loadExamples();
                }
            });
            examplesList.appendChild(item);
        });
    } catch (err) {
        console.error('Failed to load examples:', err);
    }
}

async function loadExample(ex) {
    editingName = ex.name;
    nameGenerated = true;
    nameInput.value = ex.name;
    questionInput.value = ex.question || '';
    answerInput.value = ex.answer || '';
    tweetInput.value = ex.tweet || '';
    currentImageExt = ex.image_ext || 'png';

    if (ex.has_image) {
        preview.src = `/api/image/${ex.name}?t=${Date.now()}`;
        preview.classList.remove('hidden');
        dropZone.classList.add('has-image');
        currentImage = null;
    } else {
        clearImage();
    }

    candidateBox.innerHTML = '<span class="placeholder">Enter tweet text to generate...</span>';
    loadExamples();
}

function clearImage() {
    preview.classList.add('hidden');
    preview.src = '';
    dropZone.classList.remove('has-image');
    currentImage = null;
}

function clearForm() {
    editingName = null;
    nameGenerated = false;
    clearImage();
    tweetInput.value = '';
    candidateBox.innerHTML = '<span class="placeholder">Enter tweet text to generate...</span>';
    questionInput.value = '';
    answerInput.value = '';
    nameInput.value = '';
    hideStatus();
    loadExamples();
}

function showStatus(message, type) {
    statusDiv.textContent = message;
    statusDiv.className = `status ${type}`;
}

function hideStatus() {
    statusDiv.className = 'status hidden';
}

function handleFile(file) {
    if (!file || !file.type.startsWith('image/')) return;

    currentImage = file;
    const ext = file.name.split('.').pop().toLowerCase();
    currentImageExt = ['png', 'jpg', 'jpeg', 'gif', 'webp'].includes(ext) ? ext : 'png';

    const reader = new FileReader();
    reader.onload = e => {
        preview.src = e.target.result;
        preview.classList.remove('hidden');
        dropZone.classList.add('has-image');
    };
    reader.readAsDataURL(file);

    if (!nameInput.value && !editingName && !nameGenerated) {
        generateName();
        nameGenerated = true;
    }
}

async function pasteFromClipboard() {
    try {
        const items = await navigator.clipboard.read();
        for (const item of items) {
            const imageType = item.types.find(t => t.startsWith('image/'));
            if (imageType) {
                const blob = await item.getType(imageType);
                const ext = imageType.split('/')[1] || 'png';
                const file = new File([blob], `pasted.${ext}`, { type: imageType });
                handleFile(file);
                return;
            }
        }
        showStatus('No image in clipboard', 'error');
    } catch (err) {
        showStatus('Failed to read clipboard', 'error');
    }
}

pasteBtn.addEventListener('click', pasteFromClipboard);

document.addEventListener('paste', e => {
    const items = e.clipboardData?.items;
    if (!items) return;

    for (const item of items) {
        if (item.type.startsWith('image/')) {
            e.preventDefault();
            const file = item.getAsFile();
            if (file) handleFile(file);
            return;
        }
    }
});

dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('keydown', e => {
    if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        fileInput.click();
    }
});

fileInput.addEventListener('change', () => {
    if (fileInput.files.length > 0) {
        handleFile(fileInput.files[0]);
    }
});

dropZone.addEventListener('dragover', e => {
    e.preventDefault();
    dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
});

dropZone.addEventListener('drop', e => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) {
        handleFile(e.dataTransfer.files[0]);
    }
});

tweetInput.addEventListener('input', () => {
    if (questionDebounceTimer) {
        clearTimeout(questionDebounceTimer);
    }
    questionDebounceTimer = setTimeout(generateQuestion, 300);

    maybeGenerateName();
});

questionInput.addEventListener('input', () => {
    maybeGenerateName();
});

function maybeGenerateName() {
    if (nameGenerated || editingName || nameInput.value) return;

    const hasContent = tweetInput.value.trim() || questionInput.value.trim();
    if (!hasContent) return;

    if (nameDebounceTimer) {
        clearTimeout(nameDebounceTimer);
    }
    nameDebounceTimer = setTimeout(() => {
        if (!nameGenerated && !editingName && !nameInput.value) {
            generateName();
            nameGenerated = true;
        }
    }, 500);
}

async function generateQuestion() {
    const tweet = tweetInput.value.trim();
    if (!tweet) {
        candidateBox.innerHTML = '<span class="placeholder">Enter tweet text to generate...</span>';
        return;
    }

    candidateBox.classList.add('loading');
    candidateBox.textContent = 'Generating...';

    try {
        const formData = new FormData();
        formData.append('tweet', tweet);

        const resp = await fetch('/api/generate-question', { method: 'POST', body: formData });
        if (!resp.ok) throw new Error('Failed to generate');

        const data = await resp.json();
        candidateBox.textContent = data.question;
    } catch (err) {
        candidateBox.innerHTML = '<span class="placeholder">Failed to generate question</span>';
    } finally {
        candidateBox.classList.remove('loading');
    }
}

async function generateName() {
    try {
        const formData = new FormData();
        formData.append('tweet', tweetInput.value);
        formData.append('question', questionInput.value);

        const resp = await fetch('/api/generate-name', { method: 'POST', body: formData });
        if (!resp.ok) throw new Error('Failed to generate name');

        const data = await resp.json();
        nameInput.value = data.name;
    } catch (err) {
        console.error('Failed to generate name:', err);
    }
}

useCandidateBtn.addEventListener('click', () => {
    const candidate = candidateBox.textContent;
    if (candidate && !candidate.includes('Enter tweet') && !candidate.includes('Failed') && candidate !== 'Generating...') {
        questionInput.value = candidate;
        questionInput.focus();
    }
});

regenerateNameBtn.addEventListener('click', generateName);

clearBtn.addEventListener('click', clearForm);

saveBtn.addEventListener('click', async () => {
    const name = nameInput.value.trim();
    const question = questionInput.value.trim();
    const answer = answerInput.value.trim();

    if (!question) {
        showStatus('Question is required', 'error');
        questionInput.focus();
        return;
    }

    if (!answer) {
        showStatus('Answer is required', 'error');
        answerInput.focus();
        return;
    }

    if (!name) {
        showStatus('Filename is required', 'error');
        nameInput.focus();
        return;
    }

    if (!currentImage && !editingName) {
        showStatus('Image is required', 'error');
        return;
    }

    saveBtn.disabled = true;
    saveBtn.textContent = 'Saving...';

    try {
        const formData = new FormData();
        formData.append('name', name);
        formData.append('question', question);
        formData.append('answer', answer);
        formData.append('tweet', tweetInput.value);
        formData.append('image_ext', currentImageExt);

        if (currentImage) {
            formData.append('image', currentImage);
        }

        const resp = await fetch('/api/save', { method: 'POST', body: formData });
        if (!resp.ok) {
            const err = await resp.json();
            throw new Error(err.detail || 'Failed to save');
        }

        const data = await resp.json();
        showStatus(`Saved as ${data.name}`, 'success');

        clearForm();
        showStatus(`Saved as ${data.name}`, 'success');
    } catch (err) {
        showStatus(err.message, 'error');
    } finally {
        saveBtn.disabled = false;
        saveBtn.textContent = 'Save Example';
    }
});

document.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) {
        const active = document.activeElement;
        if (active === saveBtn || (active.tagName !== 'TEXTAREA' && active.tagName !== 'INPUT')) {
            e.preventDefault();
            saveBtn.click();
        }
    }
});

loadExamples();
