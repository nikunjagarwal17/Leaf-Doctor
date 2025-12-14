/**
 * Leaf Doctor - Enhanced Chat Interface
 * Modern, accessible chat with improved UX
 */

// DOM Elements
const chatStream = document.getElementById("chatStream");
const imageInput = document.getElementById("imageInput");
const chooseImagesBtn = document.getElementById("chooseImagesBtn");
const addImagesBtn = document.getElementById("addImagesBtn");
const dropZone = document.getElementById("dropZone");
const thumbList = document.getElementById("thumbList");
const imageCount = document.getElementById("imageCount");
const predictBtn = document.getElementById("predictBtn");
const noteField = document.getElementById("noteField");
const promptField = document.getElementById("promptField");
const sendBtn = document.getElementById("sendBtn");
const thinkingToggle = document.getElementById("thinkingToggle");
const voiceBtn = document.getElementById("voiceBtn");
const speakToggle = document.getElementById("speakToggle");
const advisorPanel = document.getElementById("advisorPanel");
const clearChatBtn = document.getElementById("clearChatBtn");

// State
let latestDiagnosis = null;
let advisorBusy = false;
let recognition = null;
let selectedFiles = [];

// ============================================================================
// MESSAGE HANDLING
// ============================================================================

function pushMessage(role, nodeOrHtml) {
    const wrapper = document.createElement("div");
    wrapper.classList.add("message", role);
    
    if (typeof nodeOrHtml === "string") {
        wrapper.innerHTML = nodeOrHtml;
    } else {
        wrapper.appendChild(nodeOrHtml);
    }
    
    chatStream.appendChild(wrapper);
    chatStream.scrollTop = chatStream.scrollHeight;
    
    // Re-initialize lucide icons in new content
    if (window.lucide) {
        lucide.createIcons();
    }
    
    // Speak if enabled
    if (role === "bot" && speakToggle?.checked) {
        speak(nodeOrHtml instanceof HTMLElement ? nodeOrHtml.innerText : nodeOrHtml);
    }
}

function speak(text) {
    if (!("speechSynthesis" in window)) return;
    const utter = new SpeechSynthesisUtterance(text);
    utter.rate = 1;
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(utter);
}

function clearChat() {
    chatStream.innerHTML = "";
    latestDiagnosis = null;
    advisorPanel?.classList.add("hidden");
    pushMessage("bot", createWelcomeMessage());
}

function createWelcomeMessage() {
    const div = document.createElement("div");
    div.innerHTML = `
        <strong>👋 Welcome to Leaf Doctor!</strong>
        <p style="margin-top: 8px;">
            Upload up to 4 leaf images to get an accurate disease diagnosis. 
            Our ensemble CNN model analyzes multiple angles to improve confidence.
        </p>
        <p style="margin-top: 8px; color: var(--text-muted); font-size: 0.875rem;">
            After diagnosis, you can ask follow-up questions grounded in our embedded disease knowledge base.
        </p>
    `;
    return div;
}

// ============================================================================
// THINKING/REASONING DISPLAY
// ============================================================================

function renderThinking(reasoning, context, reasoningChain) {
    if (!reasoning && !context?.length && !reasoningChain?.length) return null;
    
    const details = document.createElement("details");
    details.classList.add("thinking");
    if (thinkingToggle?.checked) {
        details.setAttribute("open", "open");
    }
    
    const summary = document.createElement("summary");
    summary.innerHTML = reasoningChain?.length 
        ? '<i data-lucide="git-branch" style="width:14px;height:14px;display:inline;vertical-align:middle;margin-right:6px;"></i>Agent Reasoning' 
        : '<i data-lucide="brain" style="width:14px;height:14px;display:inline;vertical-align:middle;margin-right:6px;"></i>Show Thinking';
    details.appendChild(summary);

    const block = document.createElement("div");
    
    // Reasoning Chain (Agent steps)
    if (reasoningChain?.length) {
        const chainDiv = document.createElement("div");
        chainDiv.classList.add("reasoning-chain");
        
        const chainTitle = document.createElement("p");
        chainTitle.classList.add("chain-title");
        chainTitle.innerHTML = "<strong>Execution Plan:</strong>";
        chainDiv.appendChild(chainTitle);
        
        const stepList = document.createElement("ul");
        stepList.classList.add("chain-steps");
        
        reasoningChain.forEach((step) => {
            const li = document.createElement("li");
            li.classList.add("chain-step", step.status);
            
            const statusIcon = {
                "completed": "✅",
                "failed": "❌",
                "needs_retry": "🔄",
                "in_progress": "⏳",
                "pending": "⏸️"
            }[step.status] || "❓";
            
            li.innerHTML = `
                <span class="step-icon">${statusIcon}</span>
                <span class="step-content">
                    <strong>Step ${step.step}:</strong> ${step.action}
                    ${step.tool ? `<span class="tool-badge">${step.tool}</span>` : ""}
                    ${step.retries > 0 ? `<span class="retry-badge">Retried ${step.retries}x</span>` : ""}
                    ${step.reflection ? `<div class="step-reflection">↳ ${step.reflection}</div>` : ""}
                </span>
            `;
            stepList.appendChild(li);
        });
        
        chainDiv.appendChild(stepList);
        block.appendChild(chainDiv);
    }
    
    // Plain reasoning text
    if (reasoning) {
        const para = document.createElement("p");
        para.classList.add("muted");
        para.style.marginTop = "12px";
        para.textContent = reasoning;
        block.appendChild(para);
    }
    
    // Context list
    if (context?.length) {
        const contextTitle = document.createElement("p");
        contextTitle.innerHTML = "<strong style='font-size:0.85rem;'>Retrieved Context:</strong>";
        contextTitle.style.marginTop = "12px";
        block.appendChild(contextTitle);
        
        const list = document.createElement("ul");
        list.classList.add("context-list");
        context.forEach((item) => {
            const li = document.createElement("li");
            li.textContent = `${item.name} (relevance ${item.score.toFixed(2)})`;
            list.appendChild(li);
        });
        block.appendChild(list);
    }

    details.appendChild(block);
    return details;
}

// ============================================================================
// PREDICTION CARD
// ============================================================================

function renderPredictionCard(payload) {
    const card = document.createElement("div");
    card.classList.add("prediction-card");
    
    const agenticBadge = payload.agentic 
        ? `<span class="agentic-badge">⚡ Agentic</span>` 
        : "";
    
    const qualityWarning = payload.quality_check && !payload.quality_check.passed
        ? `<div class="quality-warning">
            <i data-lucide="alert-triangle" style="width:14px;height:14px;"></i>
            ${payload.quality_check.issues?.join(", ") || "Low confidence detected"}
           </div>`
        : "";
    
    const confidenceColor = payload.confidence >= 0.8 ? "var(--accent)" : 
                           payload.confidence >= 0.6 ? "#f59e0b" : "#ef4444";
    
    card.innerHTML = `
        <div class="prediction-head">
            <div>
                <p class="eyebrow">Detected Disease ${agenticBadge}</p>
                <strong>${payload.disease}</strong>
            </div>
            <div class="pill" style="background: ${confidenceColor}22; color: ${confidenceColor};">
                ${(payload.confidence * 100).toFixed(1)}% Confidence
            </div>
        </div>
        ${qualityWarning}
        <p style="color: var(--text-secondary); font-size: 0.9rem;">${payload.description}</p>
        ${payload.steps.length ? `
            <div class="disease-steps">
                <h4>
                    <i data-lucide="list-checks" style="width:12px;height:12px;"></i>
                    Treatment Steps
                </h4>
                <ul>
                    ${payload.steps.map(step => `<li>${step}</li>`).join("")}
                </ul>
            </div>
        ` : ""}
        <p class="muted">
            <i data-lucide="cpu" style="width:12px;height:12px;display:inline;vertical-align:middle;margin-right:4px;"></i>
            ${payload.agentic ? "Agentic CNN orchestration" : "Local CNN ensemble"} across ${payload.image_count || 1} image(s)
        </p>
    `;

    // Per-image breakdown
    if (payload.per_image?.length > 1) {
        const grid = document.createElement("div");
        grid.classList.add("mini-grid");
        payload.per_image.forEach((p) => {
            const chip = document.createElement("div");
            chip.classList.add("mini-chip");
            chip.innerHTML = `
                <strong>${p.disease}</strong>
                <span>${(p.confidence * 100).toFixed(1)}% • ${p.file}</span>
            `;
            grid.appendChild(chip);
        });
        card.appendChild(grid);
    }

    // Alternatives
    if (payload.alternatives?.length) {
        const alt = document.createElement("p");
        alt.classList.add("muted");
        alt.style.marginTop = "8px";
        alt.textContent = `Other possibilities: ${payload.alternatives
            .slice(1, 3)
            .map((a) => `${a.disease} ${(a.probability * 100).toFixed(1)}%`)
            .join(" | ")}`;
        card.appendChild(alt);
    }

    // Note context matches
    if (payload.note_context?.length) {
        const thinkingNode = renderThinking(null, payload.note_context, null);
        if (thinkingNode) {
            thinkingNode.querySelector("summary").innerHTML = 
                '<i data-lucide="file-text" style="width:14px;height:14px;display:inline;vertical-align:middle;margin-right:6px;"></i>Text Clue Matches';
            card.appendChild(thinkingNode);
        }
    }
    
    // Agent reasoning chain
    if (payload.reasoning_chain?.length) {
        const thinkingNode = renderThinking(null, null, payload.reasoning_chain);
        if (thinkingNode) card.appendChild(thinkingNode);
    }
    
    return card;
}

// ============================================================================
// IMAGE HANDLING
// ============================================================================

function updateThumbList() {
    thumbList.innerHTML = "";
    imageCount.textContent = selectedFiles.length;
    
    if (!selectedFiles.length) {
        thumbList.innerHTML = `
            <div class="empty-state">
                <i data-lucide="image-off"></i>
                <span>No images selected</span>
            </div>
        `;
        predictBtn.disabled = true;
        if (window.lucide) lucide.createIcons();
        return;
    }
    
    predictBtn.disabled = false;
    
    selectedFiles.forEach((file, index) => {
        const item = document.createElement("div");
        item.classList.add("thumb-item");
        
        const img = document.createElement("img");
        img.src = URL.createObjectURL(file);
        img.alt = file.name;
        item.appendChild(img);
        
        const removeBtn = document.createElement("button");
        removeBtn.classList.add("thumb-remove");
        removeBtn.innerHTML = '<i data-lucide="x" style="width:12px;height:12px;"></i>';
        removeBtn.onclick = (e) => {
            e.stopPropagation();
            selectedFiles.splice(index, 1);
            updateThumbList();
        };
        item.appendChild(removeBtn);
        
        thumbList.appendChild(item);
    });
    
    if (window.lucide) lucide.createIcons();
}

function handleFileSelect(files) {
    const newFiles = Array.from(files).filter(f => 
        f.type.startsWith("image/") && 
        !selectedFiles.some(sf => sf.name === f.name && sf.size === f.size)
    );
    
    const remaining = 4 - selectedFiles.length;
    if (remaining <= 0) {
        pushMessage("bot", "Maximum 4 images allowed. Remove some to add more.");
        return;
    }
    
    selectedFiles = [...selectedFiles, ...newFiles.slice(0, remaining)];
    updateThumbList();
    
    if (newFiles.length > remaining) {
        pushMessage("bot", `Added ${remaining} image(s). Maximum of 4 reached.`);
    }
}

// Drag and drop
dropZone?.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("drag-over");
});

dropZone?.addEventListener("dragleave", () => {
    dropZone.classList.remove("drag-over");
});

dropZone?.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("drag-over");
    handleFileSelect(e.dataTransfer.files);
});

dropZone?.addEventListener("click", () => imageInput?.click());

// File input change
imageInput?.addEventListener("change", (e) => {
    handleFileSelect(e.target.files);
    imageInput.value = ""; // Reset to allow same file selection
});

chooseImagesBtn?.addEventListener("click", (e) => {
    e.stopPropagation();
    imageInput?.click();
});

addImagesBtn?.addEventListener("click", () => imageInput?.click());

// ============================================================================
// PREDICTION
// ============================================================================

predictBtn?.addEventListener("click", async (e) => {
    e.preventDefault();
    
    if (!selectedFiles.length) {
        pushMessage("bot", "Please select at least one leaf image to run diagnosis.");
        return;
    }
    
    const formData = new FormData();
    selectedFiles.forEach((file) => formData.append("images", file));
    
    if (noteField?.value.trim()) {
        formData.append("note", noteField.value.trim());
    }

    predictBtn.disabled = true;
    predictBtn.innerHTML = '<span class="loading-spinner"></span> Analyzing...';
    
    pushMessage("user", `🔬 Diagnose using ${selectedFiles.length} image(s)`);
    pushMessage("bot", "Analyzing uploaded images with ensemble CNN...");

    try {
        const response = await fetch("/predict", { method: "POST", body: formData });
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || "Prediction failed");
        }
        
        latestDiagnosis = data;
        pushMessage("bot", renderPredictionCard(data));
        advisorPanel?.classList.remove("hidden");
        
        // Scroll advisor into view
        setTimeout(() => {
            advisorPanel?.scrollIntoView({ behavior: "smooth", block: "center" });
        }, 300);
        
    } catch (error) {
        pushMessage("bot", `❌ Prediction error: ${error.message}`);
    } finally {
        predictBtn.disabled = false;
        predictBtn.innerHTML = '<i data-lucide="scan"></i> Run Diagnosis';
        if (window.lucide) lucide.createIcons();
    }
});

// ============================================================================
// ADVISOR
// ============================================================================

sendBtn?.addEventListener("click", (e) => {
    e.preventDefault();
    const prompt = promptField?.value.trim();
    if (!prompt) return;
    triggerAdvisor(prompt);
    promptField.value = "";
});

promptField?.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendBtn?.click();
    }
});

async function triggerAdvisor(prompt) {
    if (advisorBusy) return;
    
    if (!latestDiagnosis?.disease) {
        pushMessage("bot", "Please run a diagnosis first so I can provide grounded advice.");
        return;
    }
    
    advisorBusy = true;
    sendBtn.disabled = true;
    sendBtn.innerHTML = '<span class="loading-spinner"></span>';
    
    pushMessage("user", prompt);

    try {
        const response = await fetch("/advisor", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ 
                prompt, 
                disease: latestDiagnosis.disease, 
                agentic: true 
            }),
        });
        
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || "Advisor request failed");

        const block = document.createElement("div");
        
        // Agentic indicator
        if (data.agentic) {
            const agenticInfo = document.createElement("div");
            agenticInfo.style.cssText = "margin-bottom: 12px; padding: 8px 12px; background: linear-gradient(135deg, rgba(99, 102, 241, 0.1), rgba(139, 92, 246, 0.1)); border-radius: 8px; font-size: 0.85rem;";
            agenticInfo.innerHTML = `<span class="agentic-badge">⚡ Agentic</span> ${data.agent_goal || ""}`;
            block.appendChild(agenticInfo);
        }
        
        // Main answer
        const answer = document.createElement("div");
        answer.innerHTML = `
            <div style="line-height: 1.6;">${data.message}</div>
            <p class="muted" style="margin-top: 12px;">
                <i data-lucide="database" style="width:12px;height:12px;display:inline;vertical-align:middle;margin-right:4px;"></i>
                Source: ${data.source}
            </p>
        `;
        block.appendChild(answer);
        
        // Quality indicators
        if (data.retrieval_quality && !data.retrieval_quality.passed) {
            const qualityNote = document.createElement("p");
            qualityNote.classList.add("quality-warning");
            qualityNote.innerHTML = `⚠️ ${data.retrieval_quality.issues?.join(", ") || "Low relevance context"}`;
            block.appendChild(qualityNote);
        }
        
        // Thinking section
        const thinkingNode = renderThinking(data.reasoning, data.context, data.reasoning_chain);
        if (thinkingNode) block.appendChild(thinkingNode);
        
        pushMessage("bot", block);
        
    } catch (error) {
        pushMessage("bot", `❌ Advisor error: ${error.message}`);
    } finally {
        advisorBusy = false;
        sendBtn.disabled = false;
        sendBtn.innerHTML = '<i data-lucide="send"></i><span>Send</span>';
        if (window.lucide) lucide.createIcons();
    }
}

// ============================================================================
// VOICE INPUT
// ============================================================================

voiceBtn?.addEventListener("click", () => {
    if (!("webkitSpeechRecognition" in window || "SpeechRecognition" in window)) {
        pushMessage("bot", "Voice input is not supported in this browser.");
        return;
    }
    
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    
    if (!recognition) {
        recognition = new SpeechRecognition();
        recognition.lang = "en-US";
        recognition.continuous = false;
        recognition.interimResults = false;
        
        recognition.onstart = () => {
            voiceBtn.classList.add("recording");
            voiceBtn.innerHTML = '<i data-lucide="mic-off" style="color: #ef4444;"></i>';
            if (window.lucide) lucide.createIcons();
        };
        
        recognition.onresult = (event) => {
            const transcript = event.results[0][0].transcript;
            promptField.value = `${promptField.value} ${transcript}`.trim();
        };
        
        recognition.onend = () => {
            voiceBtn.classList.remove("recording");
            voiceBtn.innerHTML = '<i data-lucide="mic"></i>';
            if (window.lucide) lucide.createIcons();
        };
        
        recognition.onerror = () => {
            pushMessage("bot", "Voice capture was interrupted or failed.");
            voiceBtn.classList.remove("recording");
            voiceBtn.innerHTML = '<i data-lucide="mic"></i>';
            if (window.lucide) lucide.createIcons();
        };
    }
    
    recognition.start();
});

// ============================================================================
// CLEAR CHAT
// ============================================================================

clearChatBtn?.addEventListener("click", () => {
    if (confirm("Clear all messages and start fresh?")) {
        clearChat();
    }
});

// ============================================================================
// INITIALIZATION
// ============================================================================

// Initialize on load
document.addEventListener("DOMContentLoaded", () => {
    updateThumbList();
    pushMessage("bot", createWelcomeMessage());
    
    // Check if coming from disease library
    const selectedDisease = sessionStorage.getItem("selectedDisease");
    if (selectedDisease) {
        sessionStorage.removeItem("selectedDisease");
        pushMessage("bot", `You selected <strong>${selectedDisease}</strong> from the disease library. Upload a leaf image to confirm diagnosis, or ask a question directly.`);
        
        // Pre-set the diagnosis for questions
        latestDiagnosis = { disease: selectedDisease };
        advisorPanel?.classList.remove("hidden");
        promptField?.focus();
    }
});
