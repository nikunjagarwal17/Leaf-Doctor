const chatStream = document.getElementById("chatStream");
const imageInput = document.getElementById("imageInput");
const chooseImagesBtn = document.getElementById("chooseImagesBtn") || document.getElementById("addImagesBtn");
const addImagesBtn = document.getElementById("addImagesBtn");
const thumbnails = document.getElementById("thumbList");
const predictBtn = document.getElementById("predictBtn");
const noteField = document.getElementById("noteField");
const promptField = document.getElementById("promptField");
const sendBtn = document.getElementById("sendBtn");
const thinkingToggle = document.getElementById("thinkingToggle");
const voiceBtn = document.getElementById("voiceBtn");
const speakToggle = document.getElementById("speakToggle");
const advisorPanel = document.getElementById("advisorPanel");

let latestDiagnosis = null;
let advisorBusy = false;
let recognition = null;

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

function renderThinking(reasoning, context, reasoningChain) {
    if (!reasoning && !context?.length && !reasoningChain?.length) return null;
    const details = document.createElement("details");
    details.classList.add("thinking");
    if (thinkingToggle?.checked) {
        details.setAttribute("open", "open");
    }
    const summary = document.createElement("summary");
    summary.textContent = reasoningChain?.length ? "Show Agent Reasoning" : "Show thinking";
    details.appendChild(summary);

    const block = document.createElement("div");
    
    
    if (reasoningChain?.length) {
        const chainDiv = document.createElement("div");
        chainDiv.classList.add("reasoning-chain");
        
        const chainTitle = document.createElement("p");
        chainTitle.classList.add("chain-title");
        chainTitle.innerHTML = "<strong>Agent Execution Plan:</strong>";
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
    
    if (reasoning) {
        const para = document.createElement("p");
        para.classList.add("muted");
        para.textContent = reasoning;
        block.appendChild(para);
    }
    if (context?.length) {
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

function renderPredictionCard(payload) {
    const card = document.createElement("div");
    card.classList.add("prediction-card");
    
    
    const agenticBadge = payload.agentic 
        ? `<span class="agentic-badge">Agentic</span>` 
        : "";
    
    
    const qualityWarning = payload.quality_check && !payload.quality_check.passed
        ? `<div class="quality-warning">${payload.quality_check.issues?.join(", ") || "Low confidence detected"}</div>`
        : "";
    
    card.innerHTML = `
        <div class="prediction-head">
            <div>
                <p class="eyebrow">Detected disease ${agenticBadge}</p>
                <strong>${payload.disease}</strong>
            </div>
            <div class="pill">Confidence ${(payload.confidence * 100).toFixed(1)}%</div>
        </div>
        ${qualityWarning}
        <p>${payload.description}</p>
        ${
            payload.steps.length
                ? "<ul>" + payload.steps.map((step) => `<li>${step}</li>`).join("") + "</ul>"
                : ""
        }
        <p class="muted">Source: ${payload.agentic ? "Agentic CNN orchestration" : "Local CNN ensemble"} across ${payload.image_count || 1} image(s).</p>
    `;

    if (payload.per_image?.length) {
        const grid = document.createElement("div");
        grid.classList.add("mini-grid");
        payload.per_image.forEach((p) => {
            const chip = document.createElement("div");
            chip.classList.add("mini-chip");
            chip.innerHTML = `<strong>${p.disease}</strong><span>${(p.confidence * 100).toFixed(
                1
            )}% • ${p.file}</span>`;
            grid.appendChild(chip);
        });
        card.appendChild(grid);
    }

    if (payload.alternatives?.length) {
        const alt = document.createElement("p");
        alt.classList.add("muted");
        alt.textContent = `Runner-up: ${payload.alternatives
            .map((a) => `${a.disease} ${(a.probability * 100).toFixed(1)}%`)
            .join(" | ")}`;
        card.appendChild(alt);
    }

    if (payload.note_context?.length) {
        const ctx = document.createElement("div");
        ctx.classList.add("thinking");
        const summary = document.createElement("summary");
        summary.textContent = "Text clues matches";
        const details = document.createElement("details");
        details.setAttribute("open", "open");
        details.appendChild(summary);
        const list = document.createElement("ul");
        list.classList.add("context-list");
        payload.note_context.forEach((item) => {
            const li = document.createElement("li");
            li.textContent = `${item.name} (relevance ${item.score.toFixed(2)})`;
            list.appendChild(li);
        });
        details.appendChild(list);
        ctx.appendChild(details);
        card.appendChild(ctx);
    }
    
    
    if (payload.reasoning_chain?.length) {
        const thinkingNode = renderThinking(null, null, payload.reasoning_chain);
        if (thinkingNode) card.appendChild(thinkingNode);
    }
    
    return card;
}

function updateThumbList(files) {
    thumbnails.innerHTML = "";
    if (!files?.length) {
        thumbnails.innerHTML = `<li class="muted">No images selected</li>`;
        return;
    }
    [...files].forEach((file) => {
        const li = document.createElement("li");
        li.textContent = file.name;
        thumbnails.appendChild(li);
    });
}

imageInput?.addEventListener("change", (event) => {
    updateThumbList(event.target.files);
});

chooseImagesBtn?.addEventListener("click", () => imageInput?.click());
addImagesBtn?.addEventListener("click", () => imageInput?.click());

predictBtn?.addEventListener("click", async (event) => {
    event.preventDefault();
    const files = imageInput?.files;
    if (!files?.length) {
        pushMessage("bot", "Attach at least one leaf image to run a diagnosis.");
        return;
    }
    const formData = new FormData();
    [...files].forEach((file) => formData.append("images", file));
    if (noteField?.value.trim()) {
        formData.append("note", noteField.value.trim());
    }

    predictBtn.disabled = true;
    pushMessage("user", `Diagnose using ${files.length} image(s).`);
    pushMessage("bot", "Analyzing uploaded images...");

    try {
        const response = await fetch("/predict", { method: "POST", body: formData });
        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.error || "Prediction failed");
        }
        latestDiagnosis = data;
        pushMessage("bot", renderPredictionCard(data));
        advisorPanel?.classList.remove("hidden");
    } catch (error) {
        pushMessage("bot", `Prediction error: ${error.message}`);
    } finally {
        predictBtn.disabled = false;
    }
});

sendBtn?.addEventListener("click", (event) => {
    event.preventDefault();
    const prompt = promptField.value.trim();
    if (!prompt) return;
    triggerAdvisor(prompt);
    promptField.value = "";
});

async function triggerAdvisor(prompt) {
    if (advisorBusy) return;
    if (!latestDiagnosis?.disease) {
        pushMessage("bot", "Run a diagnosis first so I can ground the advice.");
        return;
    }
    advisorBusy = true;
    sendBtn.disabled = true;
    pushMessage("user", prompt);

    try {
        const response = await fetch("/advisor", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ prompt, disease: latestDiagnosis.disease, agentic: true }),
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || "Advisor request failed");

        const block = document.createElement("div");
        
        // Add agentic indicator
        if (data.agentic) {
            const agenticInfo = document.createElement("div");
            agenticInfo.classList.add("agentic-info");
            agenticInfo.innerHTML = `<span class="agentic-badge">Agentic</span> ${data.agent_goal || ""}`;
            block.appendChild(agenticInfo);
        }
        
        const answer = document.createElement("div");
        answer.innerHTML = `${data.message}<br/><span class="muted">Source: ${data.source}</span>`;
        block.appendChild(answer);
        
        // Add quality indicators if present
        if (data.retrieval_quality && !data.retrieval_quality.passed) {
            const qualityNote = document.createElement("p");
            qualityNote.classList.add("quality-warning");
            qualityNote.innerHTML = `⚠️ Retrieval: ${data.retrieval_quality.issues?.join(", ") || "Low relevance"}`;
            block.appendChild(qualityNote);
        }
        
        // Render thinking with reasoning chain
        const thinkingNode = renderThinking(data.reasoning, data.context, data.reasoning_chain);
        if (thinkingNode) block.appendChild(thinkingNode);
        pushMessage("bot", block);
    } catch (error) {
        pushMessage("bot", `Advisor error: ${error.message}`);
    } finally {
        advisorBusy = false;
        sendBtn.disabled = false;
    }
}

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
        recognition.onresult = (event) => {
            const transcript = event.results[0][0].transcript;
            promptField.value = `${promptField.value} ${transcript}`.trim();
        };
        recognition.onerror = () => {
            pushMessage("bot", "Voice capture was interrupted.");
        };
    }
    recognition.start();
});

// Render empty state for attachments on load
updateThumbList(imageInput?.files || []);

// Starter message
pushMessage(
    "bot",
    "Welcome. Upload up to four leaf images to improve diagnosis confidence, then ask follow-up questions grounded in the embedded disease library."
);
