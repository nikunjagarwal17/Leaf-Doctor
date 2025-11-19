const chatStream = document.getElementById("chatStream");
const uploadButton = document.getElementById("uploadButton");
const uploadForm = document.getElementById("uploadForm");
const imageInput = document.getElementById("imageInput");
const filePill = document.getElementById("filePill");
const changeFileBtn = document.getElementById("changeFileBtn");
const adviceForm = document.getElementById("adviceForm");
const promptField = document.getElementById("promptField");
const askAdvisorBtn = document.getElementById("askAdvisorBtn");
const supplementBtn = document.getElementById("supplementBtn");

let latestDisease = null;
let advisorBusy = false;

function pushMessage(role, content) {
    const wrapper = document.createElement("div");
    wrapper.classList.add("message", role);
    if (typeof content === "string") {
        wrapper.innerHTML = content;
    } else {
        wrapper.appendChild(content);
    }
    chatStream.appendChild(wrapper);
    chatStream.scrollTop = chatStream.scrollHeight;
}

function formatPredictionCard(payload) {
    const card = document.createElement("div");
    card.innerHTML = `
        <strong>${payload.disease}</strong>
        <p>${payload.description}</p>
        ${payload.steps.length ? "<ul>" + payload.steps.map((step) => `<li>${step}</li>`).join("") + "</ul>" : ""}
        <p class="muted"><span class="muted">Source: Local CNN Model</span><br/>Use the advisor below for more detailed guidance on organic controls, chemical plans, or supply recommendations.</p>
    `;
    return card;
}

function handleFileLabel(file) {
    const label = file ? file.name : "No file selected";
    filePill.querySelector(".file-pill__label").textContent = label;
}

uploadButton?.addEventListener("click", () => imageInput?.click());
changeFileBtn?.addEventListener("click", () => imageInput?.click());

imageInput?.addEventListener("change", (event) => {
    const file = event.target.files?.[0];
    handleFileLabel(file);
});

uploadForm?.addEventListener("submit", async (event) => {
    event.preventDefault();
    const file = imageInput?.files?.[0];
    if (!file) {
        pushMessage("bot", "Please attach a leaf photo before asking for a diagnosis.");
        return;
    }

    const formData = new FormData();
    formData.append("image", file);

    uploadForm.querySelector("button.primary").disabled = true;
    pushMessage("user", `Uploaded ${file.name}`);
    pushMessage("bot", "Analyzing leaf image... hold tight 🌱");

    try {
        const response = await fetch("/predict", {
            method: "POST",
            body: formData,
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || "Prediction failed");
        }

        const payload = await response.json();
        latestDisease = payload.disease;
        pushMessage("bot", formatPredictionCard(payload));
        adviceForm?.classList.remove("hidden");
    } catch (error) {
        pushMessage("bot", `Something went wrong: ${error.message}`);
    } finally {
        uploadForm.querySelector("button.primary").disabled = false;
    }
});

adviceForm?.addEventListener("submit", async (event) => {
    event.preventDefault();
    const prompt = promptField.value.trim();
    if (!prompt) {
        return;
    }
    triggerAdvisor(prompt, prompt);
    promptField.value = "";
});

supplementBtn?.addEventListener("click", () => {
    if (!latestDisease) {
        pushMessage("bot", "Please run a diagnosis before requesting supplement ideas.");
        return;
    }
    const prompt = `Recommend 2-3 supplements (fertilizers, fungicides, or pesticides) to manage ${latestDisease}. Include organic and chemical options with application guidance.`;
    triggerAdvisor(prompt, "Get supplement suggestion");
});

function setAdvisorBusy(state) {
    advisorBusy = state;
    if (askAdvisorBtn) {
        askAdvisorBtn.disabled = state;
    }
    if (supplementBtn) {
        supplementBtn.disabled = state;
    }
}

async function triggerAdvisor(prompt, displayText) {
    if (advisorBusy) {
        return;
    }
    if (!latestDisease) {
        pushMessage("bot", "Please run a diagnosis before opening the advisor.");
        return;
    }
    setAdvisorBusy(true);
    pushMessage("user", displayText);

    try {
        const response = await fetch("/advisor", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ prompt, disease: latestDisease }),
        });
        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.error || "Advisor request failed");
        }
        pushMessage("bot", `${data.message} <br/><span class="muted">Source: ${data.source}</span>`);
    } catch (error) {
        pushMessage("bot", `Advisor error: ${error.message}`);
    } finally {
        setAdvisorBusy(false);
    }
}

pushMessage("bot", "Hello 👋 Please upload a leaf image to know about the plant disease.");

