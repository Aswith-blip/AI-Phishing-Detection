const API_URL = "http://127.0.0.1:5000/api/predict";

const urlElement = document.getElementById("url");
const scanButton = document.getElementById("scanButton");
const loading = document.getElementById("loading");
const result = document.getElementById("result");
const errorBox = document.getElementById("error");

let currentUrl = "";

async function getCurrentUrl() {
    const tabs = await chrome.tabs.query({
        active: true,
        currentWindow: true
    });

    if (!tabs.length || !tabs[0].url) {
        throw new Error("Unable to get the current website URL.");
    }

    currentUrl = tabs[0].url;
    urlElement.textContent = currentUrl;
}

async function scanWebsite() {

    if (!currentUrl) {
        showError("No website URL detected.");
        return;
    }

    loading.classList.remove("hidden");
    result.classList.add("hidden");
    errorBox.classList.add("hidden");

    try {

        const response = await fetch(API_URL, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                url: currentUrl
            })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || "Prediction failed.");
        }

        showResult(data);

    } catch (error) {

        showError(
            "Could not connect to the AI server. Make sure Flask is running."
        );

    } finally {

        loading.classList.add("hidden");
    }
}

function showResult(data) {

    result.classList.remove("hidden");

    const status = document.getElementById("resultStatus");

    if (data.prediction === "PHISHING") {

        status.textContent = "⚠️ PHISHING DETECTED";
        status.className = "danger";

    } else {

        status.textContent = "🛡️ LEGITIMATE";
        status.className = "safe";
    }

    document.getElementById("phishingProbability").textContent =
        data.phishing_probability + "%";

    document.getElementById("legitimateProbability").textContent =
        data.legitimate_probability + "%";

    document.getElementById("confidence").textContent =
        data.confidence;

    document.getElementById("riskLevel").textContent =
        data.risk_level;

    document.getElementById("riskScore").textContent =
        data.risk_score + "/100";

    document.getElementById("riskProgress").style.width =
        data.risk_score + "%";

    const reasonsList = document.getElementById("reasonsList");

    reasonsList.innerHTML = "";

    if (data.reasons && data.reasons.length) {

        data.reasons.forEach(reason => {

            const li = document.createElement("li");
            li.textContent = "⚠️ " + reason;
            reasonsList.appendChild(li);

        });

    } else {

        const li = document.createElement("li");
        li.textContent = "No major suspicious indicators detected.";
        reasonsList.appendChild(li);
    }
}

function showError(message) {

    errorBox.textContent = message;
    errorBox.classList.remove("hidden");
}

getCurrentUrl().catch(error => {
    showError(error.message);
});

scanButton.addEventListener("click", scanWebsite);