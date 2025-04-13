document.addEventListener("DOMContentLoaded", function () {
  // --- Tab Switching ---
  const tabButtons = document.querySelectorAll(".tab-btn");
  const tabContents = document.querySelectorAll(".tab-content");

  tabButtons.forEach((button) => {
    button.addEventListener("click", () => {
      tabButtons.forEach((btn) => btn.classList.remove("active"));
      tabContents.forEach((content) => content.classList.remove("active"));

      button.classList.add("active");
      const tabId = `${button.dataset.tab}-tab`;
      document.getElementById(tabId).classList.add("active");
    });
  });

  // --- Canvas Drawing Setup ---
  const canvas = document.getElementById("drawing-canvas");
  const ctx = canvas.getContext("2d");
  let isDrawing = false;

  function initCanvas() {
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = "black";
    ctx.lineWidth = 15;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
  }

  initCanvas();

  function getCoords(e) {
    if (e.type.includes("mouse")) {
      return { x: e.offsetX, y: e.offsetY };
    } else {
      const rect = canvas.getBoundingClientRect();
      return {
        x: e.touches[0].clientX - rect.left,
        y: e.touches[0].clientY - rect.top,
      };
    }
  }

  function startDrawing(e) {
    isDrawing = true;
    const { x, y } = getCoords(e);
    ctx.beginPath();
    ctx.moveTo(x, y);
  }

  function draw(e) {
    if (!isDrawing) return;
    const { x, y } = getCoords(e);
    ctx.lineTo(x, y);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x, y);
  }

  function stopDrawing() {
    isDrawing = false;
    ctx.beginPath();
  }

  canvas.addEventListener("mousedown", startDrawing);
  canvas.addEventListener("mousemove", draw);
  canvas.addEventListener("mouseup", stopDrawing);
  canvas.addEventListener("mouseout", stopDrawing);
  canvas.addEventListener("touchstart", (e) => {
    e.preventDefault();
    startDrawing(e);
  });
  canvas.addEventListener("touchmove", (e) => {
    e.preventDefault();
    draw(e);
  });
  canvas.addEventListener("touchend", stopDrawing);

  // --- Clear Canvas ---
  function clearCanvas() {
    initCanvas();
    document.getElementById("result-container").classList.add("hidden");
  }

  document
    .getElementById("clear-canvas")
    .addEventListener("click", clearCanvas);

  // --- Image Upload + Preview ---
  const imageUpload = document.getElementById("image-upload");
  const imagePreview = document.getElementById("image-preview");
  const previewContainer = document.getElementById("preview-container");

  imageUpload.addEventListener("change", function (event) {
    const file = event.target.files[0];

    imagePreview.src = "";
    previewContainer.classList.add("hidden");

    if (file && file.type.startsWith("image/")) {
      const reader = new FileReader();
      reader.onload = function (e) {
        imagePreview.src = e.target.result;
        previewContainer.classList.remove("hidden");
      };
      reader.readAsDataURL(file);
    }
  });

  // --- Predict from Canvas ---
  document.getElementById("predict-canvas").addEventListener("click", () => {
    const imageData = canvas.toDataURL("image/png");
    fetch("/predict_canvas", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image_data: imageData }),
    })
      .then((res) => res.json())
      .then((data) => displayResults(data))
      .catch((err) => console.error("Error:", err));
  });

  // --- Upload Form Prediction ---
  document
    .getElementById("upload-form")
    .addEventListener("submit", function (e) {
      e.preventDefault();
      const formData = new FormData(this);

      fetch("/predict", {
        method: "POST",
        body: formData,
      })
        .then((res) => res.json())
        .then((data) => displayResults(data))
        .catch((err) => console.error("Error:", err));
    });

  // --- Try Again Button ---
  document.getElementById("try-again").addEventListener("click", function () {
    clearCanvas();
    imageUpload.value = "";
    imagePreview.src = "";
    previewContainer.classList.add("hidden");

    document.getElementById("result-container").classList.add("hidden");
    document.querySelector(".confidence-bar").style.width = "0%";
    document.querySelector(".confidence-text").textContent = "Confidence: 0%";
    document.querySelector(".prediction-digit").textContent = "?";
    document.querySelector(".alternatives-container").innerHTML = "";
  });

  // --- Display Results ---
  function displayResults(data) {
    const resultContainer = document.getElementById("result-container");
    const predictionDigit = document.querySelector(".prediction-digit");
    const confidenceBar = document.querySelector(".confidence-bar");
    const confidenceText = document.querySelector(".confidence-text");
    const alternativesContainer = document.querySelector(
      ".alternatives-container"
    );

    predictionDigit.textContent = data.digit;
    confidenceBar.style.width = `${data.confidence}%`;
    confidenceText.textContent = `Confidence: ${data.confidence.toFixed(2)}%`;

    alternativesContainer.innerHTML = "";
    const alternatives = getTopAlternatives(data.probabilities, data.digit);
    alternatives.forEach((alt) => {
      const altElement = document.createElement("div");
      altElement.classList.add("alternative");
      altElement.innerHTML = `
          <div class="alt-digit">${alt.digit}</div>
          <div class="alt-probability">${alt.probability.toFixed(2)}%</div>
        `;
      alternativesContainer.appendChild(altElement);
    });

    resultContainer.classList.remove("hidden");
  }

  // --- Top Alternatives ---
  function getTopAlternatives(probabilities, mainPrediction, count = 3) {
    return probabilities
      .map((p, i) => ({ digit: i, probability: p * 100 }))
      .filter((item) => item.digit !== mainPrediction)
      .sort((a, b) => b.probability - a.probability)
      .slice(0, count);
  }
});
