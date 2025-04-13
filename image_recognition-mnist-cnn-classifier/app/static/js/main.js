document.addEventListener("DOMContentLoaded", function () {
  // --- Tab Switching ---
  const tabButtons = document.querySelectorAll('[data-bs-toggle="tab"]');

  tabButtons.forEach((button) => {
    button.addEventListener("shown.bs.tab", function (event) {
      // Clear results when switching tabs
      clearResults();
    });
  });

  // --- Canvas Drawing Setup ---
  const canvas = document.getElementById("canvas");
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
    const rect = canvas.getBoundingClientRect();
    if (e.type.includes("mouse")) {
      return {
        x: e.clientX - rect.left,
        y: e.clientY - rect.top,
      };
    } else {
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

  // Mouse event listeners
  canvas.addEventListener("mousedown", startDrawing);
  canvas.addEventListener("mousemove", draw);
  canvas.addEventListener("mouseup", stopDrawing);
  canvas.addEventListener("mouseout", stopDrawing);

  // Touch event listeners
  canvas.addEventListener("touchstart", function (e) {
    e.preventDefault();
    startDrawing(e);
  });

  canvas.addEventListener("touchmove", function (e) {
    e.preventDefault();
    draw(e);
  });

  canvas.addEventListener("touchend", stopDrawing);

  // --- Clear Canvas ---
  function clearCanvas() {
    initCanvas();
    document.getElementById("canvas-prediction-result").classList.add("d-none");
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

    if (!file) return;

    const reader = new FileReader();
    reader.onload = function (e) {
      imagePreview.src = e.target.result;
      previewContainer.classList.remove("d-none");
    };
    reader.readAsDataURL(file);
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
      .then((data) => displayCanvasPrediction(data))
      .catch((err) => {
        console.error("Error:", err);
        alert("Error processing the image. Please try again.");
      });
  });

  // --- Upload Form Prediction ---
  document
    .getElementById("upload-form")
    .addEventListener("submit", function (e) {
      e.preventDefault();
      const formData = new FormData(this);

      // Check if a file is selected
      if (!imageUpload.files[0]) {
        alert("Please select an image to upload.");
        return;
      }

      fetch("/predict", {
        method: "POST",
        body: formData,
      })
        .then((res) => res.json())
        .then((data) => displayUploadPrediction(data))
        .catch((err) => {
          console.error("Error:", err);
          alert("Error processing the uploaded image. Please try again.");
        });
    });

  // --- Display Canvas Prediction Results ---
  function displayCanvasPrediction(data) {
    const resultDiv = document.getElementById("canvas-prediction-result");
    const contentDiv = document.getElementById("canvas-prediction-content");

    if (data.error) {
      contentDiv.innerHTML = `<div class="alert alert-danger">${data.error}</div>`;
    } else {
      let html = `
        <div class="text-center mb-3">
          <h2 class="display-1">${data.digit}</h2>
          <p>Confidence: ${data.confidence.toFixed(2)}%</p>
          <div class="progress">
            <div class="progress-bar" role="progressbar" style="width: ${
              data.confidence
            }%" 
                 aria-valuenow="${
                   data.confidence
                 }" aria-valuemin="0" aria-valuemax="100"></div>
          </div>
        </div>
      `;

      // Handle different result formats
      if (data.alternatives && data.alternatives.length > 0) {
        html += `<p>Alternatives:</p><ul>`;
        data.alternatives.forEach((alt) => {
          html += `<li>Digit ${alt.digit} (${alt.probability.toFixed(
            2
          )}%)</li>`;
        });
        html += `</ul>`;
      }

      // Support for the augmented results format
      if (data.augmented && data.augmented.length > 0) {
        html += `<p>Augmented results:</p><ul>`;
        data.augmented.forEach((aug) => {
          html += `<li>Digit ${aug.digit} (${aug.confidence.toFixed(2)}%)</li>`;
        });
        html += `</ul>`;
      }

      // Support for consensus digit
      if (data.consensus !== undefined) {
        html += `<p>Consensus digit: <strong>${data.consensus}</strong></p>`;
      }

      contentDiv.innerHTML = html;
    }

    resultDiv.classList.remove("d-none");
  }

  // --- Display Upload Prediction Results ---
  function displayUploadPrediction(data) {
    const resultDiv = document.getElementById("upload-prediction-result");
    const contentDiv = document.getElementById("upload-prediction-content");

    if (data.error) {
      contentDiv.innerHTML = `<div class="alert alert-danger">${data.error}</div>`;
    } else {
      contentDiv.innerHTML = `
        <div class="text-center mb-3">
          <h2 class="display-1">${data.digit}</h2>
          <p>Confidence: ${data.confidence.toFixed(2)}%</p>
          <div class="progress">
            <div class="progress-bar" role="progressbar" style="width: ${
              data.confidence
            }%" 
                 aria-valuenow="${
                   data.confidence
                 }" aria-valuemin="0" aria-valuemax="100"></div>
          </div>
        </div>
      `;
    }

    resultDiv.classList.remove("d-none");
  }

  // --- Utility Functions ---
  function clearResults() {
    // Hide both result containers
    document.getElementById("canvas-prediction-result").classList.add("d-none");
    document.getElementById("upload-prediction-result").classList.add("d-none");

    // Clear the preview if in upload tab
    imageUpload.value = "";
    imagePreview.src = "";
    previewContainer.classList.add("d-none");
  }

  // --- Top Alternatives ---
  function getTopAlternatives(probabilities, mainPrediction, count = 3) {
    return probabilities
      .map((p, i) => ({ digit: i, probability: p }))
      .filter((item) => item.digit !== parseInt(mainPrediction))
      .sort((a, b) => b.probability - a.probability)
      .slice(0, count);
  }
});
