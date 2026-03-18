"""Dashboard HTML simple pour tester les prédictions et voir les métriques."""

import json
import logging
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>WhoAmAI Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #f5f5f5; color: #333; }
        .header { background: linear-gradient(135deg, #6200EE, #3700B3); color: white; padding: 24px; text-align: center; }
        .header h1 { font-size: 28px; margin-bottom: 8px; }
        .header p { opacity: 0.9; }
        .container { max-width: 1200px; margin: 0 auto; padding: 24px; }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }
        .card { background: white; border-radius: 12px; padding: 24px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
        .card h2 { font-size: 18px; margin-bottom: 16px; color: #6200EE; }
        .upload-zone { border: 2px dashed #ccc; border-radius: 8px; padding: 40px; text-align: center; cursor: pointer; transition: border-color 0.3s; }
        .upload-zone:hover { border-color: #6200EE; }
        .upload-zone.dragover { border-color: #6200EE; background: #f0e6ff; }
        input[type="file"] { display: none; }
        .btn { background: #6200EE; color: white; border: none; padding: 12px 24px; border-radius: 8px; cursor: pointer; font-size: 16px; }
        .btn:hover { background: #3700B3; }
        .btn:disabled { background: #ccc; cursor: not-allowed; }
        .results { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-top: 24px; }
        .result-card { background: #f8f8f8; border-radius: 8px; padding: 16px; }
        .result-card h3 { font-size: 14px; color: #666; margin-bottom: 12px; text-transform: uppercase; }
        .result-item { display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #eee; }
        .result-item:last-child { border-bottom: none; }
        .confidence { font-size: 12px; color: #999; }
        .metric-table { width: 100%; border-collapse: collapse; }
        .metric-table th, .metric-table td { padding: 10px 12px; text-align: left; border-bottom: 1px solid #eee; }
        .metric-table th { background: #f8f8f8; font-weight: 600; color: #666; }
        .badge { display: inline-block; padding: 4px 8px; border-radius: 4px; font-size: 12px; font-weight: 600; }
        .badge-demo { background: #FFF3E0; color: #E65100; }
        .badge-prod { background: #E8F5E9; color: #2E7D32; }
        .spinner { display: none; width: 40px; height: 40px; border: 4px solid #f3f3f3; border-top: 4px solid #6200EE; border-radius: 50%; animation: spin 1s linear infinite; margin: 20px auto; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        .preview-img { max-width: 100%; max-height: 300px; border-radius: 8px; margin: 12px 0; }
        .status-dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; margin-right: 8px; }
        .status-dot.green { background: #4CAF50; }
        .status-dot.orange { background: #FF9800; }
        #error-msg { color: #B00020; margin-top: 12px; display: none; }
        @media (max-width: 768px) { .grid { grid-template-columns: 1fr; } .results { grid-template-columns: 1fr; } }
    </style>
</head>
<body>
    <div class="header">
        <h1>WhoAmAI Dashboard</h1>
        <p>Test de prediction faciale - 3 strategies</p>
    </div>
    <div class="container">
        <div class="grid">
            <div class="card">
                <h2>Upload Image</h2>
                <div class="upload-zone" id="dropZone" onclick="document.getElementById('fileInput').click()">
                    <p>Glissez une image ici ou cliquez pour selectionner</p>
                    <input type="file" id="fileInput" accept="image/*">
                </div>
                <img id="preview" class="preview-img" style="display:none">
                <div style="margin-top: 16px; text-align: center;">
                    <button class="btn" id="analyzeBtn" onclick="analyzeAll()" disabled>Analyser avec les 3 strategies</button>
                </div>
                <div class="spinner" id="spinner"></div>
                <p id="error-msg"></p>
            </div>
            <div class="card">
                <h2>Status API</h2>
                <div id="healthStatus">Chargement...</div>
                <h2 style="margin-top: 24px;">Metriques</h2>
                <table class="metric-table" id="metricsTable">
                    <thead><tr><th>Strategie</th><th>Age MAE</th><th>Gender Acc</th><th>Ethnicity Acc</th></tr></thead>
                    <tbody id="metricsBody"><tr><td colspan="4">Chargement...</td></tr></tbody>
                </table>
            </div>
        </div>
        <div class="results" id="results" style="display:none;"></div>
    </div>

    <script>
        let selectedFile = null;
        const strategies = ['specialized', 'multitask', 'transfer'];

        // File handling
        const dropZone = document.getElementById('dropZone');
        const fileInput = document.getElementById('fileInput');

        dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('dragover'); });
        dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
        dropZone.addEventListener('drop', (e) => {
            e.preventDefault(); dropZone.classList.remove('dragover');
            if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
        });
        fileInput.addEventListener('change', (e) => { if (e.target.files.length) handleFile(e.target.files[0]); });

        function handleFile(file) {
            selectedFile = file;
            const reader = new FileReader();
            reader.onload = (e) => {
                document.getElementById('preview').src = e.target.result;
                document.getElementById('preview').style.display = 'block';
            };
            reader.readAsDataURL(file);
            document.getElementById('analyzeBtn').disabled = false;
        }

        async function analyzeAll() {
            if (!selectedFile) return;
            const spinner = document.getElementById('spinner');
            const results = document.getElementById('results');
            const errorMsg = document.getElementById('error-msg');
            spinner.style.display = 'block';
            results.innerHTML = '';
            results.style.display = 'none';
            errorMsg.style.display = 'none';

            const cards = [];
            for (const strategy of strategies) {
                try {
                    const formData = new FormData();
                    formData.append('file', selectedFile);
                    const resp = await fetch('/predict/' + strategy, { method: 'POST', body: formData });
                    const data = await resp.json();

                    if (data.success) {
                        const d = data.data;
                        const m = data.metadata;
                        cards.push(`
                            <div class="result-card">
                                <h3>${strategy} ${m.demo_mode ? '<span class="badge badge-demo">DEMO</span>' : '<span class="badge badge-prod">PROD</span>'}</h3>
                                <div class="result-item"><span>Age</span><span>${d.age.toFixed(1)} ans</span></div>
                                <div class="result-item"><span>Genre</span><span>${d.gender} <span class="confidence">(${(d.gender_confidence*100).toFixed(1)}%)</span></span></div>
                                <div class="result-item"><span>Ethnicite</span><span>${d.ethnicity} <span class="confidence">(${(d.ethnicity_confidence*100).toFixed(1)}%)</span></span></div>
                                <div class="result-item"><span>Inference</span><span>${m.inference_time_ms.toFixed(1)} ms</span></div>
                            </div>
                        `);
                    } else {
                        cards.push(`<div class="result-card"><h3>${strategy}</h3><p style="color:#B00020">${data.error.code}: ${data.error.message}</p></div>`);
                    }
                } catch (e) {
                    cards.push(`<div class="result-card"><h3>${strategy}</h3><p style="color:#B00020">Erreur: ${e.message}</p></div>`);
                }
            }

            spinner.style.display = 'none';
            results.innerHTML = cards.join('');
            results.style.display = 'grid';
        }

        // Load health and metrics on page load
        async function loadStatus() {
            try {
                const resp = await fetch('/health');
                const data = await resp.json();
                const dot = data.demo_mode ? 'orange' : 'green';
                const badge = data.demo_mode ? '<span class="badge badge-demo">DEMO</span>' : '<span class="badge badge-prod">PRODUCTION</span>';
                document.getElementById('healthStatus').innerHTML = `
                    <p><span class="status-dot ${dot}"></span>Status: ${data.status} ${badge}</p>
                    <p>Version: ${data.version}</p>
                    <p>Modeles charges: ${data.models_loaded}</p>
                    <p>Uptime: ${data.uptime.toFixed(0)}s</p>
                `;
            } catch (e) {
                document.getElementById('healthStatus').innerHTML = '<p style="color:#B00020">API non disponible</p>';
            }

            try {
                const resp = await fetch('/models/info');
                const data = await resp.json();
                let rows = '';
                for (const m of data.metrics) {
                    rows += `<tr>
                        <td>${m.strategy}</td>
                        <td>${m.age_mae != null ? m.age_mae.toFixed(2) : 'N/A'}</td>
                        <td>${m.gender_accuracy != null ? (m.gender_accuracy * 100).toFixed(1) + '%' : 'N/A'}</td>
                        <td>${m.ethnicity_accuracy != null ? (m.ethnicity_accuracy * 100).toFixed(1) + '%' : 'N/A'}</td>
                    </tr>`;
                }
                document.getElementById('metricsBody').innerHTML = rows || '<tr><td colspan="4">Aucune metrique</td></tr>';
            } catch (e) {
                document.getElementById('metricsBody').innerHTML = '<tr><td colspan="4">Erreur</td></tr>';
            }
        }

        loadStatus();
    </script>
</body>
</html>
"""


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    return DASHBOARD_HTML
