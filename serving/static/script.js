document.addEventListener('DOMContentLoaded', () => {
    
    // Sidebar view switching
    const navItems = document.querySelectorAll('.sidebar-nav .nav-item');
    const viewSections = document.querySelectorAll('.view-section');

    function switchView(targetTab) {
        // Handle Sidebar active state
        navItems.forEach(i => i.classList.remove('active'));
        const activeNav = Array.from(navItems).find(n => n.getAttribute('data-tab') === targetTab);
        if (activeNav) activeNav.classList.add('active');

        // Handle Main Panel active state
        viewSections.forEach(v => {
            v.style.display = 'none';
        });
        
        let viewId = targetTab;
        if (!document.getElementById(viewId)) {
            // Default fallback if a section doesn't exist yet
            viewId = 'view-dashboard';
        }
        document.getElementById(viewId).style.display = 'block';
    }

    navItems.forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            const target = item.getAttribute('data-tab') || 'view-dashboard';
            switchView(target);
        });
    });
    
    // Initialize view
    switchView('view-form');

    // Number Animation Logic
    const stats = {
        total: parseInt(document.getElementById('statTotal').innerText) || 154430,
        alerts: parseInt(document.getElementById('statAlerts').innerText) || 6480,
        approvals: parseInt(document.getElementById('statApprovals').innerText) || 5320,
        auc: parseInt(document.getElementById('statAUC').innerText) || 88
    };

    function animateValue(obj, start, end, duration, formatPrefix = "", formatSuffix = "") {
        let startTimestamp = null;
        const step = (timestamp) => {
            if (!startTimestamp) startTimestamp = timestamp;
            const progress = Math.min((timestamp - startTimestamp) / duration, 1);
            const current = Math.floor(progress * (end - start) + start);
            obj.innerHTML = formatPrefix + current.toLocaleString() + formatSuffix;
            if (progress < 1) {
                window.requestAnimationFrame(step);
            } else {
                obj.innerHTML = formatPrefix + end.toLocaleString() + formatSuffix;
            }
        };
        window.requestAnimationFrame(step);
    }

    // Animate summary metrics on load
    animateValue(document.getElementById('statTotal'), 0, stats.total, 1500, "");
    animateValue(document.getElementById('statAlerts'), 0, stats.alerts, 1500);
    animateValue(document.getElementById('statApprovals'), 0, stats.approvals, 1500);
    animateValue(document.getElementById('statAUC'), 0, stats.auc, 1500, "", "%");

    // Form logic
    const form = document.getElementById('predictionForm');
    const resultOutput = document.getElementById('resultOutput');
    const placeholderOutput = document.getElementById('placeholderOutput');
    
    const riskClassEl = document.getElementById('riskClass');
    const probabilityValueEl = document.getElementById('probabilityValue');
    const errorTextEl = document.getElementById('errorText');
    const submitBtn = document.getElementById('submitBtn');
    const shapListContainer = document.getElementById('shapList');

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        // UI states
        const originalBtnText = submitBtn.textContent;
        submitBtn.textContent = 'Analyzing...';
        submitBtn.disabled = true;

        const formData = new FormData(form);
        const record = {};
        for (let [key, value] of formData.entries()) {
            if (value !== '' && !isNaN(value)) {
                record[key] = Number(value);
            } else {
                record[key] = value;
            }
        }
        
        // Map UI specifics to Model needs
        if (record['stability']) {
            record['income_stability'] = record['stability'] === 'Stable' ? 1.0 : 0.0;
            delete record['stability'];
        }

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ data: [record] })
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Prediction failed');
            }

            const resultData = await response.json();
            
            if (resultData.predictions && resultData.predictions.length > 0) {
                const pred = resultData.predictions[0];
                showResult(pred);
                
                // --- Dynamic Stats Update ---
                
                // Increment Total
                stats.total += 1;
                document.getElementById('statTotal').innerText = stats.total.toLocaleString();
                
                // Increment Alert/Approval
                if (pred.probability > 0.5) {
                    stats.alerts += 1;
                    document.getElementById('statAlerts').innerText = stats.alerts.toLocaleString();
                    animateValue(document.getElementById('statAlerts'), stats.alerts - 1, stats.alerts, 300);
                } else {
                    stats.approvals += 1;
                    document.getElementById('statApprovals').innerText = stats.approvals.toLocaleString();
                    animateValue(document.getElementById('statApprovals'), stats.approvals - 1, stats.approvals, 300);
                }
                
                // Quick flash animation on Total
                const totalEl = document.getElementById('statTotal');
                totalEl.style.color = '#1890ff';
                setTimeout(() => totalEl.style.color = '', 500);
            }

        } catch (error) {
            showError(error.message);
        } finally {
            submitBtn.textContent = originalBtnText;
            submitBtn.disabled = false;
        }
    });

    function showResult(prediction) {
        placeholderOutput.style.display = 'none';
        resultOutput.style.display = 'block';
        errorTextEl.classList.add('hidden');
        
        // Basics
        const probPercentage = (prediction.probability * 100).toFixed(1);
        probabilityValueEl.textContent = `${probPercentage}%`;
        
        riskClassEl.textContent = prediction.risk_class;
        riskClassEl.className = 'risk-badge'; // reset
        
        if (prediction.probability > 0.5) {
            riskClassEl.classList.add('high');
        } else {
            riskClassEl.classList.add('low');
        }

        // Render Ranking list from SHAP
        renderRankingList(prediction.shap_values);
    }

    function renderRankingList(shapValues) {
        shapListContainer.innerHTML = '';
        if (!shapValues || shapValues.length === 0) return;

        // Get Top 7 features for the ranking list (like the image)
        const topFeatures = shapValues.slice(0, 7);
        
        topFeatures.forEach((feature, index) => {
            const isPos = feature.value > 0;
            const rankClass = index < 3 ? 'top-3' : '';
            const valClass = isPos ? 'pos' : 'neg';
            const sign = isPos ? '+' : '';
            
            const labelStr = feature.feature.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
            const displayVal = sign + feature.value.toFixed(3);

            const item = document.createElement('div');
            item.className = `rank-item ${rankClass}`;
            item.innerHTML = `
                <div class="rank-index">${index + 1}</div>
                <div class="rank-name" title="${labelStr}">${labelStr}</div>
                <div class="rank-value ${valClass}">${displayVal}</div>
            `;
            shapListContainer.appendChild(item);
        });
    }

    function showError(message) {
        placeholderOutput.style.display = 'block';
        resultOutput.style.display = 'none';
        
        errorTextEl.textContent = `Error: ${message}`;
        errorTextEl.classList.remove('hidden');
    }
});
