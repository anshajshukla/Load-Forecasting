/**
 * Delhi Load Forecast Dashboard - JavaScript
 * Handles data fetching, chart creation, and dynamic updates
 */

// Chart color scheme
const colors = {
    DELHI: '#4e73df',
    BRPL: '#1cc88a',
    BYPL: '#36b9cc',
    NDMC: '#f6c23e',
    MES: '#e74a3b',
    temperature: '#fd7e14',
    humidity: '#6610f2',
    windSpeed: '#6f42c1',
    precipitation: '#20c9a6'
};

// Chart objects
let currentLoadChart = null;
let forecastChart = null;
let historicalChart = null;
let weatherChart = null;

// Data objects
let currentData = null;
let forecastData = null;

// DOM elements
const refreshBtn = document.getElementById('refreshBtn');
const lastUpdateTime = document.getElementById('lastUpdateTime');
const updateStatus = document.getElementById('updateStatus');
const currentLoadTable = document.getElementById('currentLoadTable').querySelector('tbody');

/**
 * Initialize the dashboard
 */
function initDashboard() {
    // Load model metrics table and multi-model forecast
    loadModelMetricsTable();
    loadMultiModelForecastChart();
    // Load data immediately
    loadAllData();

    // Set up refresh button
    refreshBtn.addEventListener('click', function(e) {
        e.preventDefault();
        triggerUpdate();
    });

    // Periodic status and data refresh
    setInterval(checkStatus, 30000); // every 30s
    setInterval(loadAllData, 300000); // every 5 min
}

/**
 * Load and render the multi-model metrics table
 */
function loadModelMetricsTable() {
    fetch('/api/model_metrics')
        .then(response => response.json())
        .then(data => {
            if (!data || data.error) {
                document.getElementById('multiModelMetricsTable').innerHTML = '<div class="text-danger">No metrics available.</div>';
                return;
            }
            // Build table
            let regions = [];
            let metrics = ['RMSE', 'MAPE', 'R2'];
            let models = Object.keys(data);
            // Find all regions from the first model that has metrics
            for (let m of models) {
                if (data[m].metrics) {
                    regions = Object.keys(data[m].metrics);
                    break;
                }
            }
            let html = '<table class="table table-bordered table-striped table-sm"><thead><tr><th>Model</th>';
            for (let region of regions) {
                for (let metric of metrics) {
                    html += `<th>${region} ${metric}</th>`;
                }
            }
            html += '<th>Avg MAPE</th></tr></thead><tbody>';
            for (let m of models) {
                html += `<tr><td>${m.toUpperCase()}</td>`;
                if (data[m].metrics) {
                    for (let region of regions) {
                        for (let metric of metrics) {
                            let val = data[m].metrics[region][metric];
                            if (metric === 'MAPE') val = val?.toFixed(2) + '%';
                            else if (typeof val === 'number') val = val?.toFixed(2);
                            html += `<td>${val ?? '-'}</td>`;
                        }
                    }
                    html += `<td>${data[m].average_mape?.toFixed(2) ?? '-'}%</td>`;
                } else {
                    html += `<td colspan="${regions.length*metrics.length+1}" class="text-danger">${data[m].error ?? 'Error'}</td>`;
                }
                html += '</tr>';
            }
            html += '</tbody></table>';
            document.getElementById('multiModelMetricsTable').innerHTML = html;
        })
        .catch(err => {
            document.getElementById('multiModelMetricsTable').innerHTML = '<div class="text-danger">Error loading metrics.</div>';
        });
}

/**
 * Load and render the multi-model forecast chart
 */
function loadMultiModelForecastChart() {
    fetch('/api/model_predictions')
        .then(response => response.json())
        .then(data => {
            if (!data || data.error) return;
            // We'll plot the first region (e.g. DELHI) for all models
            let models = Object.keys(data);
            let region = 'DELHI'; // Default, or could allow user to pick
            let datasets = [];
            let labels = [];
            // Use actuals from the first model
            for (let m of models) {
                if (data[m].actual && data[m].targets && data[m].targets.includes(region)) {
                    let idx = data[m].targets.indexOf(region);
                    labels = Array.from({length: data[m].actual.length}, (_, i) => i+1);
                    datasets.push({
                        label: 'Actual',
                        data: data[m].actual.map(row => row[idx]),
                        borderColor: '#000',
                        backgroundColor: 'rgba(0,0,0,0.1)',
                        borderWidth: 2,
                        pointRadius: 1,
                        fill: false,
                        tension: 0.1,
                        order: 0
                    });
                    break;
                }
            }
            // Add each model's prediction
            const modelColors = {
                'gru': '#4e73df',
                'enhanced_gru': '#1cc88a',
                'lstm': '#36b9cc',
                'bilstm': '#f6c23e',
                'cnn_lstm': '#e74a3b'
            };
            for (let m of models) {
                if (data[m].predicted && data[m].targets && data[m].targets.includes(region)) {
                    let idx = data[m].targets.indexOf(region);
                    datasets.push({
                        label: m.toUpperCase(),
                        data: data[m].predicted.map(row => row[idx]),
                        borderColor: modelColors[m] || undefined,
                        borderWidth: 2,
                        pointRadius: 1,
                        fill: false,
                        tension: 0.1,
                        order: 1
                    });
                }
            }
            // Destroy previous chart if exists
            if (forecastChart) forecastChart.destroy();
            let ctx = document.getElementById('forecastChart').getContext('2d');
            forecastChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: datasets
                },
                options: {
                    responsive: true,
                    plugins: {
                        legend: { display: true },
                        title: {
                            display: true,
                            text: 'Next 24h Forecast: All Models (DELHI)'
                        }
                    },
                    interaction: {
                        mode: 'index',
                        intersect: false
                    },
                    scales: {
                        x: {
                            title: { display: true, text: 'Sample' }
                        },
                        y: {
                            title: { display: true, text: 'Load (MW)' }
                        }
                    }
                }
            });
        });
}



/**
 * Load all data for the dashboard
 */
function loadAllData() {
    // Load current data
    fetch('/api/current')
        .then(response => response.json())
        .then(data => {
            currentData = data;
            updateCurrentLoadChart();
            updateCurrentLoadTable();
            updateHistoricalChart();
            updateWeatherChart();
        })
        .catch(error => console.error('Error loading current data:', error));
    
    // Load forecast data
    fetch('/api/predictions')
        .then(response => response.json())
        .then(data => {
            forecastData = data;
            updateForecastChart();
        })
        .catch(error => console.error('Error loading forecast data:', error));
    
    // Update status
    checkStatus();
}

/**
 * Check the update status
 */
function checkStatus() {
    fetch('/api/status')
        .then(response => response.json())
        .then(data => {
            // Update the last update time
            if (data.last_update) {
                lastUpdateTime.textContent = data.last_update;
            }
            
            // Update the status indicator
            updateStatus.className = '';
            if (data.is_updating) {
                updateStatus.textContent = '(Updating...)';
                updateStatus.classList.add('updating');
            } else {
                updateStatus.textContent = '';
            }
        })
        .catch(error => console.error('Error checking status:', error));
}

/**
 * Trigger a manual update
 */
function triggerUpdate() {
    // Show updating status
    updateStatus.textContent = '(Updating...)';
    updateStatus.className = 'updating';
    
    // Trigger update
    fetch('/api/update')
        .then(response => response.json())
        .then(data => {
            console.log('Update triggered:', data);
            
            // Poll for status changes
            const statusCheckInterval = setInterval(() => {
                fetch('/api/status')
                    .then(response => response.json())
                    .then(statusData => {
                        if (!statusData.is_updating) {
                            // Update complete, refresh data
                            clearInterval(statusCheckInterval);
                            loadAllData();
                            
                            // Show success message
                            updateStatus.textContent = '(Update complete)';
                            updateStatus.className = 'success';
                            
                            // Clear success message after 3 seconds
                            setTimeout(() => {
                                updateStatus.textContent = '';
                                updateStatus.className = '';
                            }, 3000);
                        }
                    })
                    .catch(error => {
                        console.error('Error checking status:', error);
                        clearInterval(statusCheckInterval);
                        
                        // Show error message
                        updateStatus.textContent = '(Update failed)';
                        updateStatus.className = 'error';
                    });
            }, 1000);
        })
        .catch(error => {
            console.error('Error triggering update:', error);
            
            // Show error message
            updateStatus.textContent = '(Update failed)';
            updateStatus.className = 'error';
        });
}

/**
 * Update the current load chart
 */
function updateCurrentLoadChart() {
    if (!currentData || !currentData.targets) return;
    
    const ctx = document.getElementById('currentLoadChart').getContext('2d');
    
    // Get the latest values
    const latestIndex = currentData.timestamps ? currentData.timestamps.length - 1 : 0;
    const labels = Object.keys(currentData.targets);
    const values = labels.map(label => {
        const targetData = currentData.targets[label];
        return targetData ? targetData[latestIndex] : 0;
    });
    
    // Destroy existing chart if it exists
    if (currentLoadChart) {
        currentLoadChart.destroy();
    }
    
    // Create new chart
    currentLoadChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Current Load (MW)',
                data: values,
                backgroundColor: labels.map(label => colors[label] || '#6c757d'),
                borderColor: labels.map(label => colors[label] || '#6c757d'),
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Load (MW)'
                    }
                }
            },
            plugins: {
                title: {
                    display: true,
                    text: 'Current Load by Region'
                },
                legend: {
                    display: false
                }
            }
        }
    });
}

/**
 * Update the current load table
 */
function updateCurrentLoadTable() {
    if (!currentData || !currentData.targets) return;
    
    // Clear existing rows
    currentLoadTable.innerHTML = '';
    
    // Get the latest values
    const latestIndex = currentData.timestamps ? currentData.timestamps.length - 1 : 0;
    const regions = Object.keys(currentData.targets);
    
    // Add rows for each region
    regions.forEach(region => {
        const targetData = currentData.targets[region];
        const value = targetData ? targetData[latestIndex] : 0;
        
        const row = document.createElement('tr');
        
        const regionCell = document.createElement('td');
        regionCell.textContent = region;
        
        const valueCell = document.createElement('td');
        valueCell.textContent = value.toFixed(2) + ' MW';
        
        row.appendChild(regionCell);
        row.appendChild(valueCell);
        
        currentLoadTable.appendChild(row);
    });
}

/**
 * Update the forecast chart
 */
function updateForecastChart() {
    if (!forecastData || !forecastData.targets || !forecastData.timestamps) return;
    
    const ctx = document.getElementById('forecastChart').getContext('2d');
    
    // Format timestamps for display
    const labels = forecastData.timestamps.map(ts => {
        return moment(ts).format('HH:mm (DD/MM)');
    });
    
    // Prepare datasets
    const datasets = [];
    Object.keys(forecastData.targets).forEach(target => {
        datasets.push({
            label: target,
            data: forecastData.targets[target],
            borderColor: colors[target] || '#6c757d',
            backgroundColor: 'transparent',
            pointBackgroundColor: colors[target] || '#6c757d',
            borderWidth: 2,
            tension: 0.1
        });
    });
    
    // Destroy existing chart if it exists
    if (forecastChart) {
        forecastChart.destroy();
    }
    
    // Create new chart
    forecastChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: false,
                    title: {
                        display: true,
                        text: 'Load (MW)'
                    }
                },
                x: {
                    ticks: {
                        maxRotation: 45,
                        minRotation: 45
                    }
                }
            },
            plugins: {
                title: {
                    display: true,
                    text: 'Load Forecast for Next 24 Hours'
                }
            }
        }
    });
}

/**
 * Update the historical chart
 */
function updateHistoricalChart() {
    if (!currentData || !currentData.targets || !currentData.timestamps) return;
    
    const ctx = document.getElementById('historicalChart').getContext('2d');
    
    // Format timestamps for display
    const labels = currentData.timestamps.map(ts => {
        return moment(ts).format('HH:mm (DD/MM)');
    });
    
    // Prepare datasets
    const datasets = [];
    Object.keys(currentData.targets).forEach(target => {
        datasets.push({
            label: target,
            data: currentData.targets[target],
            borderColor: colors[target] || '#6c757d',
            backgroundColor: 'transparent',
            pointBackgroundColor: colors[target] || '#6c757d',
            borderWidth: 2,
            tension: 0.1
        });
    });
    
    // Destroy existing chart if it exists
    if (historicalChart) {
        historicalChart.destroy();
    }
    
    // Create new chart
    historicalChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: false,
                    title: {
                        display: true,
                        text: 'Load (MW)'
                    }
                },
                x: {
                    ticks: {
                        maxRotation: 45,
                        minRotation: 45
                    }
                }
            },
            plugins: {
                title: {
                    display: true,
                    text: 'Historical Load (Last 24 Hours)'
                }
            }
        }
    });
}

/**
 * Update the weather chart
 */
function updateWeatherChart() {
    if (!currentData) return;
    
    // Check if we have weather data
    const hasTemperature = currentData.temperature && currentData.temperature.length > 0;
    const hasHumidity = currentData.humidity && currentData.humidity.length > 0;
    const hasWindSpeed = currentData.wind_speed && currentData.wind_speed.length > 0;
    const hasPrecipitation = currentData.precipitation && currentData.precipitation.length > 0;
    
    if (!hasTemperature && !hasHumidity && !hasWindSpeed && !hasPrecipitation) {
        return; // No weather data available
    }
    
    const ctx = document.getElementById('weatherChart').getContext('2d');
    
    // Format timestamps for display
    const labels = currentData.timestamps.map(ts => {
        return moment(ts).format('HH:mm (DD/MM)');
    });
    
    // Prepare datasets
    const datasets = [];
    
    if (hasTemperature) {
        datasets.push({
            label: 'Temperature (°C)',
            data: currentData.temperature,
            borderColor: colors.temperature,
            backgroundColor: 'transparent',
            pointBackgroundColor: colors.temperature,
            borderWidth: 2,
            tension: 0.1,
            yAxisID: 'y'
        });
    }
    
    if (hasHumidity) {
        datasets.push({
            label: 'Humidity (%)',
            data: currentData.humidity,
            borderColor: colors.humidity,
            backgroundColor: 'transparent',
            pointBackgroundColor: colors.humidity,
            borderWidth: 2,
            tension: 0.1,
            yAxisID: 'y1'
        });
    }
    
    if (hasWindSpeed) {
        datasets.push({
            label: 'Wind Speed (km/h)',
            data: currentData.wind_speed,
            borderColor: colors.windSpeed,
            backgroundColor: 'transparent',
            pointBackgroundColor: colors.windSpeed,
            borderWidth: 2,
            tension: 0.1,
            yAxisID: 'y2'
        });
    }
    
    if (hasPrecipitation) {
        datasets.push({
            label: 'Precipitation (mm)',
            data: currentData.precipitation,
            borderColor: colors.precipitation,
            backgroundColor: 'transparent',
            pointBackgroundColor: colors.precipitation,
            borderWidth: 2,
            tension: 0.1,
            yAxisID: 'y3'
        });
    }
    
    // Destroy existing chart if it exists
    if (weatherChart) {
        weatherChart.destroy();
    }
    
    // Create new chart
    weatherChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    type: 'linear',
                    display: hasTemperature,
                    position: 'left',
                    title: {
                        display: true,
                        text: 'Temperature (°C)'
                    }
                },
                y1: {
                    type: 'linear',
                    display: hasHumidity,
                    position: 'right',
                    grid: {
                        drawOnChartArea: false
                    },
                    title: {
                        display: true,
                        text: 'Humidity (%)'
                    },
                    min: 0,
                    max: 100
                },
                y2: {
                    type: 'linear',
                    display: hasWindSpeed,
                    position: 'right',
                    grid: {
                        drawOnChartArea: false
                    },
                    title: {
                        display: true,
                        text: 'Wind Speed (km/h)'
                    }
                },
                y3: {
                    type: 'linear',
                    display: hasPrecipitation,
                    position: 'right',
                    grid: {
                        drawOnChartArea: false
                    },
                    title: {
                        display: true,
                        text: 'Precipitation (mm)'
                    }
                },
                x: {
                    ticks: {
                        maxRotation: 45,
                        minRotation: 45
                    }
                }
            },
            plugins: {
                title: {
                    display: true,
                    text: 'Weather Conditions'
                }
            }
        }
    });
}

// Initialize dashboard when the page loads
document.addEventListener('DOMContentLoaded', initDashboard);
