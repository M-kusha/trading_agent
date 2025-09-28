(() => {
  // ---------- Config ----------
  const qs = new URLSearchParams(location.search);
  const DATA_SRC = qs.get('src') || 'logs/flow_dashboard/graph.json';
  const LOG_SRC  = qs.get('log') || 'logs/flow_dashboard/flow_dashboard.log';

  // ---------- State ----------
  let cy = null;
  let autoTimer = null;
  const timeSeries = {}; // { module: { rpsR:[], rpsW:[], p95:[] } }
  const MAX_POINTS = 60;

  // UI refs
  const $ = id => document.getElementById(id);
  $('data-src').textContent = DATA_SRC;
  $('link-log').href = LOG_SRC;

  // Charts (created per selection)
  let chartRps = null;
  let chartLat = null;

  // ---------- Helpers ----------
  const fmt = {
    num: n => (n===undefined||n===null? '–' : Number(n).toFixed(2)),
    int: n => (n===undefined||n===null? '–' : String(Math.round(n))),
    time: ts => {
      if(!ts) return '–';
      const d = new Date(ts*1000);
      return d.toLocaleString();
    }
  };

  function statusToClass(s) {
    if (s === 'RED') return 'red';
    if (s === 'YELLOW') return 'yellow';
    return 'green';
  }

  function capList(arr, max=200) {
    return (arr || []).slice(0, max);
  }

  function updateSummary(state) {
    const sc = state?.status_counts || {};
    $('count-green').textContent = sc.green ?? 0;
    $('count-yellow').textContent = sc.yellow ?? 0;
    $('count-red').textContent = sc.red ?? 0;
    $('modules-total').textContent = (state?.modules ?? 0) + ' modules';
    $('edges-total').textContent = (state?.edges ?? 0) + ' edges';
    $('window-seconds').textContent = state?.window_seconds ?? '—';
    const t = state?.generated_at ? new Date(state.generated_at*1000) : null;
    $('last-updated').textContent = t ? `updated ${t.toLocaleTimeString()}` : '–';
  }

  // Build cy elements from JSON
  function buildElements(graph, healthMap) {
    const nodes = (graph?.nodes || []).map(n => {
      const hm = healthMap[n.id] || {};
      return {
        data: {
          id: n.id,
          label: n.label || n.id,
          status: hm.status || n.status || 'GREEN',
          rpsR: hm.meters?.rps_reads || 0,
          rpsW: hm.meters?.rps_writes || 0,
          p95:  hm.latency_ms?.p95 || 0,
          avg:  hm.latency_ms?.avg || 0,
          breaker: hm.breaker || 'CLOSED',
          health: hm.health_score ?? 1.0,
          reason: hm.reason || 'ok',
          provides: hm.provides_keys || [],
          consumes: hm.consumes_keys || [],
          executions: hm.executions || 0
        },
        classes: statusToClass(hm.status || n.status || 'GREEN')
      };
    });

    const edges = (graph?.edges || []).map((e, i) => ({
      data: { id: `e${i}-${e.from}-${e.to}`, source: e.from, target: e.to }
    }));

    return { nodes, edges };
  }

  function ensureSeries(mod, hm) {
    if (!timeSeries[mod]) timeSeries[mod] = { rpsR: [], rpsW: [], p95: [] };
    const s = timeSeries[mod];
    s.rpsR.push(Number(hm.meters?.rps_reads || 0));
    s.rpsW.push(Number(hm.meters?.rps_writes || 0));
    s.p95.push(Number(hm.latency_ms?.p95 || 0));
    if (s.rpsR.length > MAX_POINTS) { s.rpsR.shift(); s.rpsW.shift(); s.p95.shift(); }
  }

  function renderGraph(data) {
    const { graph, health } = data;
    const { nodes, edges } = buildElements(graph, health);

    if (!cy) {
      cy = cytoscape({
        container: $('graph'),
        elements: { nodes, edges },
        style: [
          { selector: 'node',
            style: {
              'background-color': '#8b95a5',
              'label': 'data(label)',
              'color': 'var(--text)',
              'text-outline-color': 'var(--bg)',
              'text-outline-width': 2,
              'font-size': 12,
              'width': 28, 'height': 28
            }
          },
          { selector: 'node.green',  style: {'background-color': 'var(--green)'} },
          { selector: 'node.yellow', style: {'background-color': 'var(--yellow)'} },
          { selector: 'node.red',    style: {'background-color': 'var(--red)'} },
          { selector: 'edge',
            style: {
              'width': 2, 'line-color': '#556', 'target-arrow-shape': 'triangle',
              'target-arrow-color': '#556', 'curve-style': 'bezier'
            }
          },
          { selector: '.faded', style: {'opacity': 0.15} }
        ],
        layout: { name: 'cose', animate: false, padding: 20 }
      });

      cy.on('tap', 'node', evt => selectModule(evt.target.data().id, data));
      cy.on('layoutstop', () => { $('graph-status').textContent = `${cy.nodes().length} nodes / ${cy.edges().length} edges`; });
      $('graph-status').textContent = `${nodes.length} nodes / ${edges.length} edges`;
    } else {
      cy.elements().remove();
      cy.add(nodes);
      cy.add(edges);
      cy.layout({ name: 'cose', animate: false, padding: 20 }).run();
      $('graph-status').textContent = `${nodes.length} nodes / ${edges.length} edges`;
    }

    // hide-green toggle
    applyHideGreen();
  }

  function selectModule(modName, data) {
    const hm = data.health?.[modName] || {};
    // Update side panel
    $('module-none').classList.add('hidden');
    $('module-details').classList.remove('hidden');
    $('mod-title').textContent = modName;

    const st = hm.status || 'GREEN';
    const dot = $('mod-status-dot');
    dot.className = 'dot ' + statusToClass(st);

    $('mod-status').textContent = st;
    $('mod-breaker').textContent = hm.breaker || 'CLOSED';
    $('mod-reason').textContent = hm.reason || 'ok';
    $('mod-lat-p95').textContent = fmt.int(hm.latency_ms?.p95);
    $('mod-lat-avg').textContent = fmt.int(hm.latency_ms?.avg);
    $('mod-execs').textContent = fmt.int(hm.executions);
    $('mod-rps-r').textContent = fmt.num(hm.meters?.rps_reads);
    $('mod-rps-w').textContent = fmt.num(hm.meters?.rps_writes);

    const healthPct = Math.max(0, Math.min(1, Number(hm.health_score ?? 1.0))) * 100;
    const fill = $('healthbar-fill');
    fill.style.width = `${healthPct}%`;
    fill.style.background = (st === 'RED') ? 'var(--red)' : (st === 'YELLOW' ? 'var(--yellow)' : 'var(--green)');

    // Keys
    const mkChip = k => `<span class="chip" title="${k}">${k}</span>`;
    $('mod-provides').innerHTML = (hm.provides_keys || []).map(mkChip).join('') || '<span class="muted">—</span>';
    $('mod-consumes').innerHTML = (hm.consumes_keys || []).map(mkChip).join('') || '<span class="muted">—</span>';

    // Theses
    const theses = (data.module_theses?.[modName] || []);
    $('mod-theses').innerHTML = (theses.length
      ? theses.map(t => `
          <div class="th">
            <div class="meta">
              <span class="badge">key: ${t.key}</span>
              <span class="badge">v${t.version}</span>
              <span class="badge">conf: ${fmt.num(t.confidence)}</span>
              <span class="badge">${(t.age_seconds|0)}s old</span>
              ${t.source_module ? `<span class="badge">src: ${t.source_module}</span>` : ''}
            </div>
            <div class="text">${(t.thesis || '').toString().replace(/[&<>]/g, s => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[s]))}</div>
          </div>`).join('')
      : '<span class="muted">No theses for this module.</span>');

    // Series (for charts)
    ensureSeries(modName, hm);
    drawCharts(modName);
  }

  function drawCharts(modName) {
    const s = timeSeries[modName];
    const labels = Array.from({length: s.rpsR.length}, (_, i) => i+1);

    // RPS chart
    const ctxR = $('chart-rps').getContext('2d');
    if (chartRps) chartRps.destroy();
    chartRps = new Chart(ctxR, {
      type: 'line',
      data: {
        labels,
        datasets: [
          { label: 'reads', data: s.rpsR, tension: .3, pointRadius: 0 },
          { label: 'writes', data: s.rpsW, tension: .3, pointRadius: 0 }
        ]
      },
      options: {
        responsive: true,
        animation: false,
        plugins: { legend: { display: true }, tooltip: { enabled: true } },
        scales: { x: { display: false }, y: { ticks: { precision: 2 } } }
      }
    });

    // Latency chart
    const ctxL = $('chart-latency').getContext('2d');
    if (chartLat) chartLat.destroy();
    chartLat = new Chart(ctxL, {
      type: 'line',
      data: { labels, datasets: [{ label: 'p95', data: s.p95, tension: .3, pointRadius: 0 }] },
      options: {
        responsive: true, animation: false,
        plugins: { legend: { display: true }, tooltip: { enabled: true } },
        scales: { x: { display: false }, y: { ticks: { precision: 0 } } }
      }
    });
  }

  function renderAlerts(alerts) {
    const root = $('alerts');
    if (!alerts?.length) { root.innerHTML = '<div class="row"><div class="muted">No alerts</div></div>'; return; }
    root.innerHTML = alerts.map(a => `
      <div class="row">
        <div class="sev ${a.level.toLowerCase()}">${a.level}</div>
        <div class="badge">p95 ${fmt.int(a.p95_ms)}ms</div>
        <div>${a.module} — <span class="muted">${a.reason}</span></div>
      </div>`).join('');
  }

  function renderEvents(debug) {
    const evs = capList(debug?.events_tail, 150);
    const rows = evs.map(e => `
      <div class="row">
        <div class="badge">${e.type || 'event'}</div>
        <div class="badge mono">${e.module || '–'}</div>
        <div class="mono">${(e.key || '')} ${e.reason ? '— '+e.reason : ''}</div>
      </div>`).join('');
    $('events').innerHTML = rows || '<div class="row"><div class="muted">No recent events</div></div>';
  }

  function renderTopLat(debug) {
    const tl = capList(debug?.top_latency_p95, 10);
    $('top-lat').innerHTML = (tl.length ? tl.map(e => `
      <div class="row">
        <div class="badge">${e.module}</div>
        <div class="badge">p95</div>
        <div class="mono">${fmt.int(e.p95_ms)} ms</div>
      </div>`).join('') : '<div class="row"><div class="muted">No data</div></div>');
  }

  function applyHideGreen() {
    if (!cy) return;
    const hide = $('toggle-hide-green').checked;
    cy.nodes().removeClass('faded');
    if (hide) {
      cy.nodes('.green').addClass('faded');
      cy.edges().forEach(e => {
        const sG = e.source().hasClass('green');
        const tG = e.target().hasClass('green');
        if (sG && tG) e.addClass('faded');
      });
    }
  }

  function applySearch(term) {
    if (!cy) return;
    const t = (term || '').trim().toLowerCase();
    cy.nodes().removeClass('faded');
    if (!t) { applyHideGreen(); return; }
    cy.nodes().forEach(n => {
      if (!n.id().toLowerCase().includes(t) && !(n.data('label')||'').toLowerCase().includes(t)) {
        n.addClass('faded');
      }
    });
  }

  // ---------- Fetch & Render ----------
  async function loadOnce() {
    try {
      const res = await fetch(DATA_SRC + '?_=' + Date.now(), { cache: 'no-store' });
      if (!res.ok) throw new Error(res.status + ' ' + res.statusText);
      const data = await res.json();

      // update time series
      const health = data.health || {};
      Object.keys(health).forEach(m => ensureSeries(m, health[m]));

      renderGraph(data);
      updateSummary(data.state);
      renderAlerts(data.alerts);
      renderEvents(data.debug);
      renderTopLat(data.debug);
    } catch (e) {
      console.warn('Fetch failed:', e);
      $('last-updated').textContent = 'source unavailable';
    }
  }

  function startAuto() {
    stopAuto();
    if ($('toggle-refresh').checked) {
      const ms = Number($('refresh-interval').value || 3000);
      autoTimer = setInterval(loadOnce, ms);
    }
  }
  function stopAuto(){ if (autoTimer) { clearInterval(autoTimer); autoTimer = null; } }

  // ---------- Wire UI ----------
  $('toggle-refresh').addEventListener('change', startAuto);
  $('refresh-interval').addEventListener('change', startAuto);
  $('toggle-hide-green').addEventListener('change', applyHideGreen);
  $('toggle-dark').addEventListener('change', (e) => {
    document.documentElement.classList.toggle('light', !e.target.checked);
    document.body.classList.toggle('light', !e.target.checked);
  });
  $('search').addEventListener('input', (e) => applySearch(e.target.value));
  $('btn-export').addEventListener('click', () => {
    if (!cy) return;
    const png = cy.png({ bg: getComputedStyle(document.body).backgroundColor });
    const a = document.createElement('a'); a.href = png; a.download = 'flow-graph.png'; a.click();
  });
  $('btn-copy-name').addEventListener('click', () => {
    const name = $('mod-title').textContent || '';
    if (!name) return;
    navigator.clipboard.writeText(name);
  });

  // ---------- Boot ----------
  loadOnce().then(startAuto);
})();
