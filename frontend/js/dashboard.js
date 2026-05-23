const BACKEND = window.location.protocol === 'file:' || window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1' ? 'http://localhost:8000' : 'https://sanz-ai-tutor.onrender.com';
const ONLINE_MIN = 10;
const COLORS = ['#4F6EF7', '#00D4AA', '#8B5CF6', '#F59E0B', '#EF4444', '#06B6D4', '#F97316', '#22C55E', '#EC4899', '#14B8A6'];
const TYPE_NAMES = {
  main_answer: '💬 Answer',
  cross_check: '✅ Cross-check',
  socratic_redirect: '🎯 Socratic',
  quiz_generate: '📝 Quiz',
  image_generate: '🎨 Image',
  rag_agent: '📚 RAG',
  weekly_report: '📧 Weekly Email'
};

let authed = false, allHistory = [], refreshTimer = null;
let apiChartInstance = null;
let typeChartInstance = null;

window.onload = () => {
  updateClock();
  setInterval(updateClock, 1000);
  
  // Collapse other sections by default on load except overview
  document.querySelectorAll('.sb-sec-group').forEach(group => {
    if (group.id !== 'group-overview') {
      group.classList.add('collapsed');
      const head = group.previousElementSibling;
      if (head) head.classList.add('collapsed');
    }
  });

  // Verify stored session token
  const storedToken = sessionStorage.getItem('admin_token');
  if (storedToken) {
    authed = true;
    document.getElementById('authGate').style.display = 'none';
    loadData();
    startRefresh();
  }
};

function updateClock() {
  const tb = document.getElementById('tbTime');
  if (tb) tb.textContent = new Date().toLocaleTimeString();
}

/* ── SECURE SERVER AUTHENTICATION ── */
async function doAuth() {
  const u = document.getElementById('au').value.trim();
  const p = document.getElementById('ap').value;
  const err = document.getElementById('authErr');
  
  if (!u || !p) {
    err.textContent = "❌ Please enter both fields.";
    err.classList.add('show');
    return;
  }
  
  err.classList.remove('show');
  try {
    const res = await fetch(`${BACKEND}/admin/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username: u, password: p })
    });
    
    if (!res.ok) {
      err.textContent = "❌ Invalid username or password.";
      err.classList.add('show');
      return;
    }
    
    const d = await res.json();
    if (d.status === 'success' && d.token) {
      authed = true;
      sessionStorage.setItem('admin_token', d.token);
      document.getElementById('authGate').style.display = 'none';
      loadData();
      startRefresh();
    } else {
      err.textContent = "❌ Authentication failed.";
      err.classList.add('show');
    }
  } catch (e) {
    console.error(e);
    err.textContent = "❌ Network connection error.";
    err.classList.add('show');
  }
}

function doLogout() {
  clearInterval(refreshTimer);
  sessionStorage.removeItem('admin_token');
  location.reload();
}

function startRefresh() {
  clearInterval(refreshTimer);
  refreshTimer = setInterval(loadData, 30000);
}

/* ── SIDEBAR SECTION ACCORDIONS ── */
function toggleSidebarSection(sectionId, headerEl) {
  const group = document.getElementById(sectionId);
  if (!group) return;
  
  const isCollapsed = group.classList.toggle('collapsed');
  if (headerEl) {
    headerEl.classList.toggle('collapsed', isCollapsed);
  }
}

function showPage(n, el) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(x => x.classList.remove('active'));
  
  const pg = document.getElementById('page-' + n);
  if (pg) pg.classList.add('active');
  if (el) el.classList.add('active');
  
  const title = document.getElementById('pageTitle');
  if (title && el) {
    const badge = el.querySelector('.nav-badge');
    let baseText = "";
    el.childNodes.forEach(node => {
      if (node.nodeType === Node.TEXT_NODE) {
        baseText += node.textContent;
      } else if (node.nodeName !== "I" && node !== badge) {
        baseText += node.innerText || "";
      }
    });
    baseText = baseText.trim();
    title.textContent = badge && badge.innerText !== '—' && badge.innerText !== '0' 
      ? `${baseText} - ${badge.innerText}` 
      : baseText;
  }
  
  if (n === 'resources') loadPDFs();
  if (n === 'leaderboard') loadLeaderboard();
  if (n === 'api') loadApiUsage();
  if (n === 'tokens') loadTokenUsage();
  if (n === 'parents') loadParents();
  if (n === 'accounts') loadUserAccounts();
  if (n === 'email') loadEmailLog();
}

function getAuthHeaders() {
  const token = sessionStorage.getItem('admin_token') || '';
  return {
    'Authorization': `Bearer ${token}`
  };
}

function timeAgo(ts) {
  if (!ts) return '—';
  const d = Math.floor((Date.now() - new Date(ts).getTime()) / 1000);
  if (d < 60) return d + 's ago';
  if (d < 3600) return Math.floor(d / 60) + 'm ago';
  if (d < 86400) return Math.floor(d / 3600) + 'h ago';
  return Math.floor(d / 86400) + 'd ago';
}

function isOnline(ts) {
  return ts && (Date.now() - new Date(ts).getTime()) / 60000 <= ONLINE_MIN;
}

function ini(n) {
  return n ? n.trim().split(' ').map(w => w[0]).join('').toUpperCase().slice(0, 2) : '??';
}

function esc(s) {
  return s == null ? '' : String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

function pct(v, t) {
  return t ? Math.round(v / t * 100) : 0;
}

/* ── CUSTOM SPACIOUS MODAL OVERLAYS ── */
function openCustomModal(title, bodyHtml, onConfirm, confirmText = 'Confirm', cancelText = 'Cancel') {
  document.getElementById('modalTitle').textContent = title;
  document.getElementById('modalBody').innerHTML = bodyHtml;
  
  const cancelBtn = document.getElementById('modalCancelBtn');
  if (cancelBtn) cancelBtn.textContent = cancelText;
  
  const btn = document.getElementById('modalConfirmBtn');
  if (btn) {
    if (onConfirm) {
      btn.style.display = 'inline-block';
      btn.textContent = confirmText;
      btn.onclick = () => {
        onConfirm();
        closeCustomModal();
      };
    } else {
      btn.style.display = 'none';
    }
  }
  
  document.getElementById('customModal').classList.add('show');
}

function closeCustomModal() {
  document.getElementById('customModal').classList.remove('show');
}

/* ── OVERVIEW CHART.JS RENDERERS ── */
function renderOverviewCharts(history, stats, apiUsage) {
  // Chart 1: Combined Daily API Calls & Token Usage Trends
  const apiHistory = apiUsage?.history || {};
  const dates = Object.keys(apiHistory).sort();
  const callsData = dates.map(d => apiHistory[d] || 0);
  
  const apiCtx = document.getElementById('apiCallsChart')?.getContext('2d');
  if (apiCtx) {
    if (apiChartInstance) apiChartInstance.destroy();
    
    apiChartInstance = new Chart(apiCtx, {
      type: 'line',
      data: {
        labels: dates.map(d => d.slice(5)), // MM-DD format
        datasets: [{
          label: 'API Calls',
          data: callsData,
          borderColor: '#4F6EF7',
          backgroundColor: 'rgba(79, 110, 247, 0.15)',
          fill: true,
          tension: 0.4,
          borderWidth: 2.5
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false }
        },
        scales: {
          x: { grid: { display: false }, ticks: { color: '#585D7B', font: { family: 'DM Sans', size: 9.5 } } },
          y: { grid: { color: 'rgba(255,255,255,0.03)' }, ticks: { color: '#585D7B', font: { family: 'DM Sans', size: 9.5 } } }
        }
      }
    });
  }

  // Chart 2: Call Type Distribution
  const typeHistory = apiUsage?.by_type || {};
  const typeLabels = Object.keys(typeHistory).map(k => TYPE_NAMES[k] || k);
  const typeValues = Object.values(typeHistory);
  
  const typeCtx = document.getElementById('callDistributionChart')?.getContext('2d');
  if (typeCtx) {
    if (typeChartInstance) typeChartInstance.destroy();
    
    typeChartInstance = new Chart(typeCtx, {
      type: 'doughnut',
      data: {
        labels: typeLabels,
        datasets: [{
          data: typeValues,
          backgroundColor: COLORS.slice(0, typeLabels.length),
          borderWidth: 0
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { 
            position: 'right',
            labels: { color: '#949AB8', font: { family: 'DM Sans', size: 10 } }
          }
        },
        cutout: '72%'
      }
    });
  }
}

/* ── SYSTEM DATA LOAD ── */
async function loadData() {
  if (!authed) return;
  try {
    const headers = getAuthHeaders();
    const [sR, hR, tR] = await Promise.all([
      fetch(`${BACKEND}/admin/stats`, { headers }),
      fetch(`${BACKEND}/admin/history`, { headers }),
      fetch(`${BACKEND}/admin/token-usage`, { headers })
    ]);
    
    if (sR.status === 401 || hR.status === 401 || tR.status === 401) {
      doLogout();
      return;
    }
    
    if (!sR.ok) {
      showToast('⚠️ Error loading dashboard: ' + sR.status);
      return;
    }
    
    const stats = await sR.json();
    const histData = hR.ok ? await hR.json() : {};
    const tokData = tR.ok ? await tR.json() : {};
    const history = histData.history || []; 
    allHistory = history;
    const banned = stats.blacklisted_users || [];
    const apiUsage = stats.api_usage || {};
    const tokUsers = tokData.users || [];
    const totalParents = stats.total_parents || 0;
    const totalChildren = stats.total_children || 0;

    document.getElementById('histBadge').textContent = history.length;
    document.getElementById('blBadge').textContent = banned.length;
    document.getElementById('bl-count').textContent = banned.length;
    document.getElementById('parentBadge').textContent = totalParents;
    document.getElementById('tokBadge').textContent = tokData.total_today || 0;

    const userMap = {};
    history.forEach(r => {
      const k = r.user || '?';
      if (!userMap[k]) userMap[k] = { name: k, grade: r.grade || '—', lastSeen: null, count: 0 };
      userMap[k].count++;
      if (r.time && (!userMap[k].lastSeen || new Date(r.time) > new Date(userMap[k].lastSeen))) {
        userMap[k].lastSeen = r.time;
      }
    });
    const users = Object.values(userMap);
    const online = users.filter(u => isOnline(u.lastSeen));

    // Consolidate spacious metric counts
    document.getElementById('s-total').textContent = stats.total_questions ?? history.length;
    document.getElementById('s-online').textContent = online.length;
    
    const elParents = document.getElementById('s-parents');
    const elChildren = document.getElementById('s-children');
    const elBan = document.getElementById('s-ban');
    
    if (elParents) elParents.textContent = totalParents;
    if (elChildren) elChildren.textContent = totalChildren + ' children';
    if (elBan) elBan.textContent = banned.length;

    // API limits progression bar
    const totalRem = apiUsage.total_remaining != null ? apiUsage.total_remaining : '—';
    document.getElementById('s-apirem').textContent = totalRem;
    document.getElementById('apiBadge').textContent = totalRem;
    
    const apiPct = typeof totalRem === 'number' ? Math.round(totalRem / 1500 * 100) : 0;
    const bar = document.getElementById('s-apibar');
    if (bar) {
      bar.style.width = apiPct + '%';
      bar.style.background = apiPct > 50 ? 'var(--ok)' : apiPct > 20 ? 'var(--yel)' : 'var(--dan)';
    }

    // Financial summaries
    const todayTok = tokData.total_today || 0;
    const costUsd = tokData.est_cost_usd || 0;
    const costLkr = tokData.est_cost_lkr || 0;
    document.getElementById('s-tokens').textContent = todayTok.toLocaleString();
    document.getElementById('s-tokencost').textContent = `Est: $${costUsd} · LKR ${costLkr}`;

    renderUserList('ov-online', online, true);
    renderTokenOverview(tokUsers);
    renderRecent(history.slice(-8).reverse());
    renderUsersTable(users);
    renderHistory([...history].reverse());
    renderBlacklist(banned);
    renderSubjects(history, stats.subjects || {});
    renderLiveFeed(history.slice(-15).reverse());
    populateFilters(history);
    renderApiUsage(apiUsage);
    
    // Render visual Chart.js dashboards
    renderOverviewCharts(history, stats, apiUsage);
  } catch (e) {
    console.error(e);
    showToast('⚠️ Data refresh error: ' + e.message);
  }
}

/* ── TOKEN OVERVIEW (Main page) ── */
function renderTokenOverview(users) {
  const el = document.getElementById('ov-tokens');
  if (!users || !users.length) {
    el.innerHTML = '<div class="empty-state">No token data yet</div>';
    return;
  }
  const top = users.slice(0, 5);
  const max = top[0]?.today_tokens || 1;
  el.innerHTML = top.map(u => `
    <div class="tok-row">
      <div class="tok-user">${esc(u.user)}</div>
      <div class="tok-bar"><div class="tok-fill" style="width:${pct(u.today_tokens, max)}%"></div></div>
      <div class="tok-val">${(u.today_tokens || 0).toLocaleString()}</div>
      <div class="tok-cost">$${u.est_cost_today || 0}</div>
    </div>`).join('');
}

/* ── TOKEN PAGE DATA LOAD ── */
async function loadTokenUsage() {
  const tb = document.getElementById('tokenBody');
  if (!tb) return;
  tb.innerHTML = '<tr><td colspan="8" class="empty-state">Loading usage...</td></tr>';
  try {
    const res = await fetch(`${BACKEND}/admin/token-usage`, { headers: getAuthHeaders() });
    if (!res.ok) {
      tb.innerHTML = '<tr><td colspan="8" class="empty-state">Error loading tokens</td></tr>';
      return;
    }
    const d = await res.json();
    const users = d.users || [];
    document.getElementById('tok-total').textContent = (d.total_today || 0).toLocaleString();
    document.getElementById('tok-cost-usd').textContent = '$' + d.est_cost_usd;
    document.getElementById('tok-cost-lkr').textContent = 'LKR ' + d.est_cost_lkr;
    document.getElementById('tok-users').textContent = users.length;
    
    if (!users.length) {
      tb.innerHTML = '<tr><td colspan="8" class="empty-state">No token logs</td></tr>';
      return;
    }
    tb.innerHTML = users.map((u, i) => `<tr>
      <td style="color:var(--txt3)">${i + 1}</td>
      <td><div style="display:flex;align-items:center;gap:8px"><div class="avatar">${ini(u.user)}</div>${esc(u.user)}</div></td>
      <td style="color:var(--acc);font-weight:700">${(u.today_tokens || 0).toLocaleString()}</td>
      <td>${u.today_calls || 0}</td>
      <td style="color:var(--txt2)">${(u.month_tokens || 0).toLocaleString()}</td>
      <td style="color:var(--txt3)">${(u.total_tokens || 0).toLocaleString()}</td>
      <td style="color:var(--ok);font-weight:700">$${u.est_cost_today || 0}</td>
      <td style="color:var(--yel)">LKR ${u.est_cost_lkr || 0}</td>
    </tr>`).join('');
  } catch (e) {
    tb.innerHTML = '<tr><td colspan="8" class="empty-state">⚠️ ' + esc(e.message) + '</td></tr>';
  }
}

/* ── API PAGES DETAILS ── */
async function loadApiUsage() {
  try {
    const res = await fetch(`${BACKEND}/admin/api-usage`, { headers: getAuthHeaders() });
    if (res.ok) {
      const d = await res.json();
      renderApiUsage(d.usage || {});
    }
  } catch (e) {}
}

function renderApiUsage(u) {
  if (!u || !u.models) return;
  document.getElementById('api-total').textContent = u.total_calls_today || 0;
  document.getElementById('api-rem').textContent = u.total_remaining || 0;
  document.getElementById('api-date').textContent = u.today || '—';
  
  const models = u.models || {};
  document.getElementById('apiModels').innerHTML = Object.entries(models).map(([m, d]) => {
    const p = d.limit ? Math.round(d.used / d.limit * 100) : 0;
    const cls = p < 50 ? 'safe' : p < 80 ? 'warn' : 'danger';
    return `<div class="gauge-wrap">
      <div class="gauge-label">${esc(m)}</div>
      <div class="gauge-track"><div class="gauge-fill ${cls}" style="width:${p}%"></div></div>
      <div class="gauge-txt">${d.used}/${d.limit} (${p}%)</div>
    </div>`;
  }).join('') || '<div class="empty-state">No model gauges</div>';
  
  const types = u.by_type || {};
  const maxT = Math.max(...Object.values(types), 1);
  document.getElementById('apiTypes').innerHTML = Object.entries(types).sort((a, b) => b[1] - a[1]).map(([t, c], i) => `
    <div class="bar-row">
      <div class="bar-label">${TYPE_NAMES[t] || t}</div>
      <div class="bar-track"><div class="bar-fill" style="width:${pct(c, maxT)}%;background:${COLORS[i % COLORS.length]}"></div></div>
      <div class="bar-num" style="color:${COLORS[i % COLORS.length]}">${c}</div>
    </div>`).join('') || '<div class="empty-state">No queries yet</div>';
    
  const hist = u.history || {};
  const maxH = Math.max(...Object.values(hist), 1);
  document.getElementById('apiHistory').innerHTML = Object.entries(hist).sort().map(([d, c]) => `
    <div class="bar-row">
      <div class="bar-label">${d}</div>
      <div class="bar-track"><div class="bar-fill" style="width:${pct(c, maxH)}%;background:var(--yel)"></div></div>
      <div class="bar-num" style="color:var(--yel)">${c}</div>
    </div>`).join('') || '<div class="empty-state">No historical history</div>';
}

/* ── PARENTS PORTALS ── */
async function loadParents() {
  const el = document.getElementById('parentList');
  if (!el) return;
  el.innerHTML = '<div class="empty-state">Loading parents...</div>';
  try {
    const res = await fetch(`${BACKEND}/admin/parents`, { headers: getAuthHeaders() });
    if (!res.ok) {
      el.innerHTML = '<div class="empty-state">Error loading parents</div>';
      return;
    }
    const d = await res.json();
    const parents = d.parents || [];
    document.getElementById('parent-count').textContent = parents.length + ' parents';
    
    if (!parents.length) {
      el.innerHTML = '<div class="empty-state">No parents system registered</div>';
      return;
    }
    el.innerHTML = parents.map(p => {
      const isStudent = p.type === 'student_account';
      const badge = isStudent
        ? '<span class="badge" style="background:var(--ok-dim);color:var(--ok);margin-left:8px">👤 Student</span>'
        : '<span class="badge" style="background:var(--pri-dim);color:var(--pri);margin-left:8px">👪 Parent</span>';
      
      const childrenHtml = (p.children || []).map(c => `
        <div style="display:flex;align-items:center;gap:8px;margin:4px 0;">
          <span class="child-chip">🎓 ${esc(c.name)} · G${c.grade}</span>
          <button class="btn btn-yel" style="padding:4px 8px;font-size:9px" onclick="sendEmailPromptChild('${esc(p.email)}','${esc(c.name)}')"><i class="fas fa-paper-plane"></i></button>
        </div>`).join('');
        
      return `<div class="parent-row">
        <div class="pr-top">
          <div>
            <div class="pr-email">${esc(p.name)}${badge} <span style="color:var(--txt3);font-weight:400;font-size:11px">· ${esc(p.email)}</span></div>
            <div class="pr-name">Plan: ${p.plan || 'free'} · Registered ${timeAgo(p.created)}</div>
          </div>
          <button class="btn btn-yel" style="padding:6px 12px;font-size:10px" onclick="sendEmailPrompt('${esc(p.email)}')"><i class="fas fa-envelope"></i> Send Report</button>
        </div>
        <div class="pr-children">
          ${childrenHtml || '<span style="font-size:11px;color:var(--txt3)">No registered child</span>'}
        </div>
      </div>`;
    }).join('');
  } catch (e) {
    el.innerHTML = '<div class="empty-state">⚠️ Parent records fetch failed</div>';
  }
}

function sendEmailPrompt(email) {
  document.getElementById('emailTo').value = email;
  const navItem = document.querySelector('.nav-item[onclick*="email"]');
  showPage('email', navItem);
  showToast('📧 Parent email address mapped. Add student name and send.');
}

function sendEmailPromptChild(email, childName) {
  document.getElementById('emailTo').value = email;
  document.getElementById('emailChild').value = childName;
  const navItem = document.querySelector('.nav-item[onclick*="email"]');
  showPage('email', navItem);
  showToast('📧 Ready to send advisor report! Click Dispatch.');
}

/* ── EMAIL SYSTEM CRON ── */
async function loadEmailLog() {
  const el = document.getElementById('emailLog');
  if (!el) return;
  el.innerHTML = '<div class="empty-state">Loading...</div>';
  try {
    const res = await fetch(`${BACKEND}/admin/email/log`, { headers: getAuthHeaders() });
    if (!res.ok) {
      el.innerHTML = '<div class="empty-state">Logs loading error</div>';
      return;
    }
    const d = await res.json();
    const log = [...(d.log || [])].reverse();
    
    if (!log.length) {
      el.innerHTML = '<div class="empty-state">No email advisor report dispatches active yet</div>';
      return;
    }
    el.innerHTML = log.map(e => `
      <div class="email-row">
        <div class="email-status ${e.status === 'success' ? 'es-ok' : 'es-fail'}"></div>
        <div style="flex:1">
          <div style="font-weight:600;font-size:12px">${esc(e.parent)}</div>
          <div style="color:var(--txt3);font-size:11px">Student: ${esc(e.child)} · Sent ${timeAgo(e.sent_at)}</div>
          ${e.error ? `<div style="color:var(--dan);font-size:10px">${esc(e.error)}</div>` : ''}
        </div>
        <span class="badge ${e.status === 'success' ? 'bd-online' : 'bd-ban'}">${e.status === 'success' ? '✅ Success' : '❌ Failed'}</span>
      </div>`).join('');
  } catch (e) {
    el.innerHTML = '<div class="empty-state">⚠️ Email logs load failed</div>';
  }
}

async function sendEmailNow() {
  const email = document.getElementById('emailTo').value.trim();
  const child = document.getElementById('emailChild').value.trim();
  if (!email || !child) {
    showToast('⚠️ Please fill both advisor email and student name.');
    return;
  }
  showToast('⏳ Dispatching advisor report...');
  try {
    const res = await fetch(`${BACKEND}/admin/email/send-now?parent_email=${encodeURIComponent(email)}&child_name=${encodeURIComponent(child)}`, {
      method: 'POST',
      headers: getAuthHeaders()
    });
    const d = await res.json();
    if (d.status === 'success') {
      showToast('✅ Email advisor report dispatched successfully to ' + email);
    } else {
      showToast('❌ Dispatch failed: ' + (d.message || 'Error'));
    }
    loadEmailLog();
  } catch (e) {
    showToast('❌ SMTP dispatch network error');
  }
}

/* ── LIVE RECENT QUERY PLOTS ── */
function renderLiveFeed(rows) {
  const el = document.getElementById('liveFeed');
  if (!el) return;
  document.getElementById('rt-count').textContent = rows.length + ' queries';
  if (!rows.length) {
    el.innerHTML = '<div class="empty-state">No real-time queries logged</div>';
    return;
  }
  el.innerHTML = rows.map(r => {
    const n = r.user || '?';
    const on = isOnline(r.time);
    return `<div style="display:flex;align-items:center;gap:12px;padding:10px 18px;border-bottom:1px solid var(--border);">
      <div style="width:6px;height:6px;border-radius:50%;background:${on ? 'var(--ok)' : 'var(--txt3)'};flex-shrink:0"></div>
      <div class="avatar">${ini(n)}</div>
      <div style="flex:1;min-width:0">
        <div style="font-size:12px;font-weight:600">${esc(n)} <span style="font-size:10px;color:var(--txt3)">Grade ${r.grade || '?'}</span></div>
        <div style="font-size:11px;color:var(--txt3);white-space:nowrap;overflow:hidden;text-overflow:ellipsis">${esc(r.question || '—')}</div>
      </div>
      <div style="font-size:10px;color:var(--txt3)">${timeAgo(r.time)}</div>
    </div>`;
  }).join('');
}

function renderUserList(id, users, on) {
  const el = document.getElementById(id);
  if (!el) return;
  if (!users.length) {
    el.innerHTML = '<div class="empty-state">No students online</div>';
    return;
  }
  el.innerHTML = users.slice(0, 6).map(u => `
    <div class="user-row">
      <div class="u-left">
        <div class="avatar">${ini(u.name)}</div>
        <div>
          <div class="u-name">${esc(u.name)}</div>
          <div class="u-sub">G${esc(u.grade)} · ${u.count} queries</div>
        </div>
      </div>
      ${on ? '<span class="badge bd-online"><span class="dot d-ok"></span>Online</span>' : `<span style="font-size:11px;color:var(--txt3)">${timeAgo(u.lastSeen)}</span>`}
    </div>`).join('');
}

function renderUsersTable(users) {
  const tb = document.getElementById('usersBody');
  if (!tb) return;
  if (!users.length) {
    tb.innerHTML = '<tr><td colspan="6" class="empty-state">No student profiles</td></tr>';
    return;
  }
  tb.innerHTML = [...users].sort((a, b) => b.count - a.count).map((u, i) => `<tr>
    <td style="color:var(--txt3)">${i + 1}</td>
    <td><div style="display:flex;align-items:center;gap:8px"><div class="avatar">${ini(u.name)}</div>${esc(u.name)}</div></td>
    <td>Grade ${esc(u.grade)}</td>
    <td>${isOnline(u.lastSeen) ? '<span class="badge bd-online"><span class="dot d-ok"></span>Active</span>' : '<span class="badge bd-offline">Offline</span>'}</td>
    <td><b>${u.count}</b></td>
    <td><button class="btn btn-dan" style="padding:4px 8px;font-size:10px" onclick="quickBanPrompt('${esc(u.name)}')"><i class="fas fa-ban"></i> Ban</button></td>
  </tr>`).join('');
}

function renderRecent(rows) {
  const tb = document.getElementById('recentBody');
  if (!tb) return;
  if (!rows.length) {
    tb.innerHTML = '<tr><td colspan="5" class="empty-state">No queries today</td></tr>';
    return;
  }
  tb.innerHTML = rows.map(r => `<tr>
    <td><b>${esc(r.user || '?')}</b></td>
    <td><span class="badge" style="background:var(--pri-dim);color:var(--pri)">G${esc(r.grade || '—')}</span></td>
    <td>${esc(r.subject || '—')}</td>
    <td><div class="q-cell">${esc(r.question || '—')}</div></td>
    <td style="color:var(--txt3)">${timeAgo(r.time)}</td>
  </tr>`).join('');
}

/* ── HISTORY VIEWER & FILTER PLOTS ── */
function renderHistory(rows) {
  const tb = document.getElementById('histBody');
  if (!tb) return;
  if (!rows.length) {
    tb.innerHTML = '<tr><td colspan="7" class="empty-state">No matching history queries</td></tr>';
    return;
  }
  tb.innerHTML = rows.map((r, i) => `<tr>
    <td style="color:var(--txt3)">${i + 1}</td>
    <td><b>${esc(r.user || '?')}</b></td>
    <td>Grade ${esc(r.grade || '—')}</td>
    <td>${esc(r.subject || '—')}</td>
    <td><div class="q-cell">${esc(r.question || '—')}</div></td>
    <td style="color:var(--txt3)">${timeAgo(r.time)}</td>
    <td><button class="btn btn-dan" style="padding:4px 8px;font-size:10px" onclick="quickBanPrompt('${esc(r.user || '')}')"><i class="fas fa-ban"></i> Ban</button></td>
  </tr>`).join('');
}

function filterHistory() {
  const q = document.getElementById('histSearch').value.toLowerCase();
  const gf = document.getElementById('histGrade').value;
  const sf = document.getElementById('histSubject').value;
  
  renderHistory([...allHistory.filter(r => 
    (!gf || String(r.grade) === gf) && 
    (!sf || r.subject === sf) && 
    (!q || (r.user || '').toLowerCase().includes(q) || (r.question || '').toLowerCase().includes(q))
  )].reverse());
}

function populateFilters(h) {
  const gs = [...new Set(h.map(r => r.grade).filter(Boolean))].sort((a, b) => a - b);
  const ss = [...new Set(h.map(r => r.subject).filter(Boolean))].sort();
  
  document.getElementById('histGrade').innerHTML = '<option value="">All Grades</option>' + gs.map(g => `<option value="${g}">Grade ${g}</option>`).join('');
  document.getElementById('histSubject').innerHTML = '<option value="">All Subjects</option>' + ss.map(s => `<option value="${s}">${esc(s)}</option>`).join('');
}

/* ── BLACKLIST MECHANICS ── */
function renderBlacklist(list) {
  const el = document.getElementById('blContainer');
  if (!el) return;
  if (!list || !list.length) {
    el.innerHTML = '<div class="empty-state">✅ Workspace clean, no active user bans active!</div>';
    return;
  }
  el.innerHTML = list.map(u => {
    const n = typeof u === 'string' ? u : u.name || String(u);
    return `<div class="user-row">
      <div class="u-left">
        <div class="avatar" style="background:var(--dan-dim);color:var(--dan)"><i class="fas fa-user-slash"></i></div>
        <div>
          <div class="u-name">${esc(n)}</div>
          <div class="u-sub" style="color:var(--dan)">Banned · Active (2 Hours)</div>
        </div>
      </div>
      <button class="btn btn-ok" onclick="quickUnbanPrompt('${esc(n)}')"><i class="fas fa-undo"></i> Lift Ban</button>
    </div>`;
  }).join('');
}

function renderSubjects(history, statsS) {
  const counts = { ...statsS };
  if (!Object.keys(counts).length) {
    history.forEach(r => {
      const s = r.subject || '?';
      counts[s] = (counts[s] || 0) + 1;
    });
  }
  const total = Object.values(counts).reduce((a, b) => a + b, 0);
  const entries = Object.entries(counts).sort((a, b) => b[1] - a[1]);
  
  const tagsEl = document.getElementById('subjTags');
  if (tagsEl) {
    tagsEl.innerHTML = entries.map(([s, c], i) => `
      <span class="subj-tag">
        <span>${esc(s)}</span>
        <span style="font-weight:700;color:${COLORS[i % COLORS.length]}">${c}</span>
        <span style="color:var(--txt3);font-size:9.5px">${pct(c, total)}%</span>
      </span>`).join('');
  }
}

/* ── LEADERBOARD METRICS ── */
async function loadLeaderboard() {
  const tb = document.getElementById('lbBody');
  if (!tb) return;
  tb.innerHTML = '<tr><td colspan="6" class="empty-state">Loading leaderboard ranks...</td></tr>';
  try {
    const res = await fetch(`${BACKEND}/leaderboard`);
    const d = await res.json();
    const list = d.leaderboard || [];
    
    if (!list.length) {
      tb.innerHTML = '<tr><td colspan="6" class="empty-state">No student scores yet</td></tr>';
      return;
    }
    const rankEmoji = (r) => r === 1 ? '🥇' : r === 2 ? '🥈' : r === 3 ? '🥉' : r;
    tb.innerHTML = list.map(e => `<tr>
      <td style="font-weight:800">${rankEmoji(e.rank)}</td>
      <td><div style="display:flex;align-items:center;gap:8px"><div class="avatar">${ini(e.name)}</div>${esc(e.name)}</div></td>
      <td style="color:var(--yel);font-weight:700">${e.xp || 0} XP</td>
      <td style="color:#F87171;font-weight:700">🔥${e.streak || 0}d</td>
      <td>${e.total_q || 0}</td>
      <td>Grade ${e.grade || '?'}</td>
    </tr>`).join('');
  } catch (e) {
    tb.innerHTML = '<tr><td colspan="6" class="empty-state">⚠️ Leaderboard fetch error</td></tr>';
  }
}

/* ── REPLACING BLOCKING PROMPTS WITH GLASSMODALS ── */
function quickBanPrompt(username) {
  openCustomModal(
    "🚫 Ban Suspected User?", 
    `Are you sure you want to blacklist "${username}" from the platform? This will suspend all their active Socratic solver access sessions for 2 hours.`,
    async () => {
      try {
        const res = await fetch(`${BACKEND}/admin/blacklist/add`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', ...getAuthHeaders() },
          body: JSON.stringify({ user_name: username })
        });
        if (res.ok) {
          showToast(`🚫 "${username}" has been blacklisted for 2 hours.`);
          loadData();
        } else {
          showToast('❌ Ban execution failed.');
        }
      } catch (e) {
        showToast('❌ Connection error.');
      }
    }
  );
}

function quickUnbanPrompt(username) {
  openCustomModal(
    "✅ Lift Blacklist Suspend?", 
    `Do you want to lift the blacklist suspend for student "${username}" immediately?`,
    async () => {
      try {
        const res = await fetch(`${BACKEND}/admin/blacklist/remove`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', ...getAuthHeaders() },
          body: JSON.stringify({ user_name: username })
        });
        if (res.ok) {
          showToast(`✅ "${username}" ban lifted. Access session restored.`);
          loadData();
        } else {
          showToast('❌ Action failed.');
        }
      } catch (e) {
        showToast('❌ Action error.');
      }
    }
  );
}

/* ── DATA INGESTION & VECTOR RAG ── */
async function loadPDFs() {
  const el = document.getElementById('pdfList');
  if (!el) return;
  el.innerHTML = '<div class="empty-state">Loading knowledge documents...</div>';
  try {
    const res = await fetch(`${BACKEND}/admin/pdfs`, { headers: getAuthHeaders() });
    if (!res.ok) return;
    const d = await res.json();
    const files = d.pdfs || [];
    
    if (!files.length) {
      el.innerHTML = '<div class="empty-state">No knowledge documents ingested in RAG hub yet</div>';
      return;
    }
    el.innerHTML = files.map(f => `
      <div class="pdf-row">
        <div style="display:flex;align-items:center;gap:10px">
          <span style="font-size:20px;color:var(--dan)"><i class="fas fa-file-pdf"></i></span>
          <span style="font-size:13px;font-weight:600">${esc(f)}</span>
        </div>
        <button class="btn btn-dan" onclick="deletePDFPrompt('${esc(f)}')"><i class="fas fa-trash"></i></button>
      </div>`).join('');
  } catch (e) {
    el.innerHTML = '<div class="empty-state">RAG index files fetch failed</div>';
  }
}

async function uploadPDF(e) {
  const files = [...e.target.files];
  if (!files.length) return;
  const prog = document.getElementById('uploadProgress');
  prog.style.display = 'block';
  
  let successCount = 0;
  let reportDetailsHtml = '';
  
  for (const f of files) {
    prog.textContent = '⏳ Vectorizing: ' + f.name;
    const fd = new FormData();
    fd.append('pdf', f);
    try {
      const res = await fetch(`${BACKEND}/admin/upload_pdf`, {
        method: 'POST',
        headers: getAuthHeaders(),
        body: fd
      });
      if (res.ok) {
        const data = await res.json();
        successCount++;
        const pineconeStatus = data.pinecone_indexed ? '🟢 Indexed & Active' : '🟡 Bypassed / Local Mode';
        reportDetailsHtml += `
          <div class="report-item" style="margin-top: 12px; padding: 14px; background: var(--card2); border-radius: 12px; border: 1px solid var(--border2);">
            <div style="font-weight: 700; color: var(--pri); font-size: 13.5px; display: flex; align-items: center; gap: 8px;">
              <i class="fas fa-file-pdf" style="color: var(--dan);"></i> ${esc(f.name)}
            </div>
            <div style="font-size: 11px; margin-top: 8px; color: var(--txt2); display: flex; flex-direction: column; gap: 4px;">
              <div>Vector Ingestion: <span style="color: var(--ok); font-weight: 600;">Success</span></div>
              <div>Pinecone Database: <span style="font-weight: 600;">${pineconeStatus}</span></div>
              <div>RAG Query Status: <span style="color: var(--acc); font-weight: 600;">Available for Student Queries</span></div>
            </div>
          </div>
        `;
      } else {
        reportDetailsHtml += `
          <div class="report-item" style="margin-top: 12px; padding: 14px; background: var(--card2); border-radius: 12px; border: 1px solid var(--dan-dim); border-left: 4px solid var(--dan);">
            <div style="font-weight: 700; color: var(--dan); font-size: 13.5px;">
              <i class="fas fa-exclamation-triangle"></i> ${esc(f.name)}
            </div>
            <div style="font-size: 11px; margin-top: 8px; color: var(--txt2);">
              <div>Ingestion Status: <span style="color: var(--dan); font-weight: 600;">Failed (HTTP ${res.status})</span></div>
            </div>
          </div>
        `;
      }
    } catch (err) {
      reportDetailsHtml += `
        <div class="report-item" style="margin-top: 12px; padding: 14px; background: var(--card2); border-radius: 12px; border: 1px solid var(--dan-dim); border-left: 4px solid var(--dan);">
          <div style="font-weight: 700; color: var(--dan); font-size: 13.5px;">
            <i class="fas fa-wifi"></i> ${esc(f.name)}
          </div>
          <div style="font-size: 11px; margin-top: 8px; color: var(--txt2);">
            <div>Ingestion Status: <span style="color: var(--dan); font-weight: 600;">Network Error (${esc(err.message)})</span></div>
          </div>
        </div>
      `;
    }
  }
  prog.style.display = 'none';
  loadPDFs();
  e.target.value = '';
  
  if (successCount > 0) {
    showToast('✅ Document ingestion process completed.');
    const bodyHtml = `
      <p style="font-size:12.5px; color:var(--txt2); line-height: 1.5;">
        The system has successfully completed the vectorization pipeline. Here is the ingestion report card:
      </p>
      ${reportDetailsHtml}
    `;
    openCustomModal("📦 Ingestion Report Card", bodyHtml, null, null, "Dismiss");
  } else {
    showToast('❌ Ingestion failed for all documents.');
  }
}

function deletePDFPrompt(filename) {
  openCustomModal(
    "🗑️ Delete Ingested Document?", 
    `Are you sure you want to permanently delete "${filename}" from the RAG search directory and wipe all associated Pinecone vector indices?`,
    async () => {
      try {
        const res = await fetch(`${BACKEND}/admin/pdfs/${encodeURIComponent(filename)}`, {
          method: 'DELETE',
          headers: getAuthHeaders()
        });
        if (res.ok) {
          showToast('🗑️ Document successfully deleted from RAG lookup.');
          loadPDFs();
        } else {
          showToast('❌ Deletion failed.');
        }
      } catch (e) {
        showToast('❌ Connection error.');
      }
    }
  );
}

/* ── EXPORT HISTORICAL CSV DATA ── */
function exportCSV() {
  if (!allHistory.length) {
    showToast('No logged history queries to export!');
    return;
  }
  const rows = [['#', 'Student', 'Grade', 'Subject', 'Socratic Question', 'Timestamp']];
  allHistory.forEach((r, i) => {
    rows.push([i + 1, r.user || '?', r.grade || '', r.subject || '', (r.question || '').replace(/"/g, "'"), r.time || '']);
  });
  const csv = rows.map(r => r.map(c => `"${c}"`).join(',')).join('\n');
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob(['\ufeff' + csv], { type: 'text/csv' }));
  a.download = `sanz_history_${new Date().toISOString().slice(0, 10)}.csv`;
  a.click();
  showToast('✅ CSV records log downloaded successfully.');
}

/* ── STUDENT SELF-REGISTERED PROFILES ── */
let allAccounts = [];
async function loadUserAccounts() {
  const tb = document.getElementById('accBody');
  if (tb) tb.innerHTML = '<tr><td colspan="12" class="empty-state">Loading accounts...</td></tr>';
  try {
    const res = await fetch(`${BACKEND}/admin/user-accounts`, { headers: getAuthHeaders() });
    if (!res.ok) {
      if (tb) tb.innerHTML = '<tr><td colspan="12" class="empty-state">Error</td></tr>';
      return;
    }
    const d = await res.json();
    allAccounts = d.accounts || [];
    document.getElementById('accBadge').textContent = allAccounts.length;
    renderAccounts(allAccounts);
  } catch (e) {
    if (tb) tb.innerHTML = '<tr><td colspan="12" class="empty-state">⚠️ ' + esc(e.message) + '</td></tr>';
  }
}

function renderAccounts(list) {
  const tb = document.getElementById('accBody');
  if (!tb) return;
  if (!list.length) {
    tb.innerHTML = '<tr><td colspan="12" class="empty-state">No student accounts registered yet</td></tr>';
    return;
  }
  tb.innerHTML = list.map((a, i) => {
    const contact = a.email || a.phone || '—';
    const active = a.active !== false;
    return `<tr>
      <td style="color:var(--txt3)">${i + 1}</td>
      <td>
        <div style="display:flex;align-items:center;gap:8px">
          <div class="avatar">${ini(a.full_name || a.username)}</div>
          <div>
            <div style="font-size:12px;font-weight:700">${esc(a.username)}</div>
            <div style="font-size:9.5px;color:var(--txt3)">Active ${timeAgo(a.last_login)}</div>
          </div>
        </div>
      </td>
      <td style="font-weight:600">${esc(a.full_name || '—')}</td>
      <td><span class="badge" style="background:var(--pri-dim);color:var(--pri)">Grade ${esc(a.grade || '?')}</span></td>
      <td>${a.age || '—'}</td>
      <td style="color:var(--txt2);font-size:11.5px">${esc(contact)}</td>
      <td style="color:var(--yel);font-weight:700">⚡${a.xp || 0}</td>
      <td>${a.total_questions || 0}</td>
      <td style="color:var(--acc)">${(a.tokens_today || 0).toLocaleString()}</td>
      <td style="color:var(--txt3);font-size:11.5px">${timeAgo(a.created)}</td>
      <td>${active ? '<span class="badge bd-online">Active</span>' : '<span class="badge bd-ban">Disabled</span>'}</td>
      <td>
        <div style="display:flex;gap:6px;">
          ${active
            ? `<button class="btn btn-dan" style="padding:4px 8px;font-size:10px" onclick="disableUserPrompt('${esc(a.username)}')"><i class="fas fa-user-xmark"></i></button>`
            : `<button class="btn btn-ok" style="padding:4px 8px;font-size:10px" onclick="enableUserPrompt('${esc(a.username)}')"><i class="fas fa-user-check"></i></button>`
          }
          <button class="btn" style="padding:4px 8px;font-size:10px" onclick="viewUser('${esc(a.username)}')"><i class="fas fa-eye"></i></button>
        </div>
      </td>
    </tr>`;
  }).join('');
}

function filterAccounts() {
  const q = document.getElementById('accSearch').value.toLowerCase();
  renderAccounts(allAccounts.filter(a =>
    (a.username || '').toLowerCase().includes(q) ||
    (a.full_name || '').toLowerCase().includes(q) ||
    (a.email || '').toLowerCase().includes(q) ||
    (a.grade || '').includes(q)
  ));
}

function disableUserPrompt(username) {
  openCustomModal(
    "🚫 Disable Student Account?", 
    `Are you sure you want to temporarily disable student account "${username}"? They will not be able to log in to the student portal.`,
    async () => {
      try {
        const res = await fetch(`${BACKEND}/admin/user-accounts/disable`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', ...getAuthHeaders() },
          body: JSON.stringify({ user_name: username })
        });
        if (res.ok) {
          showToast(`🚫 Account "${username}" disabled.`);
          loadUserAccounts();
        } else {
          showToast('❌ Action failed.');
        }
      } catch (e) {
        showToast('❌ Connection error.');
      }
    }
  );
}

function enableUserPrompt(username) {
  openCustomModal(
    "✅ Re-enable Student Account?", 
    `Do you want to re-enable student account "${username}" immediately?`,
    async () => {
      try {
        const res = await fetch(`${BACKEND}/admin/user-accounts/enable`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', ...getAuthHeaders() },
          body: JSON.stringify({ user_name: username })
        });
        if (res.ok) {
          showToast(`✅ Account "${username}" successfully restored.`);
          loadUserAccounts();
        } else {
          showToast('❌ Action failed.');
        }
      } catch (e) {
        showToast('❌ Connection error.');
      }
    }
  );
}

function viewUser(username) {
  const a = allAccounts.find(x => x.username === username);
  if (!a) return;
  showToast(`${a.full_name} · G${a.grade} · XP:${a.xp} · Queries:${a.total_questions} · Contact: ${a.email || a.phone || 'No contact'}`, 5000);
}

function showToast(m) {
  const t = document.getElementById('toastAdm');
  if (!t) return;
  t.textContent = m;
  t.className = 'toast-adm show';
  setTimeout(() => t.classList.remove('show'), 3500);
}
