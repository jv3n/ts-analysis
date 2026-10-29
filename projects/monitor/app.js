"use strict";

const POLL_MS = 2000;
const SESSION = 390;
const BASELINES = ["flat", "random-walk"];
const W = 1000, H = 380, M = { top: 12, right: 16, bottom: 28, left: 60 };

const $ = (id) => document.getElementById(id);
const state = { fetch: null, replay: null, view: null, compare: null, statusKey: "" };

const esc = (value) =>
  String(value).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const pct = (x) => `${x >= 0 ? "+" : ""}${(x * 100).toFixed(1)} %`;
const price = (p) => (p == null ? "–" : p.toFixed(p < 1 ? 4 : 2));
const clock = (at) => {
  const minutes = 9 * 60 + 30 + at;
  return `${String(Math.floor(minutes / 60)).padStart(2, "0")}:${String(minutes % 60).padStart(2, "0")}`;
};
const ago = (iso) => {
  const s = Math.max(0, Math.round((Date.now() - Date.parse(iso)) / 1000));
  return s < 60 ? `${s} s ago` : s < 3600 ? `${Math.round(s / 60)} min ago` : new Date(iso).toLocaleString();
};

async function get(path) {
  const response = await fetch(path, { cache: "no-store" });
  const body = await response.json();
  if (!response.ok) throw new Error(body.error);
  return body;
}

// Progress

function progressCard(title, status, inner) {
  if (!status) return `<h3>${title}</h3><p class="muted">Never run on this data folder.</p>`;
  const share = status.state === "done" ? 1 : status.days ? (status.day_number - 1 + (inner.share || 0)) / status.days : 0;
  return `
    <h3>${title} <span class="badge ${esc(status.state)}">${esc(status.state)}</span></h3>
    <p>${esc(status.start)} → ${esc(status.end)}${status.models ? ` · ${status.models.map(esc).join(", ")}` : ""}</p>
    <div class="bar" role="progressbar" aria-valuenow="${Math.round(share * 100)}" aria-valuemin="0" aria-valuemax="100">
      <span style="width:${(share * 100).toFixed(1)}%"></span>
    </div>
    <p class="num">day ${status.day_number ?? 0}/${status.days} ${esc(status.day ?? "")}${inner.text ? ` · ${inner.text}` : ""}</p>
    <p class="muted">started ${ago(status.started)} · updated ${ago(status.updated)}</p>
    ${status.note ? `<p>${esc(status.note)}</p>` : ""}
    ${status.error ? `<p class="error">${esc(status.error)}</p>` : ""}`;
}

function renderProgress(statuses) {
  const f = statuses.fetch, r = statuses.replay;
  $("job-fetch").innerHTML = progressCard("Fetch", f, {
    share: f && f.tickers ? f.ticker_number / f.tickers : 0,
    text: f && f.ticker ? `ticker ${f.ticker_number}/${f.tickers} ${esc(f.ticker)}` : "",
  });
  $("job-replay").innerHTML = progressCard("Replay", r, {
    share: r && r.cuts ? r.cut_number / r.cuts : 0,
    text: r && r.cut ? `cut ${r.cut_number}/${r.cuts} ${esc(r.cut)}` : "",
  });
}

// Fetch

function renderFetch(summary) {
  const days = summary.days;
  if (!days.length) {
    $("fetch").innerHTML = `<p class="muted">No day stored yet: run <code>ts-fetch</code>.</p>`;
    return;
  }
  const tickerDays = days.reduce((n, d) => n + d.picked.length, 0);
  const stored = days.reduce((n, d) => n + Object.values(d.bars).filter((b) => b != null).length, 0);
  const rows = [...days].reverse().map((d) => {
    const chips = d.picked
      .map((t) => (d.bars[t] == null
        ? `<span class="chip todo" title="bars not fetched yet">${esc(t)} –</span>`
        : `<span class="chip" title="${d.bars[t]} 1-min bars">${esc(t)} ${d.bars[t]}</span>`))
      .join("");
    const top = [...d.candidates].sort((a, b) => b.gap - a.gap)[0];
    return `<tr>
      <td class="num">${esc(d.day)}</td>
      <td class="num">${d.candidates.length}</td>
      <td class="num">${top ? `${esc(top.ticker)} ${pct(top.gap)}` : "–"}</td>
      <td class="num">${d.picked.length}</td>
      <td><div class="chips">${chips}</div></td>
    </tr>`;
  });
  $("fetch").innerHTML = `
    <p class="summary">${days.length} days from ${esc(days[0].day)} to ${esc(days.at(-1).day)} ·
      ${stored}/${tickerDays} ticker-days stored
      ${summary.missing.length ? ` · <span class="miss">missing: ${summary.missing.map(esc).join(", ")}</span>` : ""}
      ${summary.holidays.length ? ` · holidays: ${summary.holidays.map(esc).join(", ")}` : ""}</p>
    <div class="table-wrap"><table>
      <thead><tr><th>Day</th><th class="num">Candidates</th><th class="num">Biggest gap</th><th class="num">Picked</th><th>Bars per ticker</th></tr></thead>
      <tbody>${rows.join("")}</tbody>
    </table></div>`;
}

// Replay

function renderReplay(summary) {
  const models = summary.models;
  if (!models.length) {
    $("replay").innerHTML = `<p class="muted">No day replayed yet: run <code>ts-replay</code>.</p>`;
    return;
  }
  const byModel = Object.fromEntries(models.map((m) => [m.model, Object.fromEntries(m.days.map((d) => [d.day, d]))]));
  const days = [...new Set(models.flatMap((m) => m.days.map((d) => d.day)))].sort().reverse();
  const head = models.map((m) => `<th class="num">${esc(m.model)}</th>`).join("");
  const rows = days.map((day) => `<tr><td class="num">${esc(day)}</td>${models
    .map((m) => {
      const d = byModel[m.model][day];
      return d
        ? `<td class="num"><a href="#" data-model="${esc(m.model)}" data-day="${esc(day)}" title="${d.tickers.length} tickers, ${d.ms == null ? "–" : d.ms.toFixed(1)} ms a forecast">${d.forecasts}</a></td>`
        : `<td class="num muted">–</td>`;
    })
    .join("")}</tr>`);
  const total = models.map((m) => {
    const n = m.days.reduce((s, d) => s + d.forecasts, 0);
    const ms = n ? m.days.reduce((s, d) => s + (d.ms ?? 0) * d.forecasts, 0) / n : null;
    return `<td class="num"><b>${m.days.length}</b> days · ${n}<br><span class="muted">${ms == null ? "–" : ms.toFixed(1)} ms each</span></td>`;
  });
  $("replay").innerHTML = `
    <p class="summary">Rows forecast per day and model (one per ticker, cut and horizon). Click a count to open it.</p>
    <div class="table-wrap"><table>
      <thead><tr><th>Day</th>${head}</tr></thead>
      <tbody><tr><td>Total</td>${total.join("")}</tr>${rows.join("")}</tbody>
    </table></div>`;
}

$("replay").addEventListener("click", (event) => {
  const link = event.target.closest("a[data-model]");
  if (!link) return;
  event.preventDefault();
  $("model").value = link.dataset.model;
  fillDays(link.dataset.day);
  loadView();
  $("pick").scrollIntoView({ behavior: "smooth" });
});

// Forecast selectors

function setOptions(select, values, wanted) {
  const keep = values.includes(wanted) ? wanted : values.includes(select.value) ? select.value : values[0];
  select.innerHTML = values.map((v) => `<option>${esc(v)}</option>`).join("");
  if (keep !== undefined) select.value = keep;
}

function modelDays(model) {
  return state.replay.models.find((m) => m.model === model)?.days ?? [];
}

function fillModels() {
  const names = state.replay.models.map((m) => m.model);
  const preferred = names.find((n) => !BASELINES.includes(n));
  setOptions($("model"), names, $("model").value || preferred);
  fillDays();
}

function fillDays(wanted) {
  const days = modelDays($("model").value).map((d) => d.day).reverse();
  setOptions($("day"), days, wanted);
  fillTickers();
}

function fillTickers() {
  const day = modelDays($("model").value).find((d) => d.day === $("day").value);
  setOptions($("ticker"), day ? day.tickers : []);
}

async function loadView() {
  const model = $("model").value, day = $("day").value, ticker = $("ticker").value;
  if (!model || !day || !ticker) {
    state.view = null;
    return drawForecast();
  }
  const query = (m) => `/api/forecast?model=${encodeURIComponent(m)}&day=${day}&ticker=${encodeURIComponent(ticker)}`;
  const cut = state.view?.cuts[$("cut").value]?.cut;
  const withCompare = $("compare").checked && model !== "random-walk"
    && modelDays("random-walk").some((d) => d.day === day && d.tickers.includes(ticker));
  try {
    [state.view, state.compare] = await Promise.all([get(query(model)), withCompare ? get(query("random-walk")) : null]);
  } catch (error) {
    state.view = null;
    $("chart").innerHTML = `<p class="empty">${esc(error.message)}</p>`;
    return;
  }
  const index = Math.max(0, state.view.cuts.findIndex((c) => c.cut === cut));
  $("cut").max = state.view.cuts.length - 1;
  $("cut").value = index;
  drawForecast();
}

$("model").addEventListener("change", () => { fillDays(); loadView(); });
$("day").addEventListener("change", () => { fillTickers(); loadView(); });
$("ticker").addEventListener("change", loadView);
$("compare").addEventListener("change", loadView);
$("cut").addEventListener("input", drawForecast);
$("prev").addEventListener("click", () => { $("cut").value = Number($("cut").value) - 1; drawForecast(); });
$("next").addEventListener("click", () => { $("cut").value = Number($("cut").value) + 1; drawForecast(); });

// Forecast chart

function ticks(lo, hi, count) {
  const raw = (hi - lo) / count, mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((k) => k * mag).find((s) => s >= raw);
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(v);
  return out;
}

function drawForecast() {
  const view = state.view;
  if (!view) {
    $("chart").innerHTML = `<p class="empty">Pick a replayed day to see its forecasts.</p>`;
    $("legend").innerHTML = $("band-table").innerHTML = $("cut-label").textContent = "";
    return;
  }
  const cut = view.cuts[$("cut").value];
  const compareCut = state.compare?.cuts.find((c) => c.cut === cut.cut);
  $("cut-label").textContent = cut.cut;

  const known = view.prices.filter((p) => p != null);
  let lo = Math.min(...known), hi = Math.max(...known);
  const pad = (hi - lo) * 0.08 || hi * 0.05;
  lo -= pad; hi += pad;
  const x = (at) => M.left + (at / SESSION) * (W - M.left - M.right);
  const y = (p) => M.top + ((hi - p) / (hi - lo)) * (H - M.top - M.bottom);
  const path = (points) => points.filter(([, p]) => p != null).map(([at, p], i) => `${i ? "L" : "M"}${x(at).toFixed(1)},${y(p).toFixed(1)}`).join("");
  const series = (from, to) => view.prices.slice(from, to + 1).map((p, i) => [from + i, p]);
  const start = [cut.at, cut.last];
  const band = (c, upper, lower) => path([start, ...c.horizons.map((h) => [h.at, h[upper]]), ...c.horizons.map((h) => [h.at, h[lower]]).reverse()]) + "Z";
  const line = (c, q) => path([start, ...c.horizons.map((h) => [h.at, h[q]])]);

  const yTicks = ticks(lo, hi, 5).map((v) => `<line class="gridline" x1="${M.left}" x2="${W - M.right}" y1="${y(v)}" y2="${y(v)}"/><text x="${M.left - 8}" y="${y(v) + 4}" text-anchor="end">${price(v)}</text>`);
  const xTicks = [0, 30, 90, 150, 210, 270, 330, 390].map((at) => `<text x="${x(at)}" y="${H - 8}" text-anchor="middle">${clock(at)}</text>`);
  const actual = cut.horizons.map((h) => view.prices[h.at] == null ? "" : `<circle class="actual" cx="${x(h.at)}" cy="${y(view.prices[h.at])}" r="4.5"/>`);

  $("chart").innerHTML = `
    <svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(view.ticker)} on ${esc(view.day)}, ${esc(view.model)} forecast at ${esc(cut.cut)}">
      <defs><clipPath id="plot"><rect x="${M.left}" y="${M.top}" width="${W - M.left - M.right}" height="${H - M.top - M.bottom}"/></clipPath></defs>
      <g class="axis">${yTicks.join("")}${xTicks.join("")}</g>
      <g clip-path="url(#plot)">
        <path class="band-outer" d="${band(cut, "q90", "q10")}"/>
        <path class="band-inner" d="${band(cut, "q75", "q25")}"/>
        <path class="median" d="${line(cut, "q50")}"/>
        ${compareCut ? `<path class="compare" d="${line(compareCut, "q10")}"/><path class="compare" d="${line(compareCut, "q90")}"/>` : ""}
        <path class="price-after" d="${path(series(cut.at, SESSION))}"/>
        <path class="price-seen" d="${path(series(0, cut.at))}"/>
        ${actual.join("")}
      </g>
      <line class="cut-line" x1="${x(cut.at)}" x2="${x(cut.at)}" y1="${M.top}" y2="${H - M.bottom}"/>
      <g id="hover" hidden><line class="crosshair" y1="${M.top}" y2="${H - M.bottom}"/><circle class="hover-dot" r="4"/></g>
      <rect id="hit" x="${M.left}" y="${M.top}" width="${W - M.left - M.right}" height="${H - M.top - M.bottom}" fill="transparent"/>
    </svg>`;
  bindHover(view, cut, x, y);

  $("legend").innerHTML = `
    <span><i style="background:var(--text-primary);height:2px"></i>price seen at the cut</span>
    <span><i style="background:var(--text-muted);height:2px"></i>price after it</span>
    <span><i style="background:var(--series-1);opacity:.45"></i>${esc(view.model)} 10–90 % and 25–75 %</span>
    <span><i style="background:var(--series-1);height:2px"></i>median</span>
    ${compareCut ? `<span><i style="border-top:2px dashed var(--series-2);height:0"></i>random walk 10–90 %</span>` : ""}`;

  const rows = cut.horizons.map((h) => {
    const real = view.prices[h.at];
    const inside = real != null && h.q10 <= real && real <= h.q90;
    const rw = compareCut?.horizons.find((c) => c.horizon === h.horizon);
    const rwInside = rw && real != null && rw.q10 <= real && real <= rw.q90;
    return `<tr>
      <td class="num">+${h.horizon} min</td><td class="num">${clock(h.at)}</td>
      <td class="num">${price(h.q10)}</td><td class="num">${price(h.q50)} <span class="muted">${pct(h.q50 / cut.last - 1)}</span></td><td class="num">${price(h.q90)}</td>
      <td class="num">${price(real)} <span class="muted">${real == null ? "" : pct(real / cut.last - 1)}</span></td>
      <td class="${inside ? "hit" : "miss"}">${inside ? "✓ in band" : "✗ outside"}</td>
      ${compareCut ? `<td class="${rwInside ? "hit" : "miss"}">${rw ? (rwInside ? "✓" : "✗") : "–"}</td>` : ""}
    </tr>`;
  });
  $("band-table").innerHTML = `
    <div class="table-wrap"><table>
      <thead><tr><th>Horizon</th><th>At</th><th class="num">q10</th><th class="num">Median</th><th class="num">q90</th><th class="num">Real</th><th>10–90 %</th>${compareCut ? "<th>Random walk</th>" : ""}</tr></thead>
      <tbody>${rows.join("")}</tbody>
    </table></div>
    <p class="muted">Last price at ${esc(cut.cut)}: ${price(cut.last)}.</p>`;
}

function bindHover(view, cut, x, y) {
  const svg = $("chart").querySelector("svg"), hover = svg.querySelector("#hover"), tip = $("tooltip");
  svg.querySelector("#hit").addEventListener("mousemove", (event) => {
    const point = svg.createSVGPoint();
    point.x = event.clientX; point.y = event.clientY;
    const local = point.matrixTransform(svg.getScreenCTM().inverse());
    const at = Math.round(((local.x - M.left) / (W - M.left - M.right)) * SESSION);
    const p = view.prices[Math.min(SESSION, Math.max(0, at))];
    if (p == null) return;
    hover.hidden = false;
    hover.querySelector("line").setAttribute("x1", x(at));
    hover.querySelector("line").setAttribute("x2", x(at));
    hover.querySelector("circle").setAttribute("cx", x(at));
    hover.querySelector("circle").setAttribute("cy", y(p));
    const h = cut.horizons.find((h) => Math.abs(h.at - at) <= 2);
    tip.innerHTML = `${clock(at)} · <b>${price(p)}</b>${h ? `<br>+${h.horizon} min band ${price(h.q10)} – ${price(h.q90)}, median ${price(h.q50)}` : ""}`;
    tip.hidden = false;
    tip.style.left = `${Math.min(event.clientX + 14, innerWidth - tip.offsetWidth - 8)}px`;
    tip.style.top = `${event.clientY + 14}px`;
  });
  svg.querySelector("#hit").addEventListener("mouseleave", () => { hover.hidden = true; tip.hidden = true; });
}

// Polling: the summaries only change when a job moves to another day or ends.

async function refreshSummaries() {
  [state.fetch, state.replay] = await Promise.all([get("/api/fetch"), get("/api/replay")]);
  renderFetch(state.fetch);
  renderReplay(state.replay);
  const before = [$("model").value, $("day").value, $("ticker").value].join();
  fillModels();
  if (!state.view || before !== [$("model").value, $("day").value, $("ticker").value].join()) await loadView();
}

async function poll() {
  try {
    const statuses = await get("/api/status");
    renderProgress(statuses);
    const key = JSON.stringify(Object.values(statuses).map((s) => s && [s.state, s.day, s.ticker_number]));
    if (key !== state.statusKey) {
      state.statusKey = key;
      await refreshSummaries();
    }
    $("refreshed").textContent = `updated ${new Date().toLocaleTimeString()}`;
  } catch (error) {
    $("refreshed").textContent = `monitor unreachable: ${error.message}`;
  }
  setTimeout(poll, POLL_MS);
}

poll();
