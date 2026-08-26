/* Lumina UI — vanilla JS, no build step. Talks to the local FastAPI server. */
"use strict";

const $ = (id) => document.getElementById(id);
const state = {
  query: "",
  results: [],
  lastSearchResults: [],
  viewingSimilarOf: null,
  selected: null,
};

function imgUrl(path) {
  return `/api/image?path=${encodeURIComponent(path)}`;
}

async function api(path, options) {
  const res = await fetch(path, options);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail || detail;
    } catch (_) {}
    throw new Error(`${res.status}: ${detail}`);
  }
  return res.json();
}

function relativeDay(iso) {
  if (!iso) return "";
  const then = new Date(iso);
  if (isNaN(then)) return iso.slice(0, 10);
  const days = Math.floor((Date.now() - then.getTime()) / 86400000);
  if (days <= 0) return "today";
  if (days === 1) return "yesterday";
  if (days < 7) return `${days}d ago`;
  if (days < 60) return `${Math.floor(days / 7)}w ago`;
  return then.toISOString().slice(0, 10);
}

/* ---------------- status sidebar ---------------- */

async function refreshStatus() {
  try {
    const st = await api("/api/status");
    $("st-points").textContent = st.points >= 0 ? st.points.toLocaleString() : "—";
    const dot = $("st-watching");
    dot.className = "dot " + (st.watching ? "on" : "off");
    $("st-watchtext").textContent = st.watching
      ? "Watching for changes"
      : st.reindex_running
        ? "Reindexing…"
        : "Watcher idle";
    $("st-folders").textContent = (st.folders || []).join("  ·  ");
    const ls = st.last_scan || {};
    $("st-lastscan").textContent = ls.indexed !== undefined
      ? `Last scan: +${ls.indexed} new · ${ls.unchanged ?? "?"} unchanged`
      : "";
  } catch (err) {
    $("st-watchtext").textContent = "Server unreachable";
    $("st-watching").className = "dot off";
  }
}

$("reindex").addEventListener("click", async () => {
  try {
    await api("/api/reindex", { method: "POST" });
    refreshStatus();
  } catch (err) {
    console.error(err);
  }
});

/* ---------------- rendering ---------------- */

function card(result, opts = {}) {
  const div = document.createElement("div");
  div.className = "card";
  div.title = result.caption || result.ocr_text || "";

  const img = document.createElement("img");
  img.loading = "lazy";
  img.src = imgUrl(result.file_path);
  div.appendChild(img);

  const label = (result.caption || result.ocr_text || "").slice(0, 90);
  if (label) {
    const cap = document.createElement("div");
    cap.className = "cap";
    cap.textContent = label;
    div.appendChild(cap);
  }

  const overlay = document.createElement("div");
  overlay.className = "overlay";
  overlay.innerHTML = `<span class="date-inline">${relativeDay(result.created_at)}</span>`;
  const actions = document.createElement("span");

  const similarBtn = document.createElement("button");
  similarBtn.className = "ghost small-btn";
  similarBtn.textContent = "⤳ Similar";
  similarBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    findSimilar(result.id);
  });
  actions.appendChild(similarBtn);

  const revealBtn = document.createElement("button");
  revealBtn.className = "ghost small-btn";
  revealBtn.textContent = "📂";
  revealBtn.title = "Reveal in Finder";
  revealBtn.addEventListener("click", async (e) => {
    e.stopPropagation();
    revealBtn.textContent = "…";
    try {
      await api("/api/open", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ path: result.file_path }),
      });
    } catch (err) {
      console.error(err);
    }
    revealBtn.textContent = "📂";
  });
  actions.appendChild(revealBtn);
  overlay.appendChild(actions);
  div.appendChild(overlay);

  const dateRow = document.createElement("div");
  dateRow.className = "date";
  dateRow.textContent =
    relativeDay(result.created_at) +
    (result.media_type ? ` · ${result.media_type}` : "") +
    (opts.showScore ? ` · ${result.score.toFixed(3)}` : "");
  div.appendChild(dateRow);

  div.addEventListener("click", () => openLightbox(result));
  return div;
}

function renderResults(results, { showScore = false } = {}) {
  const grid = $("grid");
  grid.innerHTML = "";
  results.forEach((r) => grid.appendChild(card(r, { showScore })));
  $("empty").classList.toggle("hidden", results.length > 0);
}

function showEmpty(message) {
  renderResults([]);
  $("emptymsg").textContent = message;
  $("empty").classList.remove("hidden");
}

/* ---------------- search ---------------- */

let debounceTimer = null;

async function doSearch(query) {
  state.query = query;
  state.viewingSimilarOf = null;
  $("similarbar").classList.add("hidden");
  if (!query.trim()) {
    showEmpty("Type what you remember — captions, on-screen text, and visuals are all searched.");
    return;
  }
  try {
    const body = await api(`/api/search?q=${encodeURIComponent(query)}&limit=48`);
    state.lastSearchResults = body.results;
    const chip = $("datechip");
    if (body.date_filter) {
      const [from] = body.date_filter;
      chip.textContent = `📅 ${from.slice(0, 10)} onward`;
      chip.classList.remove("hidden");
    } else {
      chip.classList.add("hidden");
    }
    if (!body.results.length) {
      showEmpty(`Nothing matched “${body.query}”. Try different wording or widen the time range.`);
      return;
    }
    renderResults(body.results);
  } catch (err) {
    showEmpty(`Search failed — ${err.message}`);
  }
}

$("search").addEventListener("input", (e) => {
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(() => doSearch(e.target.value), 350);
});
$("search").addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    clearTimeout(debounceTimer);
    doSearch(e.target.value);
  }
});

/* ---------------- similar ---------------- */

async function findSimilar(pointId) {
  try {
    const body = await api(`/api/similar/${pointId}?limit=24`);
    state.viewingSimilarOf = pointId;
    $("similarbar").classList.remove("hidden");
    renderResults(body.results, { showScore: true });
    window.scrollTo({ top: 0 });
  } catch (err) {
    showEmpty(`Similar search failed — ${err.message}`);
  }
}

$("similarback").addEventListener("click", () => {
  state.viewingSimilarOf = null;
  $("similarbar").classList.add("hidden");
  renderResults(state.lastSearchResults);
});

/* ---------------- lightbox ---------------- */

function openLightbox(result) {
  state.selected = result;
  $("lb-img").src = imgUrl(result.file_path);
  $("lb-caption").textContent = result.caption || "(no caption)";
  const bits = [relativeDay(result.created_at)];
  if (result.media_type) bits.push(result.media_type);
  bits.push(`score ${result.score?.toFixed?.(3) ?? result.score}`);
  $("lb-sub").textContent = bits.join(" · ");
  $("lightbox").classList.remove("hidden");
}

function closeLightbox() {
  $("lightbox").classList.add("hidden");
  state.selected = null;
}

$("lb-close").addEventListener("click", closeLightbox);
$("lightbox").addEventListener("click", (e) => {
  if (e.target === $("lightbox")) closeLightbox();
});
$("lb-reveal").addEventListener("click", async () => {
  if (!state.selected) return;
  await api("/api/open", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path: state.selected.file_path }),
  }).catch(console.error);
});
$("lb-similar").addEventListener("click", () => {
  if (!state.selected) return;
  closeLightbox();
  findSimilar(state.selected.id);
});
$("lb-copy").addEventListener("click", async (e) => {
  if (!state.selected) return;
  const btn = e.target;
  btn.textContent = "…";
  try {
    const blob = await (await fetch(imgUrl(state.selected.file_path))).blob();
    await navigator.clipboard.write([new ClipboardItem({ [blob.type]: blob })]);
    btn.textContent = "✓ copied";
  } catch (_) {
    btn.textContent = "copy unsupported";
  }
  setTimeout(() => (btn.textContent = "⧉ Copy image"), 1500);
});

/* ---------------- duplicate finder ---------------- */

$("dupbtn").addEventListener("click", () => {
  $("duplicates").classList.toggle("hidden");
});

const dropzone = $("dropzone");
dropzone.addEventListener("click", () => $("dupfile").click());
["dragover", "dragenter"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => {
    e.preventDefault();
    dropzone.classList.add("drag");
  })
);
["dragleave", "drop"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => {
    e.preventDefault();
    dropzone.classList.remove("drag");
  })
);
dropzone.addEventListener("drop", (e) => {
  const file = e.dataTransfer.files[0];
  if (file) checkDuplicate(file);
});
$("dupfile").addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (file) checkDuplicate(file);
});

async function checkDuplicate(file) {
  const verdict = $("dupverdict");
  verdict.classList.remove("hidden", "dup", "unique");
  verdict.textContent = "Checking…";
  $("dupresults").innerHTML = "";
  try {
    const form = new FormData();
    form.append("probe", file, file.name);
    const body = await api("/api/duplicate", { method: "POST", body: form });
    verdict.classList.add(body.duplicate ? "dup" : "unique");
    verdict.textContent = body.duplicate
      ? "⧉ Already in your library — closest match shown below."
      : "✓ No duplicates found — this looks unique.";
    body.results.slice(0, 6).forEach((r) => {
      const c = card(r, { showScore: true });
      if (r.phash_distance === 0) c.style.outline = "2px solid var(--warn)";
      $("dupresults").appendChild(c);
    });
  } catch (err) {
    verdict.classList.add("dup");
    verdict.textContent = `Check failed — ${err.message}`;
  }
}

/* ---------------- global keys & boot ---------------- */

document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") {
    if (!$("lightbox").classList.contains("hidden")) return closeLightbox();
    $("search").value = "";
    doSearch("");
  }
  if (e.key === "/" && document.activeElement !== $("search")) {
    e.preventDefault();
    $("search").focus();
  }
});

refreshStatus();
setInterval(refreshStatus, 5000);
showEmpty("Type what you remember — captions, on-screen text, and visuals are all searched.");
