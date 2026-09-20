/* Vanilla JS: theme, nav, scrollspy, table sort/filter, copy buttons, lightbox, back-to-top. */
(function () {
  "use strict";
  var root = document.documentElement;
  root.classList.add("js");

  /* theme */
  var themeBtn = document.getElementById("theme-toggle");
  function currentTheme() {
    var t = root.getAttribute("data-theme");
    if (t) return t;
    return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }
  function syncThemeBtn() {
    if (!themeBtn) return;
    var dark = currentTheme() === "dark";
    themeBtn.textContent = dark ? "Light" : "Dark";
    themeBtn.setAttribute("aria-label", dark ? "Switch to light theme" : "Switch to dark theme");
  }
  try { var saved = localStorage.getItem("fbell-theme"); if (saved) root.setAttribute("data-theme", saved); } catch (e) {}
  syncThemeBtn();
  if (themeBtn) themeBtn.addEventListener("click", function () {
    var next = currentTheme() === "dark" ? "light" : "dark";
    root.setAttribute("data-theme", next);
    try { localStorage.setItem("fbell-theme", next); } catch (e) {}
    syncThemeBtn();
  });

  /* mobile nav */
  var toggle = document.getElementById("nav-toggle");
  var links = document.getElementById("nav-links");
  if (toggle && links) {
    toggle.addEventListener("click", function () {
      var open = links.classList.toggle("open");
      toggle.setAttribute("aria-expanded", open ? "true" : "false");
    });
    links.addEventListener("click", function (e) {
      if (e.target.closest("a")) { links.classList.remove("open"); toggle.setAttribute("aria-expanded", "false"); }
    });
    document.addEventListener("keydown", function (e) {
      if (e.key === "Escape" && links.classList.contains("open")) { links.classList.remove("open"); toggle.setAttribute("aria-expanded", "false"); toggle.focus(); }
    });
  }

  /* scrollspy */
  if ("IntersectionObserver" in window && links) {
    var map = {};
    Array.prototype.forEach.call(links.querySelectorAll("a[href^='#']"), function (a) { map[a.getAttribute("href").slice(1)] = a; });
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        if (en.isIntersecting && map[en.target.id]) {
          Object.keys(map).forEach(function (k) { map[k].removeAttribute("aria-current"); });
          map[en.target.id].setAttribute("aria-current", "true");
        }
      });
    }, { rootMargin: "-20% 0px -70% 0px" });
    Object.keys(map).forEach(function (id) { var el = document.getElementById(id); if (el) io.observe(el); });
  }

  /* back to top */
  var top = document.getElementById("to-top");
  if (top) {
    var onScroll = function () { top.classList.toggle("show", window.scrollY > 900); };
    window.addEventListener("scroll", onScroll, { passive: true }); onScroll();
    top.addEventListener("click", function () { window.scrollTo({ top: 0 }); });
  }

  /* copy buttons: <button data-copy="#id"> */
  Array.prototype.forEach.call(document.querySelectorAll("[data-copy]"), function (btn) {
    btn.addEventListener("click", function () {
      var src = document.querySelector(btn.getAttribute("data-copy"));
      if (!src) return;
      var text = src.textContent.replace(/^\n/, "");
      var done = function () { var o = btn.textContent; btn.textContent = "Copied"; setTimeout(function () { btn.textContent = o; }, 1600); };
      if (navigator.clipboard && window.isSecureContext) {
        navigator.clipboard.writeText(text).then(done, fallback);
      } else fallback();
      function fallback() {
        var ta = document.createElement("textarea"); ta.value = text; ta.setAttribute("readonly", "");
        ta.style.position = "fixed"; ta.style.opacity = "0"; document.body.appendChild(ta); ta.select();
        try { document.execCommand("copy"); done(); } catch (e) { btn.textContent = "Select and copy"; }
        document.body.removeChild(ta);
      }
    });
  });

  /* lightbox */
  var dlg = document.getElementById("lightbox");
  if (dlg && typeof dlg.showModal === "function") {
    var dImg = dlg.querySelector("img"), dCap = dlg.querySelector(".cap"), dOpen = dlg.querySelector("a.open");
    document.addEventListener("click", function (e) {
      var a = e.target.closest("a.zoom");
      if (!a) return;
      e.preventDefault();
      dImg.src = a.getAttribute("href");
      var im = a.querySelector("img"); dImg.alt = im ? im.alt : "";
      dCap.textContent = a.getAttribute("data-title") || "";
      dOpen.href = a.getAttribute("href");
      dlg.showModal();
    });
    dlg.querySelector("button.close").addEventListener("click", function () { dlg.close(); });
    dlg.addEventListener("click", function (e) { if (e.target === dlg) dlg.close(); });
  }

  /* results table: sort + filter */
  var table = document.getElementById("main-table");
  if (table) {
    var tbody = table.tBodies[0];
    var rows = Array.prototype.slice.call(tbody.rows);
    var fModel = document.getElementById("f-model"), fN = document.getElementById("f-budget"), fS = document.getElementById("f-sampling");
    var count = document.getElementById("row-count"), empty = document.getElementById("no-rows");
    var sortKey = null, sortDir = 1;
    function apply() {
      var shown = 0;
      rows.forEach(function (r) {
        var ok = (!fModel.value || r.dataset.model === fModel.value) &&
                 (!fN.value || r.dataset.budget === fN.value) &&
                 (!fS.value || r.dataset.sampling === fS.value);
        r.hidden = !ok; if (ok) shown++;
      });
      count.textContent = shown + " of " + rows.length + " configurations";
      empty.hidden = shown !== 0;
    }
    [fModel, fN, fS].forEach(function (f) { f.addEventListener("change", apply); });
    document.getElementById("f-reset").addEventListener("click", function () { fModel.value = fN.value = fS.value = ""; apply(); });
    Array.prototype.forEach.call(table.querySelectorAll("th[data-key]"), function (th) {
      var b = th.querySelector("button");
      b.addEventListener("click", function () {
        var k = th.getAttribute("data-key");
        sortDir = sortKey === k ? -sortDir : 1; sortKey = k;
        Array.prototype.forEach.call(table.querySelectorAll("th[data-key]"), function (o) { o.removeAttribute("aria-sort"); });
        th.setAttribute("aria-sort", sortDir === 1 ? "ascending" : "descending");
        var isNum = th.getAttribute("data-type") === "num";
        rows.sort(function (a, b2) {
          var x = a.dataset[k], y = b2.dataset[k];
          if (isNum) { x = parseFloat(x); y = parseFloat(y); return (x - y) * sortDir; }
          return x.localeCompare(y) * sortDir;
        });
        rows.forEach(function (r) { tbody.appendChild(r); });
      });
    });
    apply();
  }
})();
