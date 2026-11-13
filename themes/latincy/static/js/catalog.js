// Catalog filters + grid/table toggle. Without JS the page shows both views unfiltered.
(function () {
  const form = document.querySelector(".catalog-filters");
  if (!form) return;
  const grid = document.getElementById("catalog-grid");
  const table = document.getElementById("catalog-table");
  const toggle = form.querySelector(".view-toggle");
  const count = form.querySelector(".catalog-count");
  const items = [...grid.children, ...table.tBodies[0].rows];
  form.hidden = false;
  document.body.classList.add("js");

  let tableView = false;
  try { tableView = localStorage.getItem("catalog-view") === "table"; } catch (e) {}

  function apply() {
    const f = Object.fromEntries(new FormData(form));
    let shown = 0;
    for (const el of items) {
      const ok = ["language", "kind", "line"].every(
        (k) => !f[k] || (el.dataset[k] || "").split(" ").includes(f[k])
      );
      el.hidden = !ok;
      if (ok && el.parentElement === grid) shown++;
    }
    count.value = shown + " / " + grid.children.length;
    grid.hidden = tableView;
    table.hidden = !tableView;
    toggle.setAttribute("aria-pressed", String(tableView));
  }

  form.addEventListener("change", apply);
  toggle.addEventListener("click", () => {
    tableView = !tableView;
    try { localStorage.setItem("catalog-view", tableView ? "table" : "grid"); } catch (e) {}
    apply();
  });
  apply();
})();
