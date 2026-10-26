(() => {
  if (typeof window.Lenis === "function") {
    new window.Lenis({
      autoRaf: true,
      anchors: true,
      stopInertiaOnNavigate: true,
      respectReducedMotion: true,
    });
  }

  const image = document.getElementById("crop-hero-image");
  const form = document.getElementById("crop-filter");
  const data = document.getElementById("crop-visual-map");
  if (!image || !form || !data) return;

  const select = form.querySelector("select[name='item']");
  const status = document.getElementById("crop-preview-status");
  const title = document.getElementById("crop-photo-title");
  const credit = document.getElementById("crop-photo-creator");
  const link = document.querySelector("#crop-photo-credit a");
  const cropLabel = document.getElementById("hero-crop-label");
  let visuals;
  try {
    visuals = JSON.parse(data.textContent);
  } catch {
    return;
  }
  if (!select) return;

  select.addEventListener("change", () => {
    const visual = visuals[select.value];
    if (!visual) return;

    if (cropLabel) cropLabel.textContent = select.value;
    if (title) title.textContent = visual.title;
    if (credit) credit.textContent = visual.credit;
    if (link) link.href = visual.source_url;
    if (status) status.textContent = `Image preview updated for ${select.value}. Choose “Update view” to load its yield records.`;

    const nextImage = new Image();
    nextImage.onload = () => {
      image.classList.add("is-changing");
      window.setTimeout(() => {
        image.src = visual.url;
        image.alt = visual.alt;
        requestAnimationFrame(() => image.classList.remove("is-changing"));
      }, 120);
    };
    nextImage.src = visual.url;
  });
})();
