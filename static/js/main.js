(function () {
  function copyBib() {
    const b = document.getElementById("copy-bib");
    if (!b) return;
    b.addEventListener("click", () => {
      navigator.clipboard.writeText(document.getElementById("bibtex-text").innerText).then(() => {
        b.textContent = "Copied!";
        setTimeout(() => (b.textContent = "Copy"), 1500);
      });
    });
  }

  document.addEventListener("DOMContentLoaded", () => {
    copyBib();
    if (window.renderMathInElement) {
      renderMathInElement(document.body, {
        delimiters: [{ left: "\\[", right: "\\]", display: true }, { left: "\\(", right: "\\)", display: false }],
        throwOnError: false,
      });
    }
  });
})();
