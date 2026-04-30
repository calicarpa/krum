document.addEventListener("DOMContentLoaded", function () {
  document.querySelectorAll('.viewcode-link, .reference.external[href*="github.com"]').forEach(function (link) {
    link.setAttribute("target", "_blank");
    link.setAttribute("rel", "noopener noreferrer");
  });
});
