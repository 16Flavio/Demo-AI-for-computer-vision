// Langue du site de démos : français ou anglais.
// Ordre de choix : ?lang= dans l'adresse (lien depuis flaviodrogo.be ou bouton EN/FR), choix mémorisé,
// puis langue du navigateur. Chargé dans <head>, avant le rendu, pour éviter d'afficher les deux langues.
//
// Dans le HTML, chaque texte existe dans les deux langues (attribut lang="fr" / lang="en") et style.css
// masque la langue inactive. Les attributs se traduisent avec data-en-placeholder, data-en-title,
// data-en-alt et data-en-href, et le titre de la page avec <title data-en="...">.
(function () {
  const LANGUES = ["fr", "en"];
  let langue = new URLSearchParams(location.search).get("lang");
  if (LANGUES.includes(langue)) {
    try { localStorage.setItem("langue", langue); } catch {}
  } else {
    try { langue = localStorage.getItem("langue"); } catch { langue = null; }
  }
  if (!LANGUES.includes(langue)) {
    langue = (navigator.language || "fr").toLowerCase().startsWith("fr") ? "fr" : "en";
  }
  document.documentElement.lang = langue;

  // Texte dans la langue active : t("Bonjour", "Hello")
  window.t = (fr, en) => (langue === "en" ? en : fr);

  // Nombre avec la virgule ou le point décimal selon la langue
  window.nombre = (x, decimales = 0) => Number(x).toLocaleString(langue === "en" ? "en-GB" : "fr-BE", {
    minimumFractionDigits: decimales, maximumFractionDigits: decimales,
  });

  // Messages d'erreur de l'API (toujours en français) traduits pour les visiteurs anglophones
  const ERREURS = [
    [/^Fichier trop volumineux \(maximum (\d+) Mo\)\.$/, "File too large (maximum $1 MB)."],
    [/^Image illisible\.$/, "Unreadable image."],
    [/^PDF illisible\.$/, "Unreadable PDF."],
    [/^Serveur occupé, réessayez dans quelques secondes\.$/, "Server busy, please try again in a few seconds."],
    [/^Trop de requêtes, patientez une minute\.$/, "Too many requests, please wait a minute."],
    [/^Document introuvable ou expiré : renvoyez le PDF\.$/, "Document not found or expired: please send the PDF again."],
    [/^Le PDF a (\d+) pages \(maximum (\d+)\)\.$/, "The PDF has $1 pages (maximum $2)."],
    [/^Aucun texte trouvé dans ce PDF \(document scanné \?\)\.$/, "No text found in this PDF (scanned document?)."],
    [/^Aucune solution trouvée\.$/, "No solution found."],
    [/^(\d+) véhicule\(s\) de capacité (\d+) ne suffisent pas/, "$1 vehicle(s) with capacity $2 are not enough"],
    [/^Délai de génération dépassé\.$/, "Generation time limit exceeded."],
    [/^Le modèle de langage est indisponible pour le moment\.$/, "The language model is unavailable at the moment."],
  ];
  window.traduireErreur = (message) => {
    if (langue !== "en" || typeof message !== "string") return message;
    for (const [motif, traduction] of ERREURS) {
      if (motif.test(message)) return message.replace(motif, traduction);
    }
    return message;
  };

  document.addEventListener("DOMContentLoaded", () => {
    if (langue === "en") {
      const titre = document.querySelector("title[data-en]");
      if (titre) document.title = titre.dataset.en;
      for (const el of document.querySelectorAll("[data-en-placeholder]")) el.placeholder = el.dataset.enPlaceholder;
      for (const el of document.querySelectorAll("[data-en-title]")) el.title = el.dataset.enTitle;
      for (const el of document.querySelectorAll("[data-en-alt]")) el.alt = el.dataset.enAlt;
      for (const el of document.querySelectorAll("[data-en-href]")) el.href = el.dataset.enHref;
    }

    // Les liens entre pages du site gardent la langue, même si le navigateur bloque localStorage
    for (const a of document.querySelectorAll("a[href]")) {
      const href = a.getAttribute("href");
      if (/^(https?:|mailto:|#)/.test(href) || !/(\.html|\.\/)$/.test(href.split("?")[0])) continue;
      a.href = href.split("?")[0] + "?lang=" + langue;
    }

    // Bouton EN / FR : propose l'autre langue, sur la même page
    const bouton = document.getElementById("changer-langue");
    if (bouton) {
      const autre = langue === "fr" ? "en" : "fr";
      bouton.textContent = autre.toUpperCase();
      bouton.hreflang = autre;
      bouton.href = location.pathname + "?lang=" + autre;
    }
  });
})();
