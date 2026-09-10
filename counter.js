const episodeCount = document.querySelector("#episode-count");

window.sc_project = 13354429;
window.sc_invisible = 0;
window.sc_security = "69f4aadb";
window.sc_text = 2;
window.sc_remove_link = 1;
window.sc_first_party_cookie = 0;

const observer = new MutationObserver(() => {
  const counter = episodeCount.querySelector(".statcounter");
  const value = counter?.textContent.trim();

  if (/^\d+$/.test(value)) {
    episodeCount.textContent = value.padStart(5, "0");
    observer.disconnect();
  }
});

observer.observe(episodeCount, { childList: true, subtree: true });

const script = document.createElement("script");
script.src = "https://www.statcounter.com/counter/counter.js";
script.async = true;
episodeCount.append(script);
