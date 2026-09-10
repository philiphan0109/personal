const episodeCount = document.querySelector("#episode-count");

window.setTimeout(async () => {
  try {
    const response = await fetch(
      "https://philiphan.goatcounter.com/counter//.json",
      { cache: "no-store" },
    );
    const { count } = await response.json();
    const digits = count.replace(/\D/g, "");

    if (digits) episodeCount.textContent = digits.padStart(5, "0");
  } catch {}
}, 1000);
