const episodeCount = document.querySelector("#episode-count");

fetch("https://counterapi.com/api/yangphiliphan.com/view/home", {
  cache: "no-store",
})
  .then((response) => response.json())
  .then(({ value }) => {
    episodeCount.textContent = String(value).padStart(5, "0");
  })
  .catch(() => {});
