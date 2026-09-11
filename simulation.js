const blockStackDemo = document.querySelector("#block-stack-demo");

const episodes = [
  "assets/simulation/block-stack-11.mp4?v=17",
  "assets/simulation/block-stack-29.mp4?v=17",
  "assets/simulation/block-stack-47.mp4?v=17",
  "assets/simulation/block-stack-73.mp4?v=17",
  "assets/simulation/block-stack-101.mp4?v=17",
];

const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
const saveData = navigator.connection?.saveData === true;
const preloader = document.createElement("video");
let currentEpisode = -1;
let nextEpisode = -1;

function randomEpisode(excluding) {
  const choices = episodes
    .map((_, index) => index)
    .filter((index) => index !== excluding);

  return choices[Math.floor(Math.random() * choices.length)];
}

function prepareNextEpisode() {
  nextEpisode = randomEpisode(currentEpisode);

  if (!saveData) {
    preloader.preload = "auto";
    preloader.src = episodes[nextEpisode];
    preloader.load();
  }
}

function playEpisode(index) {
  currentEpisode = index;
  blockStackDemo.src = episodes[index];
  blockStackDemo.load();
  blockStackDemo.play().catch(() => {});
  prepareNextEpisode();
}

blockStackDemo.addEventListener("ended", () => {
  playEpisode(nextEpisode);
});

if (reducedMotion.matches) {
  blockStackDemo.removeAttribute("autoplay");
} else {
  playEpisode(randomEpisode(-1));
}
