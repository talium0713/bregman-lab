import { $ } from "./core/dom.js";
import { initT1 } from "./tasks/task1-dual-manifold.js";
import { initT2 } from "./tasks/task2-theorem-live.js";
import { initT3 } from "./tasks/task3-admissibility.js";

/* boot all three tasks; keep their public APIs */
const T1 = initT1();
initT2();
initT3();

/* left-sidebar navigation */
document.querySelectorAll(".navbtn").forEach(b => b.addEventListener("click", () => {
  document.querySelectorAll(".navbtn").forEach(x => x.classList.remove("active"));
  document.querySelectorAll("section.task").forEach(x => x.classList.remove("show"));
  b.classList.add("active");
  $(b.dataset.task).classList.add("show");
  if (b.dataset.task === "t1") T1.ensure();   // lazy-init the WebGL surface on first view
}));
