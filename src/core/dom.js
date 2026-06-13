import { REG } from "./math.js";

/* tiny DOM helpers */
const $=id=>document.getElementById(id);
const el=(tag,attrs={},html)=>{const e=document.createElementNS("http://www.w3.org/2000/svg",tag);for(const k in attrs)e.setAttribute(k,attrs[k]);if(html!=null)e.textContent=html;return e;};

function makeSeg(container, keys, cur, onSel){
  container.innerHTML="";
  keys.forEach(k=>{
    const b=document.createElement("button");b.textContent=REG[k].label;
    if(k===cur)b.classList.add("on");
    b.addEventListener("click",()=>{container.querySelectorAll("button").forEach(x=>x.classList.remove("on"));b.classList.add("on");onSel(k);});
    container.appendChild(b);
  });
}

export { $, el, makeSeg };
