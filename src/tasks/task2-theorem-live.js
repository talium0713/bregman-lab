import { L, SN, clampN, lg, rnd, normS, softmax3, projS3, REG, REGKEYS, transP, sampleCat, newRewards, randPi, solveDP } from "../core/math.js";
import { $, el, makeSeg } from "../core/dom.js";

export function initT2(){

  let regK="kl", alpha=1.0, gamma=0.9, eps=0.2;
  let rewards=newRewards();
  let pis=Array.from({length:L},()=>Array.from({length:SN},randPi));
  let sel={type:"edge",l:0,s:1,a:0}, hover=null;
  let root={l:0,s:1,a:0};
  let samples=[], lastPath=null, sol=null;
  const MAXN=5000;
  const COLX=[56,180,304,428,540], ROWY=[54,126,198];

  function recompute(resetSamples){
    sol=solveDP(regK,rewards,pis,alpha,gamma,eps);
    if(resetSamples){samples=[];lastPath=null;}
    renderAll();
  }
  function sampleTraj(){
    const{l:l0,s:s0,a:a0}=root;
    const path=[{l:l0,s:s0}];
    if(l0===L-1)return{G:0,path};
    let s=sampleCat(transP(a0,eps)),G=0,k=1;
    for(let l=l0+1;l<L;l++){
      G+=Math.pow(gamma,k)*alpha*sol.B[l][s];path.push({l,s});
      if(l<L-1){const a=sampleCat(pis[l][s]);s=sampleCat(transP(a,eps));}
      k++;
    }
    return{G,path};
  }
  function draw(n){
    const room=MAXN-samples.length;if(room<=0)return;
    const m=Math.min(n,room);let last=null;
    for(let i=0;i<m;i++){last=sampleTraj();samples.push(last.G);}
    lastPath=last.path;renderMC();renderMap();
  }

  function renderMap(){
    const svg=$("t2-map");svg.innerHTML="";
    const hN=hover&&hover.type==="node"?hover:(sel&&sel.type==="node"?sel:null);
    const hE=hover&&hover.type==="edge"?hover:(sel&&sel.type==="edge"?sel:null);
    let transHi=null;
    if(hE&&hE.l<L-1)transHi={layer:hE.l+1,probs:transP(hE.a,eps)};
    /* edges */
    for(let l=0;l<L;l++)for(let s=0;s<SN;s++)for(let a=0;a<3;a++){
      const x1=COLX[l],y1=ROWY[s],x2=COLX[l+1],y2=ROWY[a];
      const isSel=sel&&sel.type==="edge"&&sel.l===l&&sel.s===s&&sel.a===a;
      const isHov=hover&&hover.type==="edge"&&hover.l===l&&hover.s===s&&hover.a===a;
      const fromHi=hN&&hN.l===l&&hN.s===s;
      let stroke="#6a6378",width=1.1,op=0.16;
      if(fromHi){stroke="#5bc8d6";width=1+10*pis[l][s][a];op=0.85;}
      if(isSel||isHov){stroke="#d9a441";width=Math.max(width,3);op=1;}
      svg.appendChild(el("line",{x1,y1,x2,y2,stroke,"stroke-width":width,opacity:op,"stroke-linecap":"round"}));
      const hit=el("line",{x1,y1,x2,y2,stroke:"transparent","stroke-width":13,style:"cursor:pointer"});
      hit.addEventListener("click",()=>{sel={type:"edge",l,s,a};root={l,s,a};samples=[];lastPath=null;renderAll();});
      hit.addEventListener("mouseenter",()=>{hover={type:"edge",l,s,a};renderMap();});
      hit.addEventListener("mouseleave",()=>{hover=null;renderMap();});
      svg.appendChild(hit);
    }
    /* trajectory arc */
    if(lastPath&&lastPath.length>1){
      const pts=lastPath.map(({l,s})=>[COLX[l],ROWY[s]]);
      let d=`M${pts[0][0]},${pts[0][1]}`;
      for(let i=1;i<pts.length;i++){
        const[x1,y1]=pts[i-1],[x2,y2]=pts[i];
        const dx=x2-x1,dy=y2-y1,len=Math.hypot(dx,dy)||1;
        d+=` Q${(x1+x2)/2+(-dy/len)*16},${(y1+y2)/2+(dx/len)*16} ${x2},${y2}`;
      }
      svg.appendChild(el("path",{d,fill:"none",stroke:"#e84393","stroke-width":2.6,opacity:0.95,"stroke-linecap":"round","stroke-linejoin":"round"}));
      lastPath.forEach(({l,s})=>svg.appendChild(el("circle",{cx:COLX[l],cy:ROWY[s],r:4.2,fill:"#e84393",stroke:"#13101a","stroke-width":1.4})));
      const t=el("text",{x:COLX[lastPath[0].l]+4,y:ROWY[lastPath[0].s]-16,fill:"#e84393","font-size":9.5},"마지막 τ");t.setAttribute("class","math");svg.appendChild(t);
    }
    /* nodes */
    for(let l=0;l<L;l++)for(let s=0;s<SN;s++){
      const x=COLX[l],y=ROWY[s];
      const isSel=sel&&sel.type==="node"&&sel.l===l&&sel.s===s;
      const inT=transHi&&transHi.layer===l;
      const tOp=inT?0.18+0.82*(transHi.probs[s]/Math.max(...transHi.probs)):1;
      if(inT)svg.appendChild(el("circle",{cx:x,cy:y,r:15,fill:"#d9a441",opacity:0.3*tOp+0.04}));
      const c=el("circle",{cx:x,cy:y,r:11,fill:isSel?"#234a52":"#262130",stroke:isSel?"#5bc8d6":"#2c2738","stroke-width":isSel?2.2:1.5,opacity:inT?0.35+0.65*tOp:1,style:"cursor:pointer"});
      c.addEventListener("click",()=>{sel={type:"node",l,s};renderAll();});
      c.addEventListener("mouseenter",()=>{hover={type:"node",l,s};renderMap();});
      c.addEventListener("mouseleave",()=>{hover=null;renderMap();});
      svg.appendChild(c);
      if(inT){const t=el("text",{x,y:y+3.5,fill:"#d9a441","font-size":8.5,"text-anchor":"middle","pointer-events":"none"},(transHi.probs[s]*100).toFixed(0)+"%");t.setAttribute("class","mono");svg.appendChild(t);}
    }
    /* terminal + labels */
    ROWY.forEach((y)=>{
      svg.appendChild(el("rect",{x:COLX[4]-7,y:y-7,width:14,height:14,rx:3,fill:"#262130",stroke:"#2c2738"}));
      const t=el("text",{x:COLX[4],y:y+3.5,fill:"#6a6378","font-size":9,"text-anchor":"middle"},"⊥");t.setAttribute("class","math");svg.appendChild(t);
    });
    COLX.slice(0,4).forEach((x,l)=>{const t=el("text",{x,y:238,fill:"#6a6378","font-size":10,"text-anchor":"middle"},"l="+l);t.setAttribute("class","math");svg.appendChild(t);});
    const tT=el("text",{x:COLX[4],y:238,fill:"#6a6378","font-size":10,"text-anchor":"middle"},"T");tT.setAttribute("class","math");svg.appendChild(tT);
    svg.appendChild(el("text",{x:8,y:16,fill:"#6a6378","font-size":9.5},"node: π 굵기 · edge: 전이 분포 · 클릭으로 선택"));
  }

  function renderInfo(){
    const box=$("t2-info");
    if(sel&&sel.type==="edge"){
      const{l,s,a}=sel;const gap=sol.Qs[l][s][a]-sol.Qp[l][s][a];
      box.innerHTML=`<div class="card">
        <div style="display:flex;align-items:baseline;gap:8px;margin-bottom:8px">
          <span class="math" style="font-size:14px;color:var(--gold)">edge · (l=${l}, s${s}) —a${"₁₂₃"[a]}→ ${l<L-1?("s"+a):"⊥"}</span>
          <span style="font-size:10px;color:var(--faint)">이 edge가 MC root</span>
        </div>
        <div class="sl"><div class="lab"><span class="math">r(s,a) 보상</span><span class="mono" id="t2-rv">${rewards[l][s][a].toFixed(2)}</span></div>
          <input type="range" id="t2-rs" min="-1" max="1" step="0.01" value="${rewards[l][s][a]}" style="accent-color:var(--gold)"></div>
        <div style="display:flex;gap:8px;margin-top:10px">
          <div class="mini" style="flex:1"><div class="math" style="font-size:11px;color:var(--star)">Q*(s,a)</div><div class="mono" style="font-size:15px">${sol.Qs[l][s][a].toFixed(4)}</div></div>
          <div class="mini" style="flex:1"><div class="math" style="font-size:11px;color:var(--pi)">Q^π(s,a)</div><div class="mono" style="font-size:15px">${sol.Qp[l][s][a].toFixed(4)}</div></div>
          <div class="mini" style="flex:1"><div class="math" style="font-size:11px;color:var(--gold)">gap (좌변)</div><div class="mono" style="font-size:15px">${gap.toFixed(4)}</div></div>
        </div></div>`;
      $("t2-rs").addEventListener("input",e=>{
        rewards[l][s][a]=parseFloat(e.target.value);
        $("t2-rv").textContent=rewards[l][s][a].toFixed(2);
        rewards=rewards.map(x=>x); recompute(true);
      });
    }else if(sel&&sel.type==="node"){
      const{l,s}=sel;
      box.innerHTML=`<div class="card">
        <div class="math" style="font-size:14px;color:var(--pi);margin-bottom:8px">state · (l=${l}, s${s}) — π는 직접, π*는 보상·α로부터 계산</div>
        <div style="display:flex;gap:12px">
          <div style="flex:0 0 52%"><svg id="t2-tri" viewBox="0 0 190 170" style="width:100%;background:#221c30;border-radius:10px;border:1px solid var(--line);touch-action:none"></svg></div>
          <div style="flex:1;display:flex;flex-direction:column;gap:6px;justify-content:center">
            <div class="mono" style="font-size:11px;color:var(--pi)" id="t2-pv"></div>
            <div class="mono" style="font-size:11px;color:var(--star)" id="t2-psv"></div>
            <div class="mini" style="margin-top:4px"><div class="math" style="font-size:11px;color:var(--star)">B_Ω(π‖π*)</div><div class="mono" style="font-size:15px" id="t2-bv"></div></div>
          </div></div></div>`;
      drawTri(l,s);
    }else box.innerHTML="";
  }
  function drawTri(l,s){
    const svg=$("t2-tri");if(!svg)return;svg.innerHTML="";
    const W=190,H=170,VA=[W/2,14],VB=[16,H-14],VC=[W-16,H-14];
    const b2=v=>[v[0]*VA[0]+v[1]*VB[0]+v[2]*VC[0], v[0]*VA[1]+v[1]*VB[1]+v[2]*VC[1]];
    svg.appendChild(el("polygon",{points:`${VA} ${VB} ${VC}`,fill:"none",stroke:"#2c2738","stroke-width":1.3}));
    [["a₁",VA[0],VA[1]-4],["a₂",VB[0]-2,VB[1]+11],["a₃",VC[0]+2,VC[1]+11]].forEach(([t,x,y])=>{
      const tx=el("text",{x,y,fill:"#6a6378","font-size":10,"text-anchor":"middle"},t);tx.setAttribute("class","math");svg.appendChild(tx);});
    const ps=sol.pistar[l][s],p=pis[l][s];
    const[qx,qy]=b2(ps),[px,py]=b2(p);
    svg.appendChild(el("line",{x1:px,y1:py,x2:qx,y2:qy,stroke:"#ece7df","stroke-width":0.8,"stroke-dasharray":"3 3",opacity:0.4}));
    svg.appendChild(el("circle",{cx:qx,cy:qy,r:7.5,fill:"#e84393",stroke:"#13101a","stroke-width":2}));
    svg.appendChild(el("circle",{cx:px,cy:py,r:7.5,fill:"#5bc8d6",stroke:"#13101a","stroke-width":2}));
    $("t2-pv").textContent="π = ("+p.map(x=>x.toFixed(2)).join(", ")+")";
    $("t2-psv").textContent="π* = ("+ps.map(x=>x.toFixed(2)).join(", ")+")";
    $("t2-bv").textContent=sol.B[l][s].toFixed(4);
    const xy2b=(x,y)=>{const d=(VB[1]-VC[1])*(VA[0]-VC[0])+(VC[0]-VB[0])*(VA[1]-VC[1]);
      const l1=((VB[1]-VC[1])*(x-VC[0])+(VC[0]-VB[0])*(y-VC[1]))/d;
      const l2=((VC[1]-VA[1])*(x-VC[0])+(VA[0]-VC[0])*(y-VC[1]))/d;
      return normS([l1,l2,1-l1-l2]);};
    let drag=false;
    const ev=e=>{const r=svg.getBoundingClientRect();return xy2b((e.clientX-r.left)/r.width*W,(e.clientY-r.top)/r.height*H);};
    svg.onpointerdown=e=>{drag=true;svg.setPointerCapture(e.pointerId);pis[l][s]=ev(e);pis=pis.map(x=>x);recompute(true);};
    svg.onpointermove=e=>{if(!drag)return;pis[l][s]=ev(e);pis=pis.map(x=>x);recompute(true);};
    svg.onpointerup=svg.onpointercancel=()=>drag=false;
  }

  function renderMC(){
    const N=samples.length;
    const exact=sol.Qs[root.l][root.s][root.a]-sol.Qp[root.l][root.s][root.a];
    const mean=N?samples.reduce((a,b)=>a+b,0)/N:0;
    $("t2-exact-lab").textContent=`좌변 · DP 정확값 @ root (${root.l},s${root.s},a${"₁₂₃"[root.a]})`;
    $("t2-exact").textContent=exact.toFixed(5);
    $("t2-mc-lab").textContent=`우변 · MC 추정 (N=${N}${N>=MAXN?" · max":""})`;
    $("t2-mc").textContent=N?mean.toFixed(5):"—";
    $("t2-err").textContent=N?`|오차| = ${Math.abs(mean-exact).toFixed(5)} · 마지막 τ의 G = ${samples[N-1].toFixed(4)}`:"";
    /* convergence */
    const svg=$("t2-conv");svg.innerHTML="";
    const W=560,Hh=132,mL=8,mR=8,mT=12,mB=8;
    if(!N){
      const t=el("text",{x:W/2,y:Hh/2+4,fill:"#6a6378","font-size":12,"text-anchor":"middle"},"rollout을 샘플링하면 표본평균 Ḡ_N의 수렴 곡선이 나타난다");svg.appendChild(t);return;
    }
    const means=[];let sAcc=0;
    for(let i=0;i<N;i++){sAcc+=samples[i];means.push(sAcc/(i+1));}
    let lo=Math.min(...means,exact),hi=Math.max(...means,exact);
    const pad=Math.max((hi-lo)*0.2,1e-4);lo-=pad;hi+=pad;
    const X=i=>mL+(W-mL-mR)*i/Math.max(N-1,1);
    const Y=v=>mT+(1-(v-lo)/(hi-lo))*(Hh-mT-mB);
    svg.appendChild(el("line",{x1:mL,y1:Y(exact),x2:W-mR,y2:Y(exact),stroke:"#d9a441","stroke-width":1.4,"stroke-dasharray":"5 4"}));
    const tl=el("text",{x:W-mR-2,y:Y(exact)-5,fill:"#d9a441","font-size":10,"text-anchor":"end"},"좌변 Q*−Q^π (DP)");tl.setAttribute("class","math");svg.appendChild(tl);
    const step=Math.max(1,Math.floor(N/420));const pts=[];
    for(let i=0;i<N;i+=step)pts.push(X(i).toFixed(1)+","+Y(means[i]).toFixed(1));
    pts.push(X(N-1).toFixed(1)+","+Y(means[N-1]).toFixed(1));
    svg.appendChild(el("polyline",{points:pts.join(" "),fill:"none",stroke:"#5bc8d6","stroke-width":1.8}));
    const t2l=el("text",{x:mL+2,y:15,fill:"#5bc8d6","font-size":10},"우변 MC: Ḡ_N");t2l.setAttribute("class","math");svg.appendChild(t2l);
  }
  function renderWarn(){
    $("t2-warn").innerHTML=(sol.boundary&&!REG[regK].essSmooth)
      ?`<div class="warn">⚠ 일부 상태에서 π*가 simplex 경계에 닿았다. ${REG[regK].label}은 essential smoothness가 없어 등식이 ≥로 약화될 수 있다 — Lemma 1이 interior π*를 가정하는 이유. MC 평균이 DP 값보다 작게 수렴할 수 있음.</div>`:"";
  }
  function renderAll(){renderMap();renderInfo();renderWarn();renderMC();}

  /* controls */
  makeSeg($("t2-reg"),REGKEYS,regK,k=>{regK=k;recompute(true);});
  $("t2-alpha").addEventListener("input",e=>{alpha=parseFloat(e.target.value);$("t2-alpha-v").textContent=alpha.toFixed(2);recompute(true);});
  $("t2-gamma").addEventListener("input",e=>{gamma=parseFloat(e.target.value);$("t2-gamma-v").textContent=gamma.toFixed(2);recompute(true);});
  $("t2-eps").addEventListener("input",e=>{eps=parseFloat(e.target.value);$("t2-eps-v").textContent=eps.toFixed(2);recompute(true);});
  $("t2-randr").addEventListener("click",()=>{rewards=newRewards();recompute(true);});
  $("t2-snap").addEventListener("click",()=>{pis=sol.pistar.map(Ls=>Ls.map(p=>normS([...p])));recompute(true);});
  $("t2-shuf").addEventListener("click",()=>{pis=Array.from({length:L},()=>Array.from({length:SN},randPi));recompute(true);});
  $("t2-s1").addEventListener("click",()=>draw(1));
  $("t2-s100").addEventListener("click",()=>draw(100));
  $("t2-s1000").addEventListener("click",()=>draw(1000));
  $("t2-reset").addEventListener("click",()=>{samples=[];lastPath=null;renderMC();renderMap();});
  recompute(true);
  return {};

}
