import { L, SN, clampN, lg, rnd, normS, softmax3, projS3, REG, REGKEYS, transP, sampleCat, newRewards, randPi, solveDP } from "../core/math.js";
import { $, el, makeSeg } from "../core/dom.js";

export function initT3(){

  const GAMMA=0.9, STEPS=300, LR=0.08;
  let nmc=2, npairs=600, nseeds=3, targetPeak=0.6, EPS=0.2, BATCH=16, gradMode="A", offPolicy=false, DEPTH=4;
  let lastRun=null;
  let rewards=newRewards(DEPTH);
  let dataset=null;   // shared preference pairs
  let alphas=null;    // calibrated α per divergence (peak-matched)
  let running=false;
  let refPol=null;    // π_ref over actions: null → uniform; set a (DEPTH,SN,3) array for non-uniform ref
  const refOf=(l,s)=>refPol?refPol[l][s]:[1/3,1/3,1/3];
  /* sign of the inner term C_Ω in the trajectory score: telescoping Eq 11 (β=V*−αC_Ω) gives a
     NEGATIVE coefficient, score = Σ α[∇Ω]_a − α Σ C_Ω(π(·|s')). γ dropped (undiscounted score).
     KL unaffected (constC ⇒ C=0 here, and Φ'_KL≡0). */
  const ISIGN=-1;

  const uniformPis=()=>Array.from({length:DEPTH},()=>Array.from({length:SN},()=>[1/3,1/3,1/3]));

  /* ── step 1–2: α calibration — match mean over states of max_a π*(a|s) ── */
  function peakiness(rk,al){
    const sol=solveDP(rk,rewards,uniformPis(),al,GAMMA,EPS,DEPTH,refPol);
    let m=0;for(let l=0;l<DEPTH;l++)for(let s=0;s<SN;s++)m+=Math.max(...sol.pistar[l][s]);
    return m/(DEPTH*SN);
  }
  function calibrate(){
    alphas={};
    REGKEYS.forEach(rk=>{
      let lo=0.02,hi=40; // peakiness monotone↓ in α: lo side too peaked, hi side too soft
      for(let i=0;i<55;i++){
        const mid=Math.sqrt(lo*hi);
        if(peakiness(rk,mid)>targetPeak)lo=mid;else hi=mid;
      }
      alphas[rk]=Math.sqrt(lo*hi);
    });
    renderChips();
    drawCStats();
    drawSweep();
    drawMDP();
    drawCProbe();
  }
  function renderChips(){
    $("t3-chips").innerHTML=REGKEYS.map(rk=>
      `<span class="mono" style="color:${REG[rk].color}">${REG[rk].label} α=${alphas[rk].toFixed(2)} <span style="color:var(--faint)">(peak ${peakiness(rk,alphas[rk]).toFixed(2)})</span></span>`
    ).join(' <span style="color:var(--faint)">·</span> ');
  }

  /* ── C_Ω statistics at calibrated π*_Ω: π-dependence (state spread) & MC noise ── */
  function cStats(rk){
    const R=REG[rk];
    const sol=solveDP(rk,rewards,uniformPis(),alphas[rk],GAMMA,EPS,DEPTH,refPol);
    const Cs=[],Sds=[];
    for(let l=0;l<DEPTH;l++)for(let s=0;s<SN;s++){
      const p=sol.pistar[l][s].map(x=>Math.max(x,1e-9));
      const ref=refOf(l,s);
      if(R.constC){Cs.push(1);Sds.push(0);continue;}
      let E=0,E2=0,Euni=0;   // Euni = behavior (uniform) expectation of Φ — data action ~ uniform, ref-independent
      if(rk==="euc"){
        for(let a=0;a<3;a++){const x=p[a]-ref[a];E+=p[a]*x;E2+=p[a]*x*x;Euni+=(1/3)*x;}
        Cs.push(E-R.omega(p,ref));
      }else{
        for(let a=0;a<3;a++){const ph=R.Phi(p[a]/ref[a]);E+=p[a]*ph;E2+=p[a]*ph*ph;Euni+=(1/3)*ph;}
        Cs.push(E);
      }
      /* off-policy: data action ~ uniform behavior; bias = |E_uniform[Φ] − E_π*[Φ]| */
      Sds.push(offPolicy ? Math.abs(Euni-E) : Math.sqrt(Math.max(E2-E*E,0)));
    }
    const mu=Cs.reduce((a,b)=>a+b,0)/Cs.length;
    return{
      meanC:mu,
      stateStd:Math.sqrt(Cs.reduce((s,x)=>s+(x-mu)**2,0)/Cs.length),
      mcSd:Sds.reduce((a,b)=>a+b,0)/Sds.length
    };
  }
  function drawCStats(){
    const svg=$("t3-cstats");if(!svg)return;svg.innerHTML="";
    const W=560,H=190,mL=46,mR=10,mT=18,mB=44;
    const stats=REGKEYS.map(rk=>({rk,...cStats(rk)}));
    const FLOOR=1e-4;
    const CEIL=Math.max(1e-2,...stats.flatMap(s=>[s.stateStd,s.mcSd]))*1.8;
    const loL=Math.log10(FLOOR),hiL=Math.log10(CEIL);
    const lY=v=>mT+(1-(Math.log10(Math.max(v,FLOOR))-loL)/(hiL-loL))*(H-mT-mB);
    for(let d=Math.ceil(loL);d<=Math.floor(hiL);d++){
      const y=lY(Math.pow(10,d));
      svg.appendChild(el("line",{x1:mL,y1:y,x2:W-mR,y2:y,stroke:"#2c2738","stroke-width":0.5,"stroke-dasharray":"2 5"}));
      const t=el("text",{x:mL-4,y:y+3,fill:"#6a6378","font-size":8.5,"text-anchor":"end"},"1e"+d);
      t.setAttribute("class","mono");svg.appendChild(t);
    }
    const base=lY(FLOOR);
    svg.appendChild(el("line",{x1:mL,y1:base,x2:W-mR,y2:base,stroke:"#2c2738"}));
    const groupW=(W-mL-mR)/stats.length;
    stats.forEach((st,i)=>{
      const x0=mL+i*groupW,bw=Math.min(17,(groupW-22)/2);
      const bar=(x,v,op,dash)=>{
        if(v<=1e-12){
          const t=el("text",{x:x+bw/2,y:base-4,fill:REG[st.rk].color,"font-size":9,"text-anchor":"middle"},"0");
          t.setAttribute("class","mono");svg.appendChild(t);return;
        }
        const r=el("rect",{x,y:lY(v),width:bw,height:Math.max(base-lY(v),0),fill:REG[st.rk].color,opacity:op,rx:2.5});
        if(dash){r.setAttribute("stroke","#ece7df");r.setAttribute("stroke-dasharray","2 2");r.setAttribute("stroke-width","0.7");}
        svg.appendChild(r);
      };
      bar(x0+8,st.stateStd,0.9,false);
      bar(x0+8+bw+4,st.mcSd,0.4,true);
      svg.appendChild(el("text",{x:x0+groupW/2,y:H-26,fill:REG[st.rk].color,"font-size":9.5,"text-anchor":"middle"},REG[st.rk].shortLabel));
      const t2=el("text",{x:x0+groupW/2,y:H-14,fill:"#6a6378","font-size":8,"text-anchor":"middle"},"C̄="+st.meanC.toFixed(2));
      t2.setAttribute("class","mono");svg.appendChild(t2);
    });
  }

  /* ── shared preference data: uniform behavior policy (off-policy) ── */
  function rollout(){
    let s=Math.floor(Math.random()*SN);
    const trans=[];let R=0;
    for(let l=0;l<DEPTH;l++){
      const a=Math.floor(Math.random()*3);
      R+=Math.pow(GAMMA,l)*rewards[l][s][a];
      const sn=l<DEPTH-1?sampleCat(transP(a,EPS)):-1;
      trans.push({l,s,a,sn});
      s=sn;
    }
    return{trans,R};
  }
  function makeDataset(){
    dataset=[];
    for(let i=0;i<10000;i++){
      const t1=rollout(),t2=rollout();
      const pw=1/(1+Math.exp(-(t1.R-t2.R)));
      dataset.push(Math.random()<pw?{w:t1.trans,l:t2.trans}:{w:t2.trans,l:t1.trans});
    }
  }

  /* ── step 3: DPO training for one (regK, seed, nmc) at calibrated α ── */
  function trainOne(regK, nmcUse, data){
    const R=REG[regK], aReg=alphas[regK];
    const sol=solveDP(regK,rewards,uniformPis(),aReg,GAMMA,EPS,DEPTH,refPol);
    const tgt=sol.pistar;
    const th=Array.from({length:DEPTH},()=>Array.from({length:SN},()=>[rnd(-0.05,0.05),rnd(-0.05,0.05),rnd(-0.05,0.05)]));
    const m=Array.from({length:DEPTH},()=>Array.from({length:SN},()=>[0,0,0]));
    const v=Array.from({length:DEPTH},()=>Array.from({length:SN},()=>[0,0,0]));
    const polAt=(l,s)=>softmax3(th[l][s],1);
    const gap=()=>{
      let g=0;
      for(let l=0;l<DEPTH;l++)for(let s=0;s<SN;s++){
        const p=polAt(l,s),q=tgt[l][s];
        g+=0.5*(Math.abs(p[0]-q[0])+Math.abs(p[1]-q[1])+Math.abs(p[2]-q[2]));
      }
      return g/(DEPTH*SN);
    };
    /* inner term C_Ω(s'). on-policy: draw n_mc actions from π(·|s').
       off-policy (extreme): use the single data-recorded next action a'_data (no resampling). */
    const drawInner=(pol,aData,ref)=>{
      if(R.constC)return{C:0,as:[]}; // KL: C_Ω ≡ const → no estimation, no noise, mode-invariant
      if(offPolicy){
        const aj=aData;
        if(aj==null)return{C:0,as:[]}; // terminal: no next action recorded
        const phi = regK==="euc" ? (pol[aj]-ref[aj]) : R.Phi(Math.max(pol[aj],1e-12)/ref[aj]);
        return{C: regK==="euc" ? phi-R.omega(pol,ref) : phi, as:[aj]};
      }
      const as=[];let e=0;
      if(regK==="euc"){for(let j=0;j<nmcUse;j++){const aj=sampleCat(pol);as.push(aj);e+=pol[aj]-ref[aj];}return{C:e/nmcUse-R.omega(pol,ref),as};}
      for(let j=0;j<nmcUse;j++){const aj=sampleCat(pol);as.push(aj);e+=R.Phi(Math.max(pol[aj],1e-12)/ref[aj]);}
      return{C:e/nmcUse,as};
    };
    /* score a trajectory: deterministic part (exact) + inner part; cache inner samples.
       aData for state s' = the action taken at s' in this same trajectory (next transition's a). */
    const scoreTraj=(trans,pols)=>{
      let sc=0;const inner=[];
      for(let i=0;i<trans.length;i++){
        const{l,s,a,sn}=trans[i];
        sc+=aReg*R.gradDPO(pols[l][s],refOf(l,s))[a];
        if(sn>=0){
          const aData=(i+1<trans.length)?trans[i+1].a:null;
          const di=drawInner(pols[l+1][sn], aData, refOf(l+1,sn));
          sc+=aReg*ISIGN*di.C;
          inner.push({sn,l,as:di.as});
        }
      }
      return{sc,inner};
    };
    const curve=[];let t=0;
    for(let step=0;step<STEPS;step++){
      const pols=Array.from({length:DEPTH},(_,l)=>Array.from({length:SN},(_,s)=>polAt(l,s)));
      const grad=Array.from({length:DEPTH},()=>Array.from({length:SN},()=>[0,0,0]));
      for(let b=0;b<BATCH;b++){
        const pair=data[Math.floor(Math.random()*data.length)];
        const Sw=scoreTraj(pair.w,pols), Sl=scoreTraj(pair.l,pols);
        const d=Sw.sc-Sl.sc;
        const coef=-1/(1+Math.exp(d)); // dℓ/dd = −σ(−d); carries the MC noise of d
        /* outer gradient: through the chosen-action implicit reward */
        const accOuter=(trans,sign)=>{
          for(const{l,s,a}of trans){
            const pol=pols[l][s];
            for(let bb=0;bb<3;bb++)grad[l][s][bb]+=coef*sign*aReg*R.dGdTheta(pol,a,bb,refOf(l,s));
          }
        };
        accOuter(pair.w,+1);accOuter(pair.l,-1);
        /* A-direction: gradient through the inner term C_Ω(s'), via the SAME a' samples */
        if(gradMode==="A"&&!R.constC){
          const accInner=(inner,sign)=>{
            for(const{sn,l,as}of inner){
              const pol=pols[l+1][sn];
              const rf=refOf(l+1,sn);
              const nz=as.length||1;
              if(regK==="euc"){
                for(const aj of as)for(let bb=0;bb<3;bb++)
                  grad[l+1][sn][bb]+=coef*sign*aReg*ISIGN*(1/nz)*pol[aj]*((aj===bb?1:0)-pol[bb]);
                for(let bb=0;bb<3;bb++){let dO=0;for(let a2=0;a2<3;a2++)dO+=(pol[a2]-rf[a2])*pol[a2]*((a2===bb?1:0)-pol[bb]);
                  grad[l+1][sn][bb]+=coef*sign*aReg*ISIGN*(-dO);}
              }else if(R.dPhi){
                for(const aj of as){const u=Math.max(pol[aj],1e-12)/rf[aj];const w=R.dPhi(u)*(1/rf[aj])/nz;
                  for(let bb=0;bb<3;bb++)grad[l+1][sn][bb]+=coef*sign*aReg*ISIGN*w*pol[aj]*((aj===bb?1:0)-pol[bb]);}
              }
            }
          };
          accInner(Sw.inner,+1);accInner(Sl.inner,-1);
        }
      }
      t++;
      for(let l=0;l<DEPTH;l++)for(let s=0;s<SN;s++)for(let a=0;a<3;a++){
        const g=grad[l][s][a]/BATCH;
        m[l][s][a]=0.9*m[l][s][a]+0.1*g;
        v[l][s][a]=0.999*v[l][s][a]+0.001*g*g;
        const mh=m[l][s][a]/(1-Math.pow(0.9,t)),vh=v[l][s][a]/(1-Math.pow(0.999,t));
        th[l][s][a]-=LR*mh/(Math.sqrt(vh)+1e-8);
      }
      if(step%15===0||step===STEPS-1)curve.push(gap());
    }
    const pol=Array.from({length:DEPTH},(_,l)=>Array.from({length:SN},(_,s)=>polAt(l,s)));
    return{curve,final:gap(),pol};
  }

  /* async runner */
  function runAll(nmcList, done){
    running=true;$("t3-run").disabled=true;$("t3-sweep").disabled=true;
    const data=dataset.slice(0,npairs);
    const jobs=[];
    for(const nm of nmcList)for(const rk of REGKEYS)for(let sd=0;sd<nseeds;sd++)jobs.push({nm,rk,sd});
    const out={};let ji=0;
    function tick(){
      if(ji>=jobs.length){running=false;$("t3-run").disabled=false;$("t3-sweep").disabled=false;
        $("t3-prog").style.width="100%";done(out);return;}
      const{nm,rk}=jobs[ji];
      const res=trainOne(rk,nm,data);
      out[nm]=out[nm]||{};out[nm][rk]=out[nm][rk]||{curves:[],finals:[],pols:[]};
      out[nm][rk].curves.push(res.curve);out[nm][rk].finals.push(res.final);out[nm][rk].pols.push(res.pol);
      ji++;
      $("t3-prog").style.width=((ji/jobs.length)*100).toFixed(1)+"%";
      $("t3-status").textContent=`Training — ${REG[rk].label} (α=${alphas[rk].toFixed(2)}) · n_mc=${nm} · job ${ji}/${jobs.length}`;
      setTimeout(tick,0);
    }
    $("t3-prog").style.width="0%";tick();
  }

  function meanArr(arrs){
    const n=Math.min(...arrs.map(a=>a.length));
    return Array.from({length:n},(_,i)=>arrs.reduce((s,a)=>s+a[i],0)/arrs.length);
  }
  const meanStd=xs=>{
    const mu=xs.reduce((a,b)=>a+b,0)/xs.length;
    const sd=Math.sqrt(xs.reduce((s,x)=>s+(x-mu)**2,0)/Math.max(xs.length-1,1));
    return{mu,sd};
  };

  function drawCurves(res){
    const svg=$("t3-curves");svg.innerHTML="";
    const W=560,H=180,mL=36,mR=10,mT=12,mB=24;
    let hi=1e-9;
    const series=REGKEYS.map(rk=>{const m=meanArr(res[rk].curves);hi=Math.max(hi,...m);return{rk,m};});
    hi*=1.08;
    const X=i=>mL+(W-mL-mR)*i/Math.max(series[0].m.length-1,1);
    const Y=v=>mT+(1-v/hi)*(H-mT-mB);
    svg.appendChild(el("line",{x1:mL,y1:Y(0),x2:W-mR,y2:Y(0),stroke:"#2c2738"}));
    [0.25,0.5,0.75,1].forEach(f=>{
      svg.appendChild(el("line",{x1:mL,y1:Y(hi*f/1.08),x2:W-mR,y2:Y(hi*f/1.08),stroke:"#2c2738","stroke-width":0.5,"stroke-dasharray":"2 5"}));
    });
    series.forEach(({rk,m})=>{
      const pts=m.map((v,i)=>X(i).toFixed(1)+","+Y(v).toFixed(1)).join(" ");
      svg.appendChild(el("polyline",{points:pts,fill:"none",stroke:REG[rk].color,"stroke-width":rk==="kl"?2.6:1.7,opacity:rk==="kl"?1:0.9}));
    });
    REGKEYS.forEach((rk,i)=>{
      svg.appendChild(el("circle",{cx:mL+8+i*74,cy:H-8,r:4,fill:REG[rk].color}));
      svg.appendChild(el("text",{x:mL+16+i*74,y:H-4.5,fill:"#9b93a8","font-size":9.5},REG[rk].shortLabel));
    });
    svg.appendChild(el("text",{x:10,y:mT+8,fill:"#6a6378","font-size":9},"gap"));
  }

  function drawBars(byNmc,title){
    $("t3-bar-title").textContent=title;
    const svg=$("t3-bars");svg.innerHTML="";
    const W=560,H=190,mL=36,mR=10,mT=14,mB=30;
    const nms=Object.keys(byNmc).map(Number).sort((a,b)=>a-b);
    let hi=1e-9;
    nms.forEach(nm=>REGKEYS.forEach(rk=>{const{mu,sd}=meanStd(byNmc[nm][rk].finals);hi=Math.max(hi,mu+sd);}));
    hi*=1.12;
    const Y=v=>mT+(1-v/hi)*(H-mT-mB);
    svg.appendChild(el("line",{x1:mL,y1:Y(0),x2:W-mR,y2:Y(0),stroke:"#2c2738"}));
    const groupW=(W-mL-mR)/nms.length;
    nms.forEach((nm,gi)=>{
      const x0=mL+gi*groupW;
      const step=(groupW-24)/REGKEYS.length;
      const bw=Math.max(7,step-4);
      REGKEYS.forEach((rk,ri)=>{
        const{mu,sd}=meanStd(byNmc[nm][rk].finals);
        const x=x0+12+ri*step;
        svg.appendChild(el("rect",{x,y:Y(mu),width:bw,height:Math.max(Y(0)-Y(mu),0),fill:REG[rk].color,opacity:0.85,rx:3}));
        svg.appendChild(el("line",{x1:x+bw/2,y1:Y(mu-sd),x2:x+bw/2,y2:Y(mu+sd),stroke:"#ece7df","stroke-width":1.2,opacity:0.8}));
        svg.appendChild(el("line",{x1:x+bw/2-3,y1:Y(mu+sd),x2:x+bw/2+3,y2:Y(mu+sd),stroke:"#ece7df","stroke-width":1.2,opacity:0.8}));
        svg.appendChild(el("line",{x1:x+bw/2-3,y1:Y(mu-sd),x2:x+bw/2+3,y2:Y(mu-sd),stroke:"#ece7df","stroke-width":1.2,opacity:0.8}));
      });
      const t=el("text",{x:x0+groupW/2,y:H-10,fill:"#9b93a8","font-size":10.5,"text-anchor":"middle"},"n_mc = "+nm);
      t.setAttribute("class","mono");svg.appendChild(t);
    });
    svg.appendChild(el("text",{x:8,y:mT+8,fill:"#6a6378","font-size":9},"final gap"));
  }

  /* ── MDP instance visualization (depth-aware, read-only) ── */
  function drawMDP(){
    const svg=$("t3-mdp");if(!svg)return;svg.innerHTML="";
    const W=560,Hh=200,mL=40,mR=20,mT=24,mB=26;
    const cols=DEPTH+1; // decision layers + terminal
    const colX=l=>mL+(W-mL-mR)*l/(cols-1);
    const rowY=s=>mT+(Hh-mT-mB)*(SN===1?0.5:s/(SN-1));
    /* transition edges (faint) + action-aim edges */
    for(let l=0;l<DEPTH;l++)for(let s=0;s<SN;s++)for(let a=0;a<3;a++){
      if(l===DEPTH-1)continue;
      const probs=transP(a,EPS);
      for(let s2=0;s2<SN;s2++){
        if(probs[s2]<1e-6)continue;
        svg.appendChild(el("line",{x1:colX(l),y1:rowY(s),x2:colX(l+1),y2:rowY(s2),
          stroke:a===0?"#5bc8d6":a===1?"#e8a0c0":"#9b8fd0",
          "stroke-width":0.5+2.2*probs[s2],opacity:0.16+0.25*probs[s2],"stroke-linecap":"round"}));
      }
    }
    /* nodes */
    for(let l=0;l<DEPTH;l++)for(let s=0;s<SN;s++){
      const x=colX(l),y=rowY(s);
      const g=el("g",{style:"cursor:default"});
      const c=el("circle",{cx:x,cy:y,r:9,fill:"#262130",stroke:"#3a3050","stroke-width":1.4});
      g.appendChild(c);
      g.appendChild(el("text",{x,y:y+3,fill:"#9b93a8","font-size":8,"text-anchor":"middle"},"s"+s));
      /* hover → reward tooltip */
      g.addEventListener("mouseenter",()=>{
        c.setAttribute("stroke","#d9a441");c.setAttribute("stroke-width","2");
        const r=rewards[l][s];
        const tip=el("g",{id:"t3-mdp-tip","pointer-events":"none"});
        const tx=x+(l>cols-3?-118:10),ty=y-30;
        tip.appendChild(el("rect",{x:tx,y:ty,width:112,height:42,rx:6,fill:"#0f0c16",stroke:"#d9a441","stroke-width":1,opacity:0.96}));
        const t1=el("text",{x:tx+8,y:ty+15,fill:"#d9a441","font-size":9.5},`l=${l}, s${s} rewards`);t1.setAttribute("class","mono");
        const t2=el("text",{x:tx+8,y:ty+31,fill:"#ece7df","font-size":10},`a₁ ${r[0].toFixed(2)}  a₂ ${r[1].toFixed(2)}  a₃ ${r[2].toFixed(2)}`);t2.setAttribute("class","mono");
        tip.appendChild(t1);tip.appendChild(t2);svg.appendChild(tip);
      });
      g.addEventListener("mouseleave",()=>{
        c.setAttribute("stroke","#3a3050");c.setAttribute("stroke-width","1.4");
        const tip=$("t3-mdp-tip");if(tip)tip.remove();
      });
      svg.appendChild(g);
    }
    /* terminal column */
    for(let s=0;s<SN;s++){const x=colX(DEPTH),y=rowY(s);
      svg.appendChild(el("rect",{x:x-6,y:y-6,width:12,height:12,rx:3,fill:"#262130",stroke:"#3a3050"}));
      svg.appendChild(el("text",{x,y:y+3,fill:"#6a6378","font-size":8,"text-anchor":"middle"},"⊥"));}
    /* layer labels */
    for(let l=0;l<DEPTH;l++){const t=el("text",{x:colX(l),y:Hh-10,fill:"#6a6378","font-size":9,"text-anchor":"middle"},"l="+l);t.setAttribute("class","mono");svg.appendChild(t);}
    const tT=el("text",{x:colX(DEPTH),y:Hh-10,fill:"#6a6378","font-size":9,"text-anchor":"middle"},"T");tT.setAttribute("class","mono");svg.appendChild(tT);
    /* action color legend */
    [["a₁","#5bc8d6"],["a₂","#e8a0c0"],["a₃","#9b8fd0"]].forEach(([lab,col],k)=>{
      svg.appendChild(el("line",{x1:mL+k*60,y1:14,x2:mL+k*60+14,y2:14,stroke:col,"stroke-width":2}));
      svg.appendChild(el("text",{x:mL+k*60+18,y:17,fill:"#9b93a8","font-size":9},lab));
    });
  }

  /* ── c(s') admissibility probe: irreducible inner-term error from (s,a,s') alone vs ε ── */
  let CVALS=[0.6,1.0,1.6], cShowMode="both";
  function cViolation(eps){
    const pa=[1/3,1/3,1/3];
    const psp=[0,1,2].map(sp=>[0,1,2].reduce((acc,a)=>acc+pa[a]*transP(a,eps)[sp],0));
    let V=0;
    for(let sp=0;sp<SN;sp++){
      const w=[0,1,2].map(a=>pa[a]*transP(a,eps)[sp]);const Z=w[0]+w[1]+w[2];
      const post=w.map(x=>x/Z);
      let E=0,E2=0;for(let a=0;a<SN;a++){E+=post[a]*CVALS[a];E2+=post[a]*CVALS[a]*CVALS[a];}
      V+=psp[sp]*Math.max(E2-E*E,0);
    }
    return Math.sqrt(V);
  }
  function drawCProbe(){
    const svg=$("t3-cprobe");if(!svg)return;svg.innerHTML="";
    const W=560,H=230,mL=46,mR=14,mT=16,mB=46;
    const epsG=[];for(let i=0;i<=30;i++)epsG.push(0.6*i/30);
    const maxViol=Math.max(cViolation(0.6),0.05);
    const hi=Math.max(0.45,maxViol*1.15);
    const X=e=>mL+(e/0.6)*(W-mL-mR);
    const Y=v=>mT+(1-v/hi)*(H-mT-mB);
    for(const gl of [0,0.1,0.2,0.3,0.4]){svg.appendChild(el("line",{x1:mL,y1:Y(gl),x2:W-mR,y2:Y(gl),stroke:"#2c2738","stroke-width":0.5,"stroke-dasharray":"2 5"}));
      const t=el("text",{x:mL-5,y:Y(gl)+3,fill:"#6a6378","font-size":8.5,"text-anchor":"end"},gl.toFixed(1));t.setAttribute("class","mono");svg.appendChild(t);}
    for(const e of [0,0.15,0.3,0.45,0.6]){const x=X(e);svg.appendChild(el("line",{x1:x,y1:mT,x2:x,y2:H-mB,stroke:"#2c2738","stroke-width":0.5,"stroke-dasharray":"2 5"}));
      const t=el("text",{x,y:H-mB+13,fill:"#6a6378","font-size":9,"text-anchor":"middle"},e.toFixed(2));t.setAttribute("class","mono");svg.appendChild(t);}
    const showState=cShowMode!=="action", showAction=cShowMode!=="state";
    if(showState){
      svg.appendChild(el("line",{x1:X(0),y1:Y(0),x2:X(0.6),y2:Y(0),stroke:"#58c08a","stroke-width":2.6}));
    }
    if(showAction){
      const pts=epsG.map(e=>X(e).toFixed(1)+","+Y(cViolation(e)).toFixed(1));
      svg.appendChild(el("polyline",{points:pts.join(" "),fill:"none",stroke:"#9b59d0","stroke-width":2.6}));
    }
    if(showState&&showAction){
      svg.appendChild(el("circle",{cx:X(0),cy:Y(0),r:5,fill:"none",stroke:"#d9a441","stroke-width":1.4}));
      svg.appendChild(el("text",{x:X(0)+8,y:Y(0)-8,fill:"#d9a441","font-size":9},"ε=0: modes coincide (admissible)"));
    }
    const ex=X(EPS);
    svg.appendChild(el("line",{x1:ex,y1:mT,x2:ex,y2:H-mB,stroke:"#ece7df","stroke-width":1,"stroke-dasharray":"3 3",opacity:0.5}));
    if(showState)svg.appendChild(el("circle",{cx:ex,cy:Y(0),r:3.5,fill:"#58c08a",stroke:"#13101a","stroke-width":1.4}));
    if(showAction)svg.appendChild(el("circle",{cx:ex,cy:Y(cViolation(EPS)),r:3.5,fill:"#9b59d0",stroke:"#13101a","stroke-width":1.4}));
    svg.appendChild(el("text",{x:ex,y:mT-3,fill:"#9b93a8","font-size":9,"text-anchor":"middle"},`ε=${EPS.toFixed(2)} · state 0.000 · action ${cViolation(EPS).toFixed(3)}`));
    svg.appendChild(el("text",{x:(mL+W-mR)/2,y:H-6,fill:"#9b93a8","font-size":10,"text-anchor":"middle"},"transition noise ε"));
    const yl=el("text",{x:12,y:mT+6,fill:"#6a6378","font-size":9},"violation");yl.setAttribute("class","math");svg.appendChild(yl);
    if(showState){svg.appendChild(el("line",{x1:mL+8,y1:mT+8,x2:mL+24,y2:mT+8,stroke:"#58c08a","stroke-width":2.6}));
      svg.appendChild(el("text",{x:mL+28,y:mT+11,fill:"#9b93a8","font-size":9.5},"state-indexed c(s′)"));}
    if(showAction){svg.appendChild(el("line",{x1:mL+160,y1:mT+8,x2:mL+176,y2:mT+8,stroke:"#9b59d0","stroke-width":2.6}));
      svg.appendChild(el("text",{x:mL+180,y:mT+11,fill:"#9b93a8","font-size":9.5},"action-indexed c(a′)"));}
  }

  function peakOf(rk,al){
    const sol=solveDP(rk,rewards,uniformPis(),al,GAMMA,EPS,DEPTH,refPol);
    let m=0;for(let l=0;l<DEPTH;l++)for(let s=0;s<SN;s++)m+=Math.max(...sol.pistar[l][s]);
    return m/(DEPTH*SN);
  }
  function drawSweep(){
    const svg=$("t3-sweep-plot");if(!svg)return;svg.innerHTML="";
    const W=560,H=330,mL=44,mR=12,mT=14,mB=54;
    const aLo=0.05,aHi=6,NA=44;
    const alg=Math.log10(aLo),ahg=Math.log10(aHi);
    const X=a=>mL+(Math.log10(a)-alg)/(ahg-alg)*(W-mL-mR);
    const Y=p=>mT+(1-p)*(H-mT-mB);
    /* grid */
    for(let p=0;p<=1;p+=0.2){const y=Y(p);svg.appendChild(el("line",{x1:mL,y1:y,x2:W-mR,y2:y,stroke:"#2c2738","stroke-width":0.5,"stroke-dasharray":"2 5"}));
      const t=el("text",{x:mL-5,y:y+3,fill:"#6a6378","font-size":8.5,"text-anchor":"end"},p.toFixed(1));t.setAttribute("class","mono");svg.appendChild(t);}
    for(let d=Math.ceil(alg);d<=Math.floor(ahg);d++){const x=X(Math.pow(10,d));svg.appendChild(el("line",{x1:x,y1:mT,x2:x,y2:H-mB,stroke:"#2c2738","stroke-width":0.5,"stroke-dasharray":"2 5"}));
      const t=el("text",{x,y:H-mB+13,fill:"#6a6378","font-size":9,"text-anchor":"middle"},"10"+(d>=0?"":"⁻")+Math.abs(d));t.setAttribute("class","mono");svg.appendChild(t);}
    /* target peak line */
    svg.appendChild(el("line",{x1:mL,y1:Y(targetPeak),x2:W-mR,y2:Y(targetPeak),stroke:"#9b93a8","stroke-width":1.2,"stroke-dasharray":"6 4"}));
    const tp=el("text",{x:W-mR-2,y:Y(targetPeak)-4,fill:"#9b93a8","font-size":9.5,"text-anchor":"end"},"target peak ≈ "+targetPeak.toFixed(3));tp.setAttribute("class","math");svg.appendChild(tp);
    /* per-divergence curves + calibrated-α vertical */
    REGKEYS.forEach(rk=>{
      const pts=[];
      for(let i=0;i<=NA;i++){const a=Math.pow(10,alg+(ahg-alg)*i/NA);pts.push(X(a).toFixed(1)+","+Y(peakOf(rk,a)).toFixed(1));}
      svg.appendChild(el("polyline",{points:pts.join(" "),fill:"none",stroke:REG[rk].color,"stroke-width":rk==="kl"?2.6:1.5,opacity:rk==="kl"?1:0.9}));
      const ax=X(alphas[rk]);
      svg.appendChild(el("line",{x1:ax,y1:mT,x2:ax,y2:H-mB,stroke:REG[rk].color,"stroke-width":0.9,"stroke-dasharray":"2 3",opacity:0.55}));
      svg.appendChild(el("circle",{cx:ax,cy:Y(targetPeak),r:3,fill:REG[rk].color}));
    });
    /* axis labels */
    const xl=el("text",{x:(mL+W-mR)/2,y:H-6,fill:"#9b93a8","font-size":10,"text-anchor":"middle"},"regularization weight α");xl.setAttribute("class","math");svg.appendChild(xl);
    /* legend */
    REGKEYS.forEach((rk,i)=>{const x=mL+6+(i%4)*120,y=mT+6+((i/4)|0)*14;
      svg.appendChild(el("line",{x1:x,y1:y,x2:x+16,y2:y,stroke:REG[rk].color,"stroke-width":rk==="kl"?2.6:1.5}));
      svg.appendChild(el("text",{x:x+20,y:y+3.5,fill:"#9b93a8","font-size":9},REG[rk].label));});
  }

  /* ── downloads ── */
  function svgToBlob(svgEl){
    const clone=svgEl.cloneNode(true);
    clone.setAttribute("xmlns","http://www.w3.org/2000/svg");
    const css='<style>text{font-family:Georgia,serif}</style>';
    const body=clone.outerHTML.replace(/<svg([^>]*)>/, '<svg$1 style="background:#1c1825">'+css);
    return new Blob(['<?xml version="1.0" encoding="UTF-8"?>\n'+body],{type:"image/svg+xml"});
  }
  function download(blob,name){
    const url=URL.createObjectURL(blob);const a=document.createElement("a");
    a.href=url;a.download=name;document.body.appendChild(a);a.click();
    setTimeout(()=>{URL.revokeObjectURL(url);a.remove();},100);
  }
  function dlSVG(id,name){const e=$(id);if(!e||!e.querySelector("*")){$("t3-status").textContent="Nothing to save yet — run training / draw first.";return;}download(svgToBlob(e),name);}

  function drawPolicies(res,nmNote){
    $("t3-pol-title").textContent=`Policy comparison — π*_Ω (dashed, open circles) vs π_θ (solid, squares) · n_mc=${nmNote} · seed mean · target peak=${targetPeak.toFixed(2)}`;
    const svg=$("t3-pols");svg.innerHTML="";
    const PW=258,PH=142,X0=[18,300],Y0=[46,238,430,622];
    /* legend */
    svg.appendChild(el("line",{x1:140,y1:16,x2:170,y2:16,stroke:"#ece7df","stroke-width":1.2,"stroke-dasharray":"4 3",opacity:0.8}));
    svg.appendChild(el("circle",{cx:155,cy:16,r:2.8,fill:"#181420",stroke:"#ece7df","stroke-width":1}));
    svg.appendChild(el("text",{x:176,y:20,fill:"#9b93a8","font-size":10},"π* (ground truth)"));
    svg.appendChild(el("line",{x1:312,y1:16,x2:342,y2:16,stroke:"#e84393","stroke-width":2}));
    svg.appendChild(el("rect",{x:324.5,y:13.5,width:5,height:5,fill:"#e84393"}));
    svg.appendChild(el("text",{x:348,y:20,fill:"#9b93a8","font-size":10},"π_θ (DPO-trained)"));
    REGKEYS.forEach((rk,i)=>{
      const x0=X0[i%2],y0=Y0[(i/2)|0];
      const sol=solveDP(rk,rewards,uniformPis(),alphas[rk],GAMMA,EPS,DEPTH,refPol);
      const pols=res[rk].pols;
      const star=[],est=[];
      for(let l=0;l<DEPTH;l++)for(let s=0;s<SN;s++)for(let a=0;a<3;a++){
        star.push(sol.pistar[l][s][a]);
        est.push(pols.reduce((acc,P)=>acc+P[l][s][a],0)/pols.length);
      }
      const ymax=Math.max(...star,...est)*1.14;
      const NPTS=DEPTH*SN*3;
      const X=idx=>x0+6+(PW-12)*idx/Math.max(NPTS-1,1);
      const Y=v=>y0+PH-14-(PH-24)*(v/ymax);
      svg.appendChild(el("rect",{x:x0,y:y0,width:PW,height:PH,fill:"#181420",stroke:"#2c2738",rx:8}));
      for(let li=1;li<DEPTH;li++){const b=li*SN*3;svg.appendChild(el("line",{x1:X(b-0.5),y1:y0+8,x2:X(b-0.5),y2:y0+PH-12,stroke:"#2c2738","stroke-width":0.8,"stroke-dasharray":"2 4"}));}
      for(let li=0;li<DEPTH;li++){
        const t=el("text",{x:X(li*SN*3+(SN*3-1)/2),y:y0+PH-3,fill:"#6a6378","font-size":8.5,"text-anchor":"middle"},"l="+li);
        t.setAttribute("class","mono");svg.appendChild(t);
      }
      const title=el("text",{x:x0+4,y:y0-7,fill:REG[rk].color,"font-size":11.5},`${REG[rk].label}  (α=${alphas[rk].toFixed(2)})`);
      title.setAttribute("class","math");svg.appendChild(title);
      svg.appendChild(el("polyline",{points:star.map((v,j)=>X(j).toFixed(1)+","+Y(v).toFixed(1)).join(" "),fill:"none",stroke:"#ece7df","stroke-width":1.1,"stroke-dasharray":"4 3",opacity:0.75}));
      svg.appendChild(el("polyline",{points:est.map((v,j)=>X(j).toFixed(1)+","+Y(v).toFixed(1)).join(" "),fill:"none",stroke:REG[rk].color,"stroke-width":1.8}));
      star.forEach((v,j)=>svg.appendChild(el("circle",{cx:X(j),cy:Y(v),r:2.2,fill:"#181420",stroke:"#ece7df","stroke-width":0.9,opacity:0.85})));
      est.forEach((v,j)=>svg.appendChild(el("rect",{x:X(j)-2.1,y:Y(v)-2.1,width:4.2,height:4.2,fill:REG[rk].color})));
    });
  }

  /* controls */
  $("t3-peak").addEventListener("input",e=>{targetPeak=parseFloat(e.target.value);$("t3-peak-v").textContent=targetPeak.toFixed(2);});
  $("t3-peak").addEventListener("change",()=>{calibrate();$("t3-status").textContent=`α recalibrated — target peak=${targetPeak.toFixed(2)}. Run training to compare at these α.`;});
  $("t3-nmc").addEventListener("input",e=>{nmc=parseInt(e.target.value);$("t3-nmc-v").textContent=nmc;});
  $("t3-np").addEventListener("input",e=>{npairs=parseInt(e.target.value);$("t3-np-v").textContent=npairs;});
  $("t3-sd").addEventListener("input",e=>{nseeds=parseInt(e.target.value);$("t3-sd-v").textContent=nseeds;});
  $("t3-dep").addEventListener("input",e=>{DEPTH=parseInt(e.target.value);$("t3-dep-v").textContent=DEPTH;});
  $("t3-dep").addEventListener("change",()=>{
    rewards=newRewards(DEPTH);makeDataset();calibrate();drawMDP();
    $("t3-status").textContent=`depth L=${DEPTH} applied — new MDP of this depth, data rebuilt, α recalibrated.`;
  });
  $("t3-eps").addEventListener("input",e=>{EPS=parseFloat(e.target.value);$("t3-eps-v").textContent=EPS.toFixed(2);});
  $("t3-eps").addEventListener("change",()=>{makeDataset();calibrate();$("t3-status").textContent=`ε=${EPS.toFixed(2)} applied — transitions changed, so data was rebuilt and α recalibrated.`;});
  $("t3-bs").addEventListener("input",e=>{BATCH=Math.pow(2,parseInt(e.target.value));$("t3-bs-v").textContent=BATCH;});
  $("t3-mode").querySelectorAll("button").forEach(b=>b.addEventListener("click",()=>{
    $("t3-mode").querySelectorAll("button").forEach(x=>x.classList.remove("on"));b.classList.add("on");gradMode=b.dataset.v;
    $("t3-status").textContent=`gradient mode = ${gradMode} (${gradMode==="A"?"exact ∂Ĉ/∂θ propagation":"scale-only approximation"}). Run training to compare.`;
  }));
  $("t3-offp").querySelectorAll("button").forEach(b=>b.addEventListener("click",()=>{
    $("t3-offp").querySelectorAll("button").forEach(x=>x.classList.remove("on"));b.classList.add("on");
    offPolicy=(b.dataset.v==="off");
    $("t3-nmc").disabled=offPolicy;
    $("t3-nmc").style.opacity=offPolicy?0.4:1;
    const ct=$("t3-cstats-title");
    if(ct)ct.textContent=offPolicy
      ? "C_Ω statistics — extreme off-policy: solid bar = spread across states, faint bar = |bias| proxy of the data-a' estimate (uniform a' vs π* expectation) (log scale)"
      : "C_Ω statistics — at the calibrated π*_Ω · solid bar = spread across states std_s[C], faint bar = std of a 1-sample estimate (log scale)";
    $("t3-status").textContent=offPolicy
      ?"Extreme off-policy — inner term evaluated only at the data-recorded a' (1-sample, n_mc ignored). Sharpest demonstration that KL is the only divergence admissible from data alone."
      :"On-policy inner — resample n_mc fresh actions a' from π(·|s'). The n_mc slider matters again.";
    drawCStats();
  }));
  $("t3-newmdp").addEventListener("click",()=>{
    rewards=newRewards(DEPTH);makeDataset();calibrate();
    $("t3-status").textContent="New MDP and preference data generated, α recalibrated — all seven Ω share the same data.";
  });
  function summarize(out){
    const o={meta:{targetPeak,EPS,BATCH,DEPTH,gradMode,offPolicy,npairs,nseeds,STEPS,GAMMA,LR},
      alphas:Object.fromEntries(REGKEYS.map(rk=>[rk,alphas[rk]])),
      gaps:{}};
    for(const nm of Object.keys(out)){o.gaps[nm]={};
      for(const rk of REGKEYS){const f=out[nm][rk].finals;
        const mu=f.reduce((a,b)=>a+b,0)/f.length;
        const sd=Math.sqrt(f.reduce((s,x)=>s+(x-mu)**2,0)/Math.max(f.length-1,1));
        o.gaps[nm][rk]={mean:mu,std:sd,finals:f};}}
    return o;
  }
  $("t3-run").addEventListener("click",()=>{
    if(running)return;
    runAll([nmc],out=>{
      lastRun=summarize(out);
      drawCurves(out[nmc]);
      drawBars(out,`Final policy gap @ n_mc=${nmc} (± seed std)`);
      drawPolicies(out[nmc],nmc);
      $("t3-status").textContent=`Done — n_mc=${nmc}, N_pairs=${npairs}, seeds=${nseeds}, calibrated α. Compare KL stability vs non-KL noise in the policy panels.`;
    });
  });
  $("t3-sweep").addEventListener("click",()=>{
    if(running)return;
    runAll([1,4,16],out=>{
      lastRun=summarize(out);
      drawCurves(out[1]);
      drawBars(out,"n_mc sweep — final policy gap (± seed std) · curves & policy panels shown at n_mc=1");
      drawPolicies(out[1],1);
      $("t3-status").textContent="Sweep done — KL bars stay flat across n_mc, non-KL bars fall as n_mc grows: the off-policy admissibility fingerprint.";
    });
  });
  const stamp=()=>`peak${targetPeak.toFixed(2)}_eps${EPS.toFixed(2)}_b${BATCH}_L${DEPTH}_${gradMode}${offPolicy?"_offpol":""}`;
  $("t3-dl-sweep").addEventListener("click",()=>dlSVG("t3-sweep-plot",`alpha_peak_sweep_${stamp()}.svg`));
  $("t3-dl-cprobe").addEventListener("click",()=>dlSVG("t3-cprobe",`c_admissibility_probe_${stamp()}.svg`));
  ["c0","c1","c2"].forEach((id,k)=>{
    $("t3-"+id).addEventListener("input",e=>{CVALS[k]=parseFloat(e.target.value);$("t3-"+id+"-v").textContent=CVALS[k].toFixed(2);drawCProbe();});
  });
  $("t3-cidx").querySelectorAll("button").forEach(b=>b.addEventListener("click",()=>{
    $("t3-cidx").querySelectorAll("button").forEach(x=>x.classList.remove("on"));b.classList.add("on");
    cShowMode=b.dataset.v;drawCProbe();
  }));
  $("t3-dl-curves").addEventListener("click",()=>dlSVG("t3-curves",`training_curves_${stamp()}.svg`));
  $("t3-dl-bars").addEventListener("click",()=>dlSVG("t3-bars",`policy_gap_bars_${stamp()}.svg`));
  $("t3-dl-pols").addEventListener("click",()=>dlSVG("t3-pols",`policy_panels_${stamp()}.svg`));
  $("t3-dl-json").addEventListener("click",()=>{
    if(!lastRun){$("t3-status").textContent="No results to save yet — run training first.";return;}
    download(new Blob([JSON.stringify(lastRun,null,2)],{type:"application/json"}),`admissibility_results_${stamp()}.json`);
  });
  makeDataset();calibrate();
  return {};

}
