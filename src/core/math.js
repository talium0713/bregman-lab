/* ═══════════════════ shared math: 3 actions, ARBITRARY π_ref (default uniform ⅓) ═══════════════════
 *
 * π_ref generalization (mirrors python/regularizers.py design principle 2):
 *   Every quantity accepts an optional reference vector `ref` (length |A|, sums to 1) and respects it.
 *   Omit `ref` and it defaults to uniform — so all existing callers keep identical behavior.
 *   With u_a := π_a / ref_a :
 *       f-divergence:  Ω(π;ref) = Σ_a ref_a f(u_a),   [∇Ω]_a = f'(u_a)
 *       Bregman (euc): Ω(π;ref) = ½ Σ_a (π_a − ref_a)²
 *   The inner integrand Φ(u)=f'(u)−f(u)/u is a function of u alone (ref enters only through u),
 *   so the closed-form Φ/Φ' below are ref-independent; only u changes with ref.
 *   soft-argmax is the unified ν-bisection:  π_a = ref_a·(f')^{-1}((Q_a−ν)/α), ν s.t. Σπ=1.
 */
const L=4, SN=3, NA=3, EPSI=0.02;
const clampN=(x,a,b)=>Math.min(b,Math.max(a,x));
const lg=x=>Math.log(Math.max(x,1e-12));
const rnd=(a,b)=>a+Math.random()*(b-a);
const normS=v=>{const c=v.map(x=>clampN(x,EPSI,1));const s=c[0]+c[1]+c[2];return c.map(x=>x/s);};
const softmax3=(q,al)=>{const m=Math.max(...q);const w=q.map(x=>Math.exp((x-m)/al));const z=w[0]+w[1]+w[2];return[w[0]/z,w[1]/z,w[2]/z];};
function projS3(y){const u=[...y].sort((a,b)=>b-a);let css=0,th=0;for(let i=0;i<3;i++){css+=u[i];const t=(css-1)/(i+1);if(u[i]-t>0)th=t;}return y.map(v=>Math.max(v-th,0));}

/* π_ref helpers */
const uniformRef=na=>Array.from({length:na||NA},()=>1/(na||NA));
const asRef=(ref,na)=>{
  if(!ref)return uniformRef(na);
  const s=ref.reduce((a,b)=>a+b,0);
  return Math.abs(s-1)>1e-9?ref.map(x=>x/s):ref;
};
/* normalize a ref spec into (depth,SN,NA): null→uniform, (NA,)→broadcast, (depth,SN,NA)→verbatim */
const resolveRef=(ref,depth,ns,na)=>{
  ns=ns||SN;na=na||NA;
  if(!ref)return Array.from({length:depth},()=>Array.from({length:ns},()=>uniformRef(na)));
  if(Array.isArray(ref[0])&&Array.isArray(ref[0][0]))return ref;        // (depth,ns,na)
  const r=asRef(ref,na);                                               // (na,) → broadcast
  return Array.from({length:depth},()=>Array.from({length:ns},()=>r.slice()));
};

/* ── unified soft-argmax: π_a = ref_a·(f')^{-1}((Q_a−ν)/α); ν found by bracketing+bisection ── */
function argmaxNu(R,Q,al,ref){
  ref=asRef(ref,Q.length);
  if(R.key==="kl"){                                                     // closed form π_a ∝ ref_a·exp(Q_a/α)
    const z=Q.map(q=>q/al);const m=Math.max(...z);
    const w=z.map((zi,a)=>ref[a]*Math.exp(zi-m));const Z=w.reduce((a,b)=>a+b,0);
    return w.map(x=>x/Z);
  }
  const piOf=nu=>R.type==="euc"
    ? Q.map((qa,a)=>Math.max(ref[a]+(qa-nu)/al,0))
    : Q.map((qa,a)=>ref[a]*Math.max(R.inv((qa-nu)/al),0));
  const S=nu=>piOf(nu).reduce((a,b)=>a+b,0)-1;
  let lo,hi;
  if(R.nuBracket){const b=R.nuBracket(Q,al);lo=b[0];hi=b[1];}
  else{lo=Math.min(...Q)-10*al-10;hi=Math.max(...Q)+10*al+10;}
  for(let i=0;i<60;i++){if(S(lo)>0)break;lo-=(hi-lo);}                  // expand until bracket straddles root
  for(let i=0;i<60;i++){if(S(hi)<0)break;hi+=(hi-lo);}
  for(let i=0;i<100;i++){const mid=0.5*(lo+hi);if(S(mid)>0)lo=mid;else hi=mid;}
  const pi=piOf(0.5*(lo+hi));const Z=pi.reduce((a,b)=>a+b,0);
  return pi.map(x=>x/Z);
}

/* ── per-divergence scalar building blocks: f, f'(=fp), f''(=fpp), inverse marginal (f')^{-1}(=inv) ── */
const fKL=t=>t*lg(t),                          fpKL=t=>lg(t)+1,            fppKL=t=>1/t;
const fAD=t=>(Math.pow(t,1.5)-1.5*t+0.5)/0.75, fpAD=t=>2*(Math.sqrt(t)-1), fppAD=t=>1/Math.sqrt(t);   // α-div a=1.5
const fRK=t=>-lg(t),                           fpRK=t=>-1/t,               fppRK=t=>1/(t*t);
const fJS=t=>t*lg(2*t/(1+t))+lg(2/(1+t)),      fpJS=t=>lg(2*t/(1+t)),      fppJS=t=>1/(t*(1+t));
const fHE=t=>(Math.sqrt(t)-1)**2,              fpHE=t=>1-1/Math.sqrt(t),   fppHE=t=>0.5*Math.pow(t,-1.5);
const fX2=t=>(t-1)**2,                         fpX2=t=>2*(t-1),            fppX2=t=>2;

/* build a Regularizer with ref-aware methods (ref optional → uniform) */
function makeReg(o){
  const R={...o};
  if(o.type==="euc"){
    R.omega=(p,ref)=>{ref=asRef(ref,p.length);return 0.5*p.reduce((s,x,a)=>s+(x-ref[a])**2,0);};
    R.grad =(p,ref)=>{ref=asRef(ref,p.length);return p.map((x,a)=>x-ref[a]);};
    R.breg =(p,q)=>0.5*p.reduce((s,x,a)=>s+(x-q[a])**2,0);              // ½‖p−q‖² (ref-independent)
  }else{
    R.omega=(p,ref)=>{ref=asRef(ref,p.length);return p.reduce((s,x,a)=>s+ref[a]*o.f(Math.max(x,1e-12)/ref[a]),0);};
    R.grad =(p,ref)=>{ref=asRef(ref,p.length);return p.map((x,a)=>o.fp(Math.max(x,1e-12)/ref[a]));};
    R.breg =(p,q,ref)=>{ref=asRef(ref,p.length);return p.reduce((s,x,a)=>{
      const up=Math.max(x,1e-12)/ref[a],uq=Math.max(q[a],1e-12)/ref[a];
      return s+ref[a]*(o.f(up)-o.f(uq)-o.fp(uq)*(up-uq));},0);};
  }
  R.gradDPO=R.grad;                                                     // implicit-reward chosen term α·[∇Ω]_a
  R.dGdTheta=(p,a,b,ref)=>{                                             // ∂/∂θ_b of [∇Ω]_a, π=softmax(θ)
    ref=asRef(ref,p.length);const d=(a===b?1:0)-p[b];
    if(o.type==="euc")return p[a]*d;
    const ua=Math.max(p[a],1e-12)/ref[a];
    return o.fpp(ua)*(1/ref[a])*p[a]*d;                                 // f''(u_a)·(1/ref_a)·π_a(δ−π_b)
  };
  R.argmax=(Q,al,ref)=>argmaxNu(R,Q,al,ref);
  return R;
}

const REG = {
  kl:makeReg({key:"kl",label:"reverse KL",shortLabel:"RKL",color:"#e84393",type:"fdiv",essSmooth:true,
    f:fKL,fp:fpKL,fpp:fppKL,inv:y=>Math.exp(y-1),
    Phi:u=>1, dPhi:null, constC:true, range:2.3}),
  adiv:makeReg({key:"adiv",label:"α-div (a=1.5)",shortLabel:"α-div",color:"#c455b8",type:"fdiv",essSmooth:false,
    f:fAD,fp:fpAD,fpp:fppAD,inv:y=>Math.pow(Math.max(1+0.5*y,0),2),
    Phi:u=>(2/3)*(Math.sqrt(u)-1/u), dPhi:u=>(2/3)*(0.5/Math.sqrt(u)+1/(u*u)), constC:false, range:2.0}),
  rkl:makeReg({key:"rkl",label:"forward KL",shortLabel:"FKL",color:"#9b59d0",type:"fdiv",essSmooth:true,
    f:fRK,fp:fpRK,fpp:fppRK,inv:y=>-1/Math.min(y,-1e-12),
    nuBracket:(Q,al)=>[Math.max(...Q)+1e-10,Math.max(...Q)+1000*al+10],
    Phi:u=>(lg(u)-1)/u, dPhi:u=>(2-lg(u))/(u*u), constC:false, range:4.5}),
  js:makeReg({key:"js",label:"JS",shortLabel:"JS",color:"#6f6ae0",type:"fdiv",essSmooth:true,
    f:fJS,fp:fpJS,fpp:fppJS,inv:y=>{const e=Math.exp(Math.min(y,0.6931));return e/Math.max(2-e,1e-12);},
    nuBracket:(Q,al)=>[Math.max(...Q)-al*Math.LN2+1e-9,Math.max(...Q)+60*al],
    Phi:u=>lg((1+u)/2)/u, dPhi:u=>((u/(1+u))-lg((1+u)/2))/(u*u), constC:false, range:1.8}),
  hel:makeReg({key:"hel",label:"sq. Hellinger",shortLabel:"Hel",color:"#5585e0",type:"fdiv",essSmooth:true,
    f:fHE,fp:fpHE,fpp:fppHE,inv:y=>{const r=1/(1-Math.min(y,0.999999));return r*r;},
    nuBracket:(Q,al)=>[Math.max(...Q)-al+1e-9,Math.max(...Q)+60*al],
    Phi:u=>1/Math.sqrt(u)-1/u, dPhi:u=>-0.5*Math.pow(u,-1.5)+1/(u*u), constC:false, range:2.4}),
  chi2:makeReg({key:"chi2",label:"Pearson χ²",shortLabel:"χ²",color:"#4fa8d8",type:"fdiv",essSmooth:false,
    f:fX2,fp:fpX2,fpp:fppX2,inv:y=>Math.max(1+y/2,0),
    nuBracket:(Q,al)=>[Math.min(...Q)-10*al-10,Math.max(...Q)+10*al+10],
    Phi:u=>2*(u-1)-((u-1)**2)/u, dPhi:u=>1+1/(u*u), constC:false, range:2.8}),
  euc:makeReg({key:"euc",label:"sq. Euclidean",shortLabel:"Euc",color:"#5bc8d6",type:"euc",essSmooth:false,
    Phi:null, dPhi:null, constC:false, range:0.95}),
};
const REGKEYS=["kl","adiv","rkl","js","hel","chi2","euc"];

const transP=(a,eps)=>[0,1,2].map(s2=>s2===a?1-eps:eps/2);
const sampleCat=pr=>{let u=Math.random();for(let i=0;i<pr.length;i++){u-=pr[i];if(u<=0)return i;}return pr.length-1;};
const newRewards=(depth)=>Array.from({length:depth||L},()=>Array.from({length:SN},()=>[rnd(-0.8,0.8),rnd(-0.8,0.8),rnd(-0.8,0.8)]));
const randPi=()=>normS([rnd(0.05,1),rnd(0.05,1),rnd(0.05,1)]);

/* regularized backward-induction solver. `ref` (optional): null→uniform, (NA,) vector, or (depth,SN,NA). */
function solveDP(regK, r, pis, alpha, gamma, eps, depth, ref){
  const R=REG[regK];const LL=depth||L;
  const refA=resolveRef(ref,LL,SN,NA);
  const Qs=[],Qp=[],pistar=[],Vs=[],Vp=[],B=[];
  for(let l=0;l<LL;l++){Qs.push([]);Qp.push([]);pistar.push([]);Vs.push(new Array(SN));Vp.push(new Array(SN));B.push(new Array(SN));}
  let boundary=false;
  const EV=(V,l,a)=>l===LL-1?0:transP(a,eps).reduce((acc,p,s2)=>acc+p*V[l+1][s2],0);
  for(let l=LL-1;l>=0;l--)for(let s=0;s<SN;s++){
    const q=[0,1,2].map(a=>r[l][s][a]+gamma*EV(Vs,l,a));Qs[l][s]=q;
    const ps=R.argmax(q,alpha,refA[l][s]);if(Math.min(...ps)<1e-7)boundary=true;
    pistar[l][s]=ps;Vs[l][s]=ps[0]*q[0]+ps[1]*q[1]+ps[2]*q[2]-alpha*R.omega(ps.map(x=>Math.max(x,1e-12)),refA[l][s]);
  }
  for(let l=LL-1;l>=0;l--)for(let s=0;s<SN;s++){
    const q=[0,1,2].map(a=>r[l][s][a]+gamma*EV(Vp,l,a));Qp[l][s]=q;
    const p=pis[l][s];Vp[l][s]=p[0]*q[0]+p[1]*q[1]+p[2]*q[2]-alpha*R.omega(p,refA[l][s]);
    B[l][s]=R.breg(p,pistar[l][s].map(x=>Math.max(x,1e-12)),refA[l][s]);
  }
  return {Qs,Qp,pistar,Vs,Vp,B,boundary};
}

export { L, SN, NA, EPSI, clampN, lg, rnd, normS, softmax3, projS3, REG, REGKEYS, transP, sampleCat, newRewards, randPi, solveDP, uniformRef, asRef, resolveRef };
