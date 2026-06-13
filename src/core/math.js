/* ═══════════════════ shared math: 3 actions, ref = uniform(⅓) ═══════════════════ */
const L=4, SN=3, EPSI=0.02;
const clampN=(x,a,b)=>Math.min(b,Math.max(a,x));
const lg=x=>Math.log(Math.max(x,1e-12));
const rnd=(a,b)=>a+Math.random()*(b-a);
const normS=v=>{const c=v.map(x=>clampN(x,EPSI,1));const s=c[0]+c[1]+c[2];return c.map(x=>x/s);};
const softmax3=(q,al)=>{const m=Math.max(...q);const w=q.map(x=>Math.exp((x-m)/al));const z=w[0]+w[1]+w[2];return[w[0]/z,w[1]/z,w[2]/z];};
function projS3(y){const u=[...y].sort((a,b)=>b-a);let css=0,th=0;for(let i=0;i<3;i++){css+=u[i];const t=(css-1)/(i+1);if(u[i]-t>0)th=t;}return y.map(v=>Math.max(v-th,0));}

/* f-divergence helper: Ω(π)=Σ ref·f(u), u=π/ref, ref=⅓ */
const mkF=(f,fp)=>({
  omega:p=>p.reduce((s,x)=>s+(1/3)*f(3*Math.max(x,1e-12)),0),
  grad:p=>p.map(x=>fp(3*Math.max(x,1e-12))),
  breg:(p,q)=>p.reduce((s,pi,i)=>{const up=3*Math.max(pi,1e-12),uq=3*Math.max(q[i],1e-12);
    return s+(1/3)*(f(up)-f(uq)-fp(uq)*(up-uq));},0),
});
const bis=(fn,lo,hi,it)=>{for(let i=0;i<it;i++){const m=(lo+hi)/2;if(fn(m)>0)lo=m;else hi=m;}return(lo+hi)/2;};
const fKL=t=>t*lg(t),                          fpKL=t=>lg(t)+1;
const fAD=t=>(Math.pow(t,1.5)-1.5*t+0.5)/0.75, fpAD=t=>2*(Math.sqrt(t)-1);
const fRK=t=>-lg(t),                           fpRK=t=>-1/t;
const fJS=t=>t*lg(2*t/(1+t))+lg(2/(1+t)),      fpJS=t=>lg(2*t/(1+t));
const fHE=t=>(Math.sqrt(t)-1)**2,              fpHE=t=>1-1/Math.sqrt(t);
const fX2=t=>(t-1)**2,                         fpX2=t=>2*(t-1);

const REG = {
  kl:{label:"KL",shortLabel:"KL",color:"#e84393",essSmooth:true,...mkF(fKL,fpKL),
    argmax:(Q,al)=>softmax3(Q,al),
    gradDPO:p=>p.map(x=>lg(3*x)+1),
    dGdTheta:(p,a,b)=>(a===b?1:0)-p[b],
    Phi:u=>1, dPhi:null, constC:true, range:2.3},
  adiv:{label:"α-div (a=1.5)",shortLabel:"α-div",color:"#c455b8",essSmooth:false,...mkF(fAD,fpAD),
    argmax:(Q,al)=>{
      const sum=c=>Q.reduce((s,qa)=>{const y=(qa-c)/al;const r=Math.max((y+2)/2,0);return s+(1/3)*r*r;},0)-1;
      const c=bis(sum,Math.min(...Q)-40*al,Math.max(...Q)+2*al,90);
      const p=Q.map(qa=>{const y=(qa-c)/al;const r=Math.max((y+2)/2,0);return (1/3)*r*r;});
      const z=p[0]+p[1]+p[2];return p.map(x=>x/z);},
    gradDPO:p=>p.map(x=>fpAD(3*Math.max(x,1e-12))),
    dGdTheta:(p,a,b)=>3*p[a]*((a===b?1:0)-p[b])/Math.sqrt(3*Math.max(p[a],1e-12)),
    Phi:u=>(2/3)*(Math.sqrt(u)-1/u), dPhi:u=>(2/3)*(0.5/Math.sqrt(u)+1/(u*u)), constC:false, range:2.0},
  rkl:{label:"Reverse KL",shortLabel:"RKL",color:"#9b59d0",essSmooth:true,...mkF(fRK,fpRK),
    argmax:(Q,al)=>{const mx=Math.max(...Q);
      const sum=c=>Q.reduce((s,qa)=>s+al/3/(c-qa),0)-1;
      const c=bis(sum,mx+1e-10,mx+1000*al+10,90);
      const p=Q.map(qa=>al/3/(c-qa));const z=p[0]+p[1]+p[2];return p.map(x=>x/z);},
    gradDPO:p=>p.map(x=>-1/(3*x)),
    dGdTheta:(p,a,b)=>((a===b?1:0)-p[b])/(3*p[a]),
    Phi:u=>(lg(u)-1)/u, dPhi:u=>(2-lg(u))/(u*u), constC:false, range:4.5},
  js:{label:"JS",shortLabel:"JS",color:"#6f6ae0",essSmooth:true,...mkF(fJS,fpJS),
    argmax:(Q,al)=>{const mx=Math.max(...Q);
      const tOf=y=>{const e=Math.exp(Math.min(y,0.6931));return e/Math.max(2-e,1e-12);};
      const sum=c=>Q.reduce((s,qa)=>s+(1/3)*tOf((qa-c)/al),0)-1;
      const c=bis(sum,mx-al*Math.LN2+1e-9,mx+60*al,90);
      const p=Q.map(qa=>(1/3)*tOf((qa-c)/al));
      const z=p[0]+p[1]+p[2];return p.map(x=>x/z);},
    gradDPO:p=>p.map(x=>fpJS(3*Math.max(x,1e-12))),
    dGdTheta:(p,a,b)=>((a===b?1:0)-p[b])/(1+3*p[a]),
    Phi:u=>lg((1+u)/2)/u, dPhi:u=>((u/(1+u))-lg((1+u)/2))/(u*u), constC:false, range:1.8},
  hel:{label:"sq. Hellinger",shortLabel:"Hel",color:"#5585e0",essSmooth:true,...mkF(fHE,fpHE),
    argmax:(Q,al)=>{const mx=Math.max(...Q);
      const tOf=y=>{const r=1/(1-Math.min(y,0.999999));return r*r;};
      const sum=c=>Q.reduce((s,qa)=>s+(1/3)*tOf((qa-c)/al),0)-1;
      const c=bis(sum,mx-al+1e-9,mx+60*al,90);
      const p=Q.map(qa=>(1/3)*tOf((qa-c)/al));
      const z=p[0]+p[1]+p[2];return p.map(x=>x/z);},
    gradDPO:p=>p.map(x=>fpHE(3*Math.max(x,1e-12))),
    dGdTheta:(p,a,b)=>1.5*p[a]*((a===b?1:0)-p[b])/Math.pow(3*Math.max(p[a],1e-12),1.5),
    Phi:u=>1/Math.sqrt(u)-1/u, dPhi:u=>-0.5*Math.pow(u,-1.5)+1/(u*u), constC:false, range:2.4},
  chi2:{label:"Pearson χ²",shortLabel:"χ²",color:"#4fa8d8",essSmooth:false,...mkF(fX2,fpX2),
    argmax:(Q,al)=>projS3(Q.map(q=>1/3+q/(6*al))),
    gradDPO:p=>p.map(x=>2*(3*x-1)),
    dGdTheta:(p,a,b)=>6*p[a]*((a===b?1:0)-p[b]),
    Phi:u=>2*(u-1)-((u-1)**2)/u, dPhi:u=>1+1/(u*u), constC:false, range:2.8},
  euc:{label:"sq. Euclidean",shortLabel:"Euc",color:"#5bc8d6",essSmooth:false,
    omega:p=>0.5*p.reduce((s,x)=>s+(x-1/3)**2,0),
    grad:p=>p.map(x=>x-1/3),
    breg:(p,q)=>0.5*p.reduce((s,pi,i)=>s+(pi-q[i])**2,0),
    argmax:(Q,al)=>projS3(Q.map(q=>1/3+q/al)),
    gradDPO:p=>p.map(x=>x-1/3),
    dGdTheta:(p,a,b)=>p[a]*((a===b?1:0)-p[b]),
    Phi:null, dPhi:null, constC:false, range:0.95},
};
const REGKEYS=["kl","adiv","rkl","js","hel","chi2","euc"];

const transP=(a,eps)=>[0,1,2].map(s2=>s2===a?1-eps:eps/2);
const sampleCat=pr=>{let u=Math.random();for(let i=0;i<pr.length;i++){u-=pr[i];if(u<=0)return i;}return pr.length-1;};
const newRewards=(depth)=>Array.from({length:depth||L},()=>Array.from({length:SN},()=>[rnd(-0.8,0.8),rnd(-0.8,0.8),rnd(-0.8,0.8)]));
const randPi=()=>normS([rnd(0.05,1),rnd(0.05,1),rnd(0.05,1)]);

function solveDP(regK, r, pis, alpha, gamma, eps, depth){
  const R=REG[regK];const LL=depth||L;
  const Qs=[],Qp=[],pistar=[],Vs=[],Vp=[],B=[];
  for(let l=0;l<LL;l++){Qs.push([]);Qp.push([]);pistar.push([]);Vs.push(new Array(SN));Vp.push(new Array(SN));B.push(new Array(SN));}
  let boundary=false;
  const EV=(V,l,a)=>l===LL-1?0:transP(a,eps).reduce((acc,p,s2)=>acc+p*V[l+1][s2],0);
  for(let l=LL-1;l>=0;l--)for(let s=0;s<SN;s++){
    const q=[0,1,2].map(a=>r[l][s][a]+gamma*EV(Vs,l,a));Qs[l][s]=q;
    const ps=R.argmax(q,alpha);if(Math.min(...ps)<1e-7)boundary=true;
    pistar[l][s]=ps;Vs[l][s]=ps[0]*q[0]+ps[1]*q[1]+ps[2]*q[2]-alpha*R.omega(ps.map(x=>Math.max(x,1e-12)));
  }
  for(let l=LL-1;l>=0;l--)for(let s=0;s<SN;s++){
    const q=[0,1,2].map(a=>r[l][s][a]+gamma*EV(Vp,l,a));Qp[l][s]=q;
    const p=pis[l][s];Vp[l][s]=p[0]*q[0]+p[1]*q[1]+p[2]*q[2]-alpha*R.omega(p);
    B[l][s]=R.breg(p,pistar[l][s].map(x=>Math.max(x,1e-12)));
  }
  return {Qs,Qp,pistar,Vs,Vp,B,boundary};
}

export { L, SN, EPSI, clampN, lg, rnd, normS, softmax3, projS3, REG, REGKEYS, transP, sampleCat, newRewards, randPi, solveDP };
