/* global THREE */
import { L, SN, clampN, lg, rnd, normS, softmax3, projS3, REG, REGKEYS, transP, sampleCat, newRewards, randPi, solveDP } from "../core/math.js";
import { $, el, makeSeg } from "../core/dom.js";

export function initT1(){

  let regK="kl", mode="primal";
  let pi=normS([0.15,0.55,0.3]), pistar=normS([0.6,0.22,0.18]);
  let inited=false, renderer, scene, camera, surfGroup, dynGroup, hFn, Yfn;
  const cam={theta:0.55,phi:1.08,dist:4.3};

  /* sum-zero basis for dual coords */
  const D1=[1/Math.SQRT2,-1/Math.SQRT2,0], D2=[1/Math.sqrt(6),1/Math.sqrt(6),-2/Math.sqrt(6)];
  const toUV=q=>{const m=(q[0]+q[1]+q[2])/3;const c=[q[0]-m,q[1]-m,q[2]-m];
    return[c[0]*D1[0]+c[1]*D1[1]+c[2]*D1[2], c[0]*D2[0]+c[1]*D2[1]+c[2]*D2[2]];};
  const fromUV=(u,v)=>[u*D1[0]+v*D2[0],u*D1[1]+v*D2[1],u*D1[2]+v*D2[2]];

  const dualPi=q=>REG[regK].argmax(q,1);
  const omegaStar=q=>{
    const p=dualPi(q).map(x=>Math.max(x,1e-12));
    return p[0]*q[0]+p[1]*q[1]+p[2]*q[2]-REG[regK].omega(p);
  };
  const bregDual=(qs,qq)=>{
    const pq=dualPi(qq);
    return omegaStar(qs)-omegaStar(qq)-(pq[0]*(qs[0]-qq[0])+pq[1]*(qs[1]-qq[1])+pq[2]*(qs[2]-qq[2]));
  };

  const TA=[0,-1.05],TB=[-0.95,0.78],TC=[0.95,0.78];
  const baryXZ=l=>[l[0]*TA[0]+l[1]*TB[0]+l[2]*TC[0], l[0]*TA[1]+l[1]*TB[1]+l[2]*TC[1]];
  const xzBary=(x,z)=>{const d=(TB[1]-TC[1])*(TA[0]-TC[0])+(TC[0]-TB[0])*(TA[1]-TC[1]);
    const l1=((TB[1]-TC[1])*(x-TC[0])+(TC[0]-TB[0])*(z-TC[1]))/d;
    const l2=((TC[1]-TA[1])*(x-TC[0])+(TA[0]-TC[0])*(z-TC[1]))/d;
    return[l1,l2,1-l1-l2];};

  const RAMP=[[26,22,34],[62,32,64],[128,42,96],[225,70,148],[255,213,235]];
  const ramp=t=>{const x=clampN(t,0,1)*(RAMP.length-1);const i=Math.min(Math.floor(x),RAMP.length-2);const f=x-i;
    const a=RAMP[i],b=RAMP[i+1];return[(a[0]+(b[0]-a[0])*f)/255,(a[1]+(b[1]-a[1])*f)/255,(a[2]+(b[2]-a[2])*f)/255];};

  function buildHF(){
    if(mode==="primal")return (x,z)=>{const l=xzBary(x,z);if(l[0]<0.004||l[1]<0.004||l[2]<0.004)return null;
      return REG[regK].omega(normS(l));};
    const r=REG[regK].range;
    return (x,z)=>omegaStar(fromUV((x/1.05)*r,(z/1.05)*r));
  }

  function buildSurface(){
    surfGroup.clear();
    hFn=buildHF();
    let hMax=-1e18,hMin=1e18;
    for(let i=0;i<=40;i++)for(let j=0;j<=40;j++){
      const x=-1.05+2.1*i/40,z=-1.05+2.1*j/40;const h=hFn(x,z);if(h==null)continue;
      if(h>hMax)hMax=h;if(h<hMin)hMin=h;
    }
    const span=Math.max(hMax-hMin,1e-9), S=1.15/span;
    Yfn=h=>(h-hMin)*S;
    const positions=[],colors=[],indices=[];
    if(mode==="primal"){
      const N=52,idx=[];let k=0;
      for(let i=0;i<=N;i++){idx.push([]);
        for(let j=0;j<=N-i;j++){
          const l=[i/N,j/N,(N-i-j)/N];const[x,z]=baryXZ(l);
          const h=REG[regK].omega(normS(l));
          positions.push(x,Yfn(h),z);
          const c=ramp(Math.sqrt((h-hMin)/span));colors.push(c[0],c[1],c[2]);idx[i].push(k++);
        }}
      for(let i=0;i<N;i++)for(let j=0;j<N-i;j++){
        indices.push(idx[i][j],idx[i+1][j],idx[i][j+1]);
        if(j<N-i-1)indices.push(idx[i+1][j],idx[i+1][j+1],idx[i][j+1]);
      }
    }else{
      const N=56;
      for(let i=0;i<=N;i++)for(let j=0;j<=N;j++){
        const x=-1.05+2.1*i/N,z=-1.05+2.1*j/N;const h=hFn(x,z);
        positions.push(x,Yfn(h),z);
        const c=ramp(Math.sqrt((h-hMin)/span));colors.push(c[0],c[1],c[2]);
      }
      for(let i=0;i<N;i++)for(let j=0;j<N;j++){
        const a=i*(N+1)+j;indices.push(a,a+N+1,a+1,a+N+1,a+N+2,a+1);
      }
    }
    const geo=new THREE.BufferGeometry();
    geo.setAttribute("position",new THREE.Float32BufferAttribute(positions,3));
    geo.setAttribute("color",new THREE.Float32BufferAttribute(colors,3));
    geo.setIndex(indices);geo.computeVertexNormals();
    surfGroup.add(new THREE.Mesh(geo,new THREE.MeshPhongMaterial({vertexColors:true,side:THREE.DoubleSide,shininess:26,specular:new THREE.Color(0x332b44)})));
    const out= mode==="primal"
      ? [TA,TB,TC,TA].map(([x,z])=>new THREE.Vector3(x,0.001,z))
      : [[-1.05,-1.05],[1.05,-1.05],[1.05,1.05],[-1.05,1.05],[-1.05,-1.05]].map(([x,z])=>new THREE.Vector3(x,0.001,z));
    surfGroup.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(out),new THREE.LineBasicMaterial({color:0x4a4360})));
  }

  function buildDyn(){
    dynGroup.clear();
    const hAt=(x,z)=>{const h=hFn(x,z);return h==null?0:Yfn(h);};
    let anchor,probe;
    if(mode==="primal"){anchor=baryXZ(pistar);probe=baryXZ(pi);}
    else{
      const r=REG[regK].range;
      const[us,vs]=toUV(REG[regK].grad(pistar.map(x=>Math.max(x,1e-9))));
      const[up,vp]=toUV(REG[regK].grad(pi));
      anchor=[(clampN(up,-r,r)/r)*1.05,(clampN(vp,-r,r)/r)*1.05];
      probe=[(clampN(us,-r,r)/r)*1.05,(clampN(vs,-r,r)/r)*1.05];
    }
    const[ax,az]=anchor,[px,pz]=probe;const ay=hAt(ax,az);
    const d=0.012;
    const gx=(hAt(ax+d,az)-hAt(ax-d,az))/(2*d), gz=(hAt(ax,az+d)-hAt(ax,az-d))/(2*d);
    const planeY=(x,z)=>ay+gx*(x-ax)+gz*(z-az);
    const pd=0.6, corners=[[ax-pd,az-pd],[ax+pd,az-pd],[ax+pd,az+pd],[ax-pd,az+pd]];
    const pg=new THREE.BufferGeometry();
    pg.setAttribute("position",new THREE.Float32BufferAttribute(corners.flatMap(([x,z])=>[x,planeY(x,z),z]),3));
    pg.setIndex([0,1,2,0,2,3]);pg.computeVertexNormals();
    dynGroup.add(new THREE.Mesh(pg,new THREE.MeshBasicMaterial({color:0x8d7dd6,transparent:true,opacity:0.32,side:THREE.DoubleSide,depthWrite:false})));
    dynGroup.add(new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(corners.map(([x,z])=>new THREE.Vector3(x,planeY(x,z),z))),new THREE.LineBasicMaterial({color:0x8d7dd6})));
    const yS=hAt(px,pz),yP=planeY(px,pz),gapH=Math.abs(yS-yP);
    if(gapH>1e-4){
      const cyl=new THREE.Mesh(new THREE.CylinderGeometry(0.014,0.014,gapH,8),new THREE.MeshBasicMaterial({color:0xe84393}));
      cyl.position.set(px,(yS+yP)/2,pz);dynGroup.add(cyl);
    }
    const mk=(x,z,col)=>{const m=new THREE.Mesh(new THREE.SphereGeometry(0.052,18,14),new THREE.MeshPhongMaterial({color:col,shininess:60}));m.position.set(x,hAt(x,z),z);dynGroup.add(m);};
    mk(ax,az, mode==="primal"?0xe84393:0x5bc8d6);
    mk(px,pz, mode==="primal"?0x5bc8d6:0xe84393);
  }

  function updateReadouts(){
    const Bp=REG[regK].breg(pi,pistar.map(x=>Math.max(x,1e-12)));
    const qs=REG[regK].grad(pistar.map(x=>Math.max(x,1e-9)));
    const qp=REG[regK].grad(pi);
    const Bd=bregDual(qs,qp);
    $("t1-bp").textContent=Bp.toFixed(5);
    $("t1-bd").textContent=Bd.toFixed(5);
    $("t1-diff").textContent="|차이| = "+Math.abs(Bp-Bd).toExponential(1);
  }

  /* minimap */
  function drawMap(){
    const svg=$("t1-map");svg.innerHTML="";
    const W=200,H=178,A=[W/2,14],B=[16,H-16],Cv=[W-16,H-16];
    const b2=(l)=>[l[0]*A[0]+l[1]*B[0]+l[2]*Cv[0], l[0]*A[1]+l[1]*B[1]+l[2]*Cv[1]];
    svg.appendChild(el("polygon",{points:`${A} ${B} ${Cv}`,fill:"#221c30",stroke:"#2c2738","stroke-width":1.4}));
    [["a₁",A[0],A[1]-4],["a₂",B[0]-2,B[1]+12],["a₃",Cv[0]+2,Cv[1]+12]].forEach(([t,x,y])=>{
      const tx=el("text",{x,y,fill:"#6a6378","font-size":10,"text-anchor":"middle"},t);tx.setAttribute("class","math");svg.appendChild(tx);});
    const[px,py]=b2(pi),[qx,qy]=b2(pistar);
    svg.appendChild(el("line",{x1:px,y1:py,x2:qx,y2:qy,stroke:"#ece7df","stroke-width":0.8,"stroke-dasharray":"3 3",opacity:0.45}));
    svg.appendChild(el("circle",{cx:qx,cy:qy,r:8,fill:"#e84393",stroke:"#13101a","stroke-width":2}));
    svg.appendChild(el("circle",{cx:px,cy:py,r:8,fill:"#5bc8d6",stroke:"#13101a","stroke-width":2}));
    const t1=el("text",{x:qx,y:qy-12,fill:"#e84393","font-size":11,"text-anchor":"middle"},"π*");t1.setAttribute("class","math");svg.appendChild(t1);
    const t2=el("text",{x:px,y:py-12,fill:"#5bc8d6","font-size":11,"text-anchor":"middle"},"π");t2.setAttribute("class","math");svg.appendChild(t2);
  }
  function mapHandlers(){
    const svg=$("t1-map");let drag=null;
    const W=200,H=178,A=[W/2,14],B=[16,H-16],Cv=[W-16,H-16];
    const xy2b=(x,y)=>{const d=(B[1]-Cv[1])*(A[0]-Cv[0])+(Cv[0]-B[0])*(A[1]-Cv[1]);
      const l1=((B[1]-Cv[1])*(x-Cv[0])+(Cv[0]-B[0])*(y-Cv[1]))/d;
      const l2=((Cv[1]-A[1])*(x-Cv[0])+(A[0]-Cv[0])*(y-Cv[1]))/d;
      return normS([l1,l2,1-l1-l2]);};
    const ev=e=>{const r=svg.getBoundingClientRect();return xy2b((e.clientX-r.left)/r.width*W,(e.clientY-r.top)/r.height*H);};
    svg.addEventListener("pointerdown",e=>{
      const l=ev(e);
      const b2=(v)=>{const A2=[W/2,14],B2=[16,H-16],C2=[W-16,H-16];return[v[0]*A2[0]+v[1]*B2[0]+v[2]*C2[0],v[0]*A2[1]+v[1]*B2[1]+v[2]*C2[1]];};
      const r=svg.getBoundingClientRect();
      const ex=(e.clientX-r.left)/r.width*W, ey=(e.clientY-r.top)/r.height*H;
      const[px,py]=b2(pi),[qx,qy]=b2(pistar);
      drag=((ex-px)**2+(ey-py)**2 < (ex-qx)**2+(ey-qy)**2)?"p":"q";
      svg.setPointerCapture(e.pointerId);
      if(drag==="p")pi=l;else pistar=l;refresh(false);
    });
    svg.addEventListener("pointermove",e=>{if(!drag)return;const l=ev(e);if(drag==="p")pi=l;else pistar=l;refresh(false);});
    const up=()=>drag=null;
    svg.addEventListener("pointerup",up);svg.addEventListener("pointercancel",up);
  }

  function refresh(rebuildSurf){
    if(!inited)return;
    if(rebuildSurf)buildSurface();
    buildDyn();drawMap();updateReadouts();
  }

  function ensure(){
    if(inited){refresh(false);return;}
    inited=true;
    const mount=$("t1-mount");
    const w=mount.clientWidth||560, h=Math.round(w*0.86);
    renderer=new THREE.WebGLRenderer({antialias:true});
    renderer.setPixelRatio(Math.min(window.devicePixelRatio,2));
    renderer.setSize(w,h);mount.appendChild(renderer.domElement);
    scene=new THREE.Scene();scene.background=new THREE.Color(0x1c1825);
    camera=new THREE.PerspectiveCamera(42,w/h,0.1,100);
    scene.add(new THREE.AmbientLight(0xffffff,0.6));
    const dl=new THREE.DirectionalLight(0xffffff,0.75);dl.position.set(2.5,4,3);scene.add(dl);
    surfGroup=new THREE.Group();dynGroup=new THREE.Group();scene.add(surfGroup);scene.add(dynGroup);
    (function tick(){
      camera.position.set(cam.dist*Math.sin(cam.phi)*Math.sin(cam.theta),cam.dist*Math.cos(cam.phi),cam.dist*Math.sin(cam.phi)*Math.cos(cam.theta));
      camera.lookAt(0,0.32,0);renderer.render(scene,camera);requestAnimationFrame(tick);
    })();
    /* rotate */
    let dragR=null;
    renderer.domElement.style.touchAction="none";renderer.domElement.style.cursor="grab";
    renderer.domElement.addEventListener("pointerdown",e=>{dragR={x:e.clientX,y:e.clientY};renderer.domElement.setPointerCapture(e.pointerId);});
    renderer.domElement.addEventListener("pointermove",e=>{if(!dragR)return;
      cam.theta-=(e.clientX-dragR.x)*0.0065;cam.phi=clampN(cam.phi-(e.clientY-dragR.y)*0.005,0.25,1.45);
      dragR={x:e.clientX,y:e.clientY};});
    const up=()=>dragR=null;
    renderer.domElement.addEventListener("pointerup",up);renderer.domElement.addEventListener("pointercancel",up);
    $("t1-zin").addEventListener("click",()=>cam.dist=clampN(cam.dist-0.55,2.6,7));
    $("t1-zout").addEventListener("click",()=>cam.dist=clampN(cam.dist+0.55,2.6,7));
    makeSeg($("t1-reg"),REGKEYS,regK,k=>{regK=k;refresh(true);});
    $("t1-mode").querySelectorAll("button").forEach(b=>b.addEventListener("click",()=>{
      $("t1-mode").querySelectorAll("button").forEach(x=>x.classList.remove("on"));b.classList.add("on");
      mode=b.dataset.v;refresh(true);
    }));
    mapHandlers();
    buildSurface();refresh(false);
  }
  return {ensure};

}
