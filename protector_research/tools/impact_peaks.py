import glob, os, sys, numpy as np, pandas as pd
from scipy.signal import find_peaks
D=sys.argv[1]
for f in sorted(glob.glob(D+'/*.csv')):
  d=pd.read_csv(f,skiprows=6,encoding='utf-8-sig'); d.columns=['i','t','F','a']
  F=d.F.to_numpy()*1e3; t=d.t.to_numpy(); base=np.median(F[:200]); F=F-base
  pk,_=find_peaks(F,height=max(0.15*F.max(),300),distance=int(0.02*25000))
  out=[]
  for p in pk:
    thr=0.05*F[p]; l=p
    while l>0 and F[l]>thr: l-=1
    r=p
    while r<len(F)-1 and F[r]>thr: r+=1
    half=np.where(F[l:r+1]>0.5*F[p])[0]
    I=np.trapz(F[l:r+1],t[l:r+1])
    out.append(f"{F[p]/1e3:.2f}kN dur{(t[r]-t[l])*1e3:.2f}ms fwhm{(half[-1]-half[0]+1)*0.04:.2f}ms I{I:.1f}Ns")
  print(f"{os.path.basename(f)[:-9]:24s} n{len(F)} pk:", ' | '.join(out[:6]))
