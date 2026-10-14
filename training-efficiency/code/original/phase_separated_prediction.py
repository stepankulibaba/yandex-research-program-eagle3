"""Prospective independent phase fits, immutable evaluation predictions."""
import statistics
PHASES=('transfer','forward','backward','update')
def coefficients(xs,ys,quadratic):
 if not quadratic:
  a=(ys[-1]-ys[0])/(xs[-1]-xs[0]);return [ys[0]-a*xs[0],a,0.]
 x0,x1,x2=xs;y0,y1,y2=ys
 d01=(y1-y0)/(x1-x0);d12=(y2-y1)/(x2-x1)
 c=(d12-d01)/(x2-x0);b=d01-c*(x0+x1);a=y0-b*x0-c*x0*x0
 return [a,b,c]
def predict(rows,cfg):
 xs=cfg['calibration_lengths'];out={'model':cfg['model'],'anchors':xs,'coefficients':{},'forms':{}}
 for mode in ('dense','split'):
  co={}
  for phase in PHASES:
   ys=[statistics.median(r['phase_peaks_mib'][phase] for r in rows if r['mode']==mode and r['length']==n and r['kind']=='memory') for n in xs]
   co[phase]={'abc':coefficients(xs,ys,phase in ('forward','backward')),'anchor_peaks_mib':ys}
  out['coefficients'][mode]=co
  for n in cfg['evaluation_lengths']:
   p={k:c['abc'][0]+c['abc'][1]*n+c['abc'][2]*n*n for k,c in co.items()};rank=sorted(p,key=p.get,reverse=True)
   out['forms'].setdefault(str(n),{})[mode]={'phase_peaks_mib':p,'full_peak_mib':max(p.values()),'dominance':rank[0] if p[rank[0]]-p[rank[1]]>cfg['dominance_gap_mib'] else 'ambiguous'}
 for n,m in out['forms'].items():m['saved_mib']=m['dense']['full_peak_mib']-m['split']['full_peak_mib']
 return out
