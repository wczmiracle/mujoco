import glob, sys, pandas as pd
S = sys.argv[1]
d = pd.concat([pd.read_csv(f) for f in glob.glob(f'{S}/*_P*_S*.csv')])
d['p'] = d.job.str.split('_').str[0]
d['grp'] = d.dom_link.fillna('none').str.replace(r'_[LR]$', '', regex=True)
for tag, g in d.groupby('tag'):
  print(f'===== {tag}')
  print(g.groupby('p').agg(n=('load_ratio', 'size'), mean=('load_ratio', 'mean'), median=('load_ratio', 'median'),
        over1=('load_ratio', lambda x: (x > 1).mean()), hand_touch=('hand_touch', 'mean'), hand_first=('hand_first', 'mean'),
        t_peak=('t_peak', 'median')).round(3).to_string())
  t = g.groupby(['p', 'grp']).load_ratio.agg(['size', 'mean']).reset_index()
  t['share'] = t['size'] / t.groupby('p')['size'].transform('sum')
  print(t[t.share > 0.04].sort_values(['p', 'share'], ascending=[True, False]).round(3).to_string(index=False))
