"""Self-contained scientific PGFPlots/TikZ figures for the existing manuscript.

Full precision numerical fields remain in NPZ files. Finer-grid heatmaps use
explicit block averages for display only; quantitative errors use full arrays.
"""
import numpy as np
import pandas as pd
from kernel import *

def block64(u):
    n=u.shape[0]
    if n==64:return u
    assert n%64==0
    b=n//64
    return u.reshape(64,b,64,b).mean(axis=(1,3))

def heatmap(u,title,limit=1.25):
    u=block64(u);c=np.clip(np.rint(32*(1+u/limit)),0,64).astype(int)
    title=title.replace(', $t=',r'\\$t=')
    lines=[r'\begin{tikzpicture}[x=.058cm,y=.058cm]',r'\node[font=\footnotesize,align=center] at (32,72) {'+title+'};']
    # Merge adjacent cells, then batch paths by color to keep portable TeX
    # memory use low. Every displayed sample retains its quantized value.
    paths={j:[] for j in range(65)}
    for x in range(64):
        y=0
        while y<64:
            end=y+1
            while end<64 and c[x,end]==c[x,y]:end+=1
            paths[int(c[x,y])].append(fr'\pgfpathrectangle{{\pgfpointxy{{{x}}}{{{y}}}}}{{\pgfpointxy{{1.01}}{{{end-y+.01}}}}}')
            y=end
    for color,rectangles in paths.items():
        if rectangles:lines.append(fr'\pgfsetfillcolor{{valc{color}}}'+'\n'+'\n'.join(rectangles)+r'\pgfusepath{fill}')
    lines += [r'\draw (0,0) rectangle (64,64);',r'\node[font=\scriptsize] at (0,-4) {$0$};',r'\node[font=\scriptsize] at (64,-4) {$2\pi$};',r'\node[font=\scriptsize] at (32,-5) {$x$};',r'\node[font=\scriptsize] at (-5,32) {$y$};',r'\node[font=\scriptsize] at (-4,64) {$2\pi$};',r'\end{tikzpicture}']
    return '\n'.join(lines)

def coords(x,y):return ' '.join(f'({a:.9g},{b:.9g})' for a,b in zip(x,y) if np.isfinite(a) and np.isfinite(b) and b>0)
def colorbar():
    lines=[r'\par\smallskip\begin{tikzpicture}[x=.10cm,y=.10cm]']
    for j in range(65):lines.append(fr'\fill[valc{j}] ({j},0) rectangle ({j+1.01},2);')
    lines += [r'\draw (0,0) rectangle (65,2);',r'\node[font=\scriptsize] at (0,-2) {$-1.25$};',r'\node[font=\scriptsize] at (32.5,-2) {$0$};',r'\node[font=\scriptsize] at (65,-2) {$1.25$};',r'\node[font=\footnotesize] at (71,1) {$\phi$};',r'\end{tikzpicture}']
    return '\n'.join(lines)
def plot(series,xlabel,ylabel,title='',logx=False,logy=True,width='.46\\textwidth',legend='legend pos=north west',extra=''):
    opts=[f'width={width}', 'height=5.4cm', 'grid=major','grid style={gray!15}','tick label style={font=\\footnotesize}',
        'label style={font=\\small}',f'xlabel={{{xlabel}}}',f'ylabel={{{ylabel}}}',f'title={{{title}}}',
        'title style={font=\\small}','legend style={font=\\scriptsize,draw=none}',legend]
    if extra:opts.append(extra)
    if logx:opts+=['xmode=log']
    if logy:opts+=['ymode=log']
    lines=[r'\begin{tikzpicture}',r'\begin{axis}['+','.join(opts)+']']
    colors=['blue!65!black','orange!85!black','teal!80!black','purple','black','red!70!black']
    for i,(label,x,y) in enumerate(series):
        lines += [r'\addplot[thick,mark=*,mark size=1.5pt,'+colors[i%len(colors)]+'] coordinates {'+coords(x,y)+'};',r'\addlegendentry{'+label+'}']
    lines += [r'\end{axis}',r'\end{tikzpicture}']
    return '\n'.join(lines)

def figure(body,caption,label):
    palette=''
    if 'pgfsetfillcolor{valc' in body:
        for j in range(65):
            low=np.array([.13,.29,.62]);mid=np.array([.98,.98,.97]);high=np.array([.7,.12,.18])
            rgb=low+(mid-low)*(j/32) if j<=32 else mid+(high-mid)*((j-32)/32)
            palette+=fr'\definecolor{{valc{j}}}{{rgb}}{{{rgb[0]:.4f},{rgb[1]:.4f},{rgb[2]:.4f}}}'+'\n'
        body+='\n'+colorbar()
    return '\n'+r'\begin{figure}[htbp]\centering'+'\n'+palette+body+'\n'+r'\rev{\caption{'+caption+r'}\label{'+label+r'}}'+'\n'+r'\end{figure}'+'\n'

def build():
    figures={}
    p=DATA/'legacy_rollout_000_700.npz'
    if p.exists():
        z=np.load(p);body=''
        for step in [350,700]:
            body+='\n'+r'\par\medskip'+'\n'
            for method,label in [('ETD_half','Reference'),('C6','Six substeps'),('residual','Residual + safeguard')]:
                body+=heatmap(z[f'{method}_{step}'],f'{label}, $t={step*DT:g}$')+'\n'+r'\hspace{.3cm}'+'\n'
        figures['long']=figure(body,r'Original main-test index zero at the two long horizons on the legacy $64^2$ collocation grid. The reference is independently refined ETDRK4 on that same grid. All samples are shown; the common blue--white--red range is $[-1.25,1.25]$. The corresponding matched errors are reported separately. These fields do not constitute a spatially converged morphology claim.', 'fig:newlong')
    p=DATA/'galerkin_rollout_N256_rectangles_700.npz'
    if p.exists():
        z=np.load(p);body=''
        for row in [[('initial','Initial'),('ETD_700','Reference, $t=0.07$')],[('C6_700','Six substeps, $t=0.07$'),('refitted_700','Refitted residual, $t=0.07$')]]:
            body+='\n'+r'\par\medskip'+'\n'
            for key,label in row:body+=heatmap(z[key],label)+'\n'+r'\hspace{.5cm}'+'\n'
        figures['rectangles']=figure(body,r'Smooth periodic two-rectangle test for the dealiased $256^2$ formulation. The newly fitted filter is used, and the reference is same-grid ETDRK4. For display, each cell is the average of a $4\times4$ block; computations, errors and saved fields retain all $256^2$ samples. All panels use $[-1.25,1.25]$. This deterministic geometry lies outside the small-amplitude training regime.', 'fig:rectangles')
    p=DATA/'convergence_comparisons.csv'
    if p.exists():
        f=pd.read_csv(p);series=[]
        for case in ['random0','rectangles']:
            for t in [.0035,.07]:
                a=f[(f.comparison=='spatial')&(f.case==case)&(f.fine_n==512)&np.isclose(f.time,t)].sort_values('n')
                series.append((('Random' if case=='random0' else 'Rectangles')+f', $t={t:g}$',a.n,a.field_error))
        figures['convergence']=figure(plot(series,'$N$','Relative field difference',logx=True,width='.78\\textwidth',legend='legend pos=south west',extra='xtick={64,128,256},xticklabels={64,128,256}'),r'Spatial differences against the $512^2$ dealiased solution after Fourier interpolation of each coarser field. Temporal refinement is assessed separately in the accompanying text and supplementary data. The three times and additional random fields are retained in the supplementary data; the plot selects the earliest and latest times for readability.', 'fig:convergence')
    p=DATA/'stress_rollout.csv'
    if p.exists():
        f=pd.read_csv(p);sel=f.iloc[::5]
        body=plot([(name,sel.time,sel['error_'+key]) for key,name in [('safe','Safeguarded'),('unsafe','Unconstrained'),('C6','Six substeps')]],'$t$','Relative field error',title='(a) Stress trajectory',legend='legend style={at={(0.97,0.50)},anchor=east}')
        body+=r'\hfill'+'\n'+plot([(r'Accepted $\alpha$',sel.time,sel.alpha)],'$t$',r'$\alpha$',title='(b) Accepted fraction')
        figures['stress']=figure(body,r'Activating diagnostic trajectory selected from the separately labeled 12-case exploratory small-step extension: mean 1, standard deviation 0.08, $\epsilon=0.02$ and macro step $10^{-7}$, with the filter trained at $10^{-4}$. Errors use a time-refined reference on the same spatial discretization. All 350 accepted fractions are $2^{-26}$, nearly freezing the safeguarded field. Full energy, mass, error and acceptance records accompany the figure.', 'fig:stress')
    p=DATA/'cost_summary.csv'
    if p.exists():
        f=pd.read_csv(p).set_index('method');series=[]
        for label,names in [('Coarse',[f'C{q}' for q in [1,6,12,24,100]]),('ETDRK4',[f'ETD_h{h:g}' for h in [4e-4,2e-4,1e-4]]),
            ('Adaptive coarse',[f'adaptive_semi_tol{t:g}' for t in [1e-2,1e-3,1e-4]]),('Adaptive ETD',[f'adaptive_etd_tol{t:g}' for t in [1e-2,1e-3,1e-4]]),
            ('Residual',['residual']),('piDMD',['piDMD'])]:
            a=f.loc[names];series.append((label,a.seconds_mean,a.l2_mean))
        figures['cost']=figure(plot(series,'Mean wall time (s)','Mean relative field error',logx=True,width='.78\\textwidth',legend='legend pos=south east'),r'Achieved field error versus measured online wall time in one common environment on five initial states. Times average five repetitions per state; dispersion is reported in the table and raw data. All methods use the same legacy spatial operator, with a refined ETDRK4 reference. The residual includes its intrinsic correction and acceptance checks. Fixed ETDRK4 coefficients are warmed before timing.', 'fig:cost')
    for key,value in figures.items():(HERE/f'figure_{key}.tex').write_text(value,encoding='utf-8')
    print('Generated figures:',', '.join(figures))

if __name__=='__main__':build()
