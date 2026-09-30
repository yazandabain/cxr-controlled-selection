"""Final aggregate-only quantitative graphics with an auditable numeric ledger.

No patient data, predictions, fitting or resampling is used here. The conceptual
study schematic is authored separately in draw.io and is not rendered by Python.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter
import numpy as np

INK='#203849'; BLUE='#236B8E'; TEAL='#197F7A'; ORANGE='#BA642F'
GREY='#7D8994'; GRID='#D8E0E5'
ARMS=('explicit','random','prevalence_matched')
COLORS=dict(zip(ARMS,(BLUE,GREY,ORANGE)))
LABELS={'explicit':'Explicit reports','random':'All states: random','prevalence_matched':'All states: class-matched'}
STATES=('explicit_positive','explicit_negative','uncertain','blank')
STATE_NAMES=('Explicit positive','Explicit negative','Uncertain','Blank')
STATE_COLORS=(BLUE,TEAL,'#CE9345','#CDD8E0')
REPS=('score14','chex128')
REP_LABELS=('R1: historical finding scores','R2: CheXpert features')


class Ledger:
    """Every source value retrieved for rendering is checked on fresh reread."""
    def __init__(self, root):
        self.root=Path(root)
        self.docs={n:json.loads((self.root/n).read_text(encoding='utf-8')) for n in
                   ['core.json','strength_20260913/addback.json','strength_20260913/class_source.json',
                    'strength_20260913/pair_accounting.json']}
        self.values={}; self.derived={}

    def get(self, doc, *path):
        value=self.docs[doc]
        for key in path: value=value[key]
        key=doc+'::'+'/'.join(map(str,path))
        self.values[key]={'source':doc,'path':list(path),'value':value}
        return value

    def derive(self, name, values, definition):
        self.derived[name]={'definition':definition,'value':np.asarray(values).tolist()}
        return values

    def verify(self):
        reread={n:json.loads((self.root/n).read_text(encoding='utf-8')) for n in self.docs}
        for item in self.values.values():
            value=reread[item['source']]
            for key in item['path']: value=value[key]
            assert value==item['value'], item['path']
        return {'sources':{n:hashlib.sha256((self.root/n).read_bytes()).hexdigest() for n in self.docs},
                'source_values':self.values,'derived_values':self.derived,
                'verified_source_entries':len(self.values),
                'note':'Figures use only retained aggregate entries or the explicitly defined arithmetic recorded here.'}


def setup():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.6,'axes.titlesize':9.5,
        'axes.labelsize':8.6,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,
        'text.color':INK,'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK,
        'axes.edgecolor':GRID,'axes.spines.top':False,'axes.spines.right':False,
        'axes.linewidth':.7,'grid.color':GRID,'grid.linewidth':.55,
        'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
        'svg.hashsalt':'cxr-final-20260922','savefig.facecolor':'white'})


def save(fig,out,name):
    fig.savefig(out/(name+'.pdf'),bbox_inches='tight',pad_inches=.08,
                metadata={'Creator':'CXR aggregate figure builder','CreationDate':None,'ModDate':None})
    fig.savefig(out/(name+'.svg'),bbox_inches='tight',pad_inches=.08,metadata={'Date':None})
    plt.close(fig)


def title(ax,letter,text):
    ax.set_title(letter+'  '+text,loc='left',fontweight='bold',pad=12)


def interval(ax,point,limits,y,color=BLUE,marker='o',label=None):
    lo,hi=limits
    ax.errorbar(point,y,xerr=[[point-lo],[hi-point]],fmt=marker,color=color,
                markersize=4.8,elinewidth=1.25,capsize=2.2,label=label,zorder=3)


def composition(out,L):
    core='core.json'; cross='strength_20260913/class_source.json'
    states=L.get(core,'description','panels','train','states')
    positive=np.array([states[s]['coded_positive'] for s in STATES])
    negative=np.array([states[s]['studies']-states[s]['coded_positive'] for s in STATES])
    assert positive.sum()==2264 and negative.sum()==35563
    assert positive[:2].sum()==745 and negative[:2].sum()==3192
    pool=np.vstack((positive,negative))
    pool_fraction=L.derive('source_class_composition',pool/pool.sum(axis=1)[:,None],
        'Each retained source-pool report-state count divided by its coded-class total; negative count = studies - coded_positive.')
    compositions=L.get(cross,'new_training_composition')
    # The cross cells contain the exact parent class membership counts.
    fractions=[]
    for cell,code in [('p0n1','1'),('p1n0','1'),('p1n0','0'),('p0n1','0')]:
        rows=np.array([[r['counts'][code][s] for s in STATES] for r in compositions[cell]])
        assert len(rows)==20 and np.all(rows.sum(axis=1)>0)
        fractions.append((rows/rows.sum(axis=1)[:,None]).mean(axis=0))
    fractions=L.derive('retained_fitting_composition',np.array(fractions),
        'Mean of 20 within-seed coded-class fractions, in order explicit positives, broad positives, explicit negatives, broad negatives. Exact parent counts recovered from the crossed cells p0n1 and p1n0.')
    assert np.allclose(np.sum(fractions,axis=1),1,atol=1e-14)
    fig,axs=plt.subplots(2,1,figsize=(7.1,4.55),gridspec_kw={'height_ratios':[1,1.55]})
    for ax,data,labels in [(axs[0],pool_fraction,['Coded positive  (n = 2,264)','Coded negative  (n = 35,563)']),
            (axs[1],fractions,['Positive source: explicit','Positive source: broad','Negative source: explicit','Negative source: broad'])]:
        for i,row in enumerate(data):
            left=0
            for j,(v,color) in enumerate(zip(row,STATE_COLORS)):
                ax.barh(i,100*v,left=100*left,height=.62,color=color,edgecolor='white',linewidth=.9)
                if v>=.08:
                    ax.text(100*(left+v/2),i,f'{v*100:.1f}%',ha='center',va='center',fontsize=8,
                            color='white' if j<2 else INK)
                left+=v
        ax.set_xlim(0,100);ax.set_xticks([0,25,50,75,100]);ax.set_yticks(range(len(labels)),labels)
        ax.invert_yaxis();ax.spines[['left','bottom']].set_visible(False);ax.tick_params(axis='y',length=0)
        ax.xaxis.set_major_formatter(PercentFormatter(100,decimals=0))
    axs[0].set_ylim(1.6,-.6);axs[1].set_ylim(3.6,-.6)
    title(axs[0],'a','Available patients: report composition within each coded class')
    title(axs[1],'b','Retained fitting samples: class counts match, composition does not')
    axs[0].text(.0,-.37,'Explicit eligibility retains 32.9% of coded positives and 9.0% of coded negatives.',
                transform=axs[0].transAxes,fontsize=8)
    axs[1].set_xlabel('Mean share within the coded class across 20 fitting samples')
    handles=[Patch(facecolor=c,label=n) for c,n in zip(STATE_COLORS,STATE_NAMES)]
    fig.legend(handles=handles,ncols=4,frameon=False,loc='lower center',bbox_to_anchor=(.52,-.005),fontsize=8)
    fig.subplots_adjust(left=.27,right=.985,top=.93,bottom=.14,hspace=.72)
    save(fig,out,'figure2_selection')


def curves(out,L):
    fig,axs=plt.subplots(2,2,figsize=(7.1,4.75),sharex=True)
    budgets=[512,1536,3072]
    for row,rep in enumerate(REPS):
        for arm,color in COLORS.items():
            group=[L.get('core.json','campaigns',rep+'_logistic','groups',f'{n}__{arm}__fusion') for n in budgets]
            im=np.array([L.get('core.json','campaigns',rep+'_logistic','groups',f'{n}__{arm}__image','auroc') for n in budgets])
            fusion=np.array([v['auroc'] for v in group]);spread=np.array([v['subsample_range']['auroc'] for v in group])
            gain=L.derive('curves_'+rep+'_'+arm,fusion-im,'Per-budget mean fusion AUROC minus mean image AUROC.')
            axs[row,0].plot(budgets,fusion,color=color,marker='o',ms=3.3,lw=1.5,label=LABELS[arm])
            axs[row,0].plot(budgets,im,color=color,ls='--',lw=1.15)
            axs[row,0].fill_between(budgets,spread[:,0],spread[:,1],color=color,alpha=.11,lw=0)
            axs[row,1].plot(budgets,gain,color=color,marker='o',ms=3.3,lw=1.5)
        axs[row,0].set_ylabel(('R1: finding scores' if row==0 else 'R2: CheXpert features')+'\nAUROC')
        axs[row,1].set_ylabel('Fusion - image AUROC')
        axs[row,0].set_ylim(.52,.87);axs[row,1].set_ylim(-.003,.115)
        axs[row,1].axhline(0,color=GREY,ls=':',lw=.8)
        for ax in axs[row]:ax.grid(axis='y');ax.set_xticks(budgets,['512','1,536','3,072'])
    title(axs[0,0],'a','Absolute discrimination');title(axs[0,1],'b','Incremental fusion gain')
    for ax in axs[1]:ax.set_xlabel('Patients used to fit each head')
    fig.legend(*axs[0,0].get_legend_handles_labels(),ncols=3,frameon=False,loc='lower center',bbox_to_anchor=(.51,.02),fontsize=7.8)
    fig.text(.52,.009,'Solid: fusion   Dashed: image   Shading: subset min-max, not confidence intervals',ha='center',fontsize=7.6)
    fig.subplots_adjust(left=.115,right=.985,top=.925,bottom=.195,hspace=.27,wspace=.38)
    save(fig,out,'figure3_learning_curves')


def contrasts(out,L):
    keys=['score14_logistic','score14_xgboost','chex128_logistic','chex128_xgboost']
    labels=['R1 / logistic','R1 / trees','R2 / logistic (primary)','R2 / trees']
    fig,axs=plt.subplots(1,2,figsize=(7.1,2.65),sharey=True)
    for j,mod in enumerate(['fusion','gain']):
        ax=axs[j]
        for i,key in enumerate(keys):
            d=L.get('core.json','campaigns',key,'contrasts','prevalence_matched_minus_explicit__'+mod)
            interval(ax,d['auroc'],d['ci95']['auroc'],i,ORANGE if i==2 else BLUE)
        ax.axvline(0,color=GREY,ls=':',lw=1);ax.grid(axis='x');ax.set_ylim(3.55,-.55)
        ax.set_xlabel('Class-matched all-state minus explicit')
    title(axs[0],'a','Absolute fusion AUROC change')
    title(axs[1],'b','Fusion-gain change')
    axs[0].set_yticks(range(4),labels);axs[0].set_xlim(-.005,.045);axs[1].set_xlim(-.018,.007)
    fig.subplots_adjust(left=.235,right=.98,top=.83,bottom=.24,wspace=.24)
    save(fig,out,'figure4_controlled_contrasts')


def era(out,L):
    fig,axs=plt.subplots(1,2,figsize=(7.1,2.65),sharey=True)
    for i,arm in enumerate(ARMS):
        for mod,off,marker in [('image',-.12,'o'),('fusion',.12,'s')]:
            d=L.get('core.json','era','results',arm+'__'+mod,'auroc')
            interval(axs[0],d['point'],d['ci95'],i+off,COLORS[arm],marker)
        d=L.get('core.json','era','results',arm+'__gain','auroc')
        interval(axs[1],d['point'],d['ci95'],i,COLORS[arm])
    for ax in axs:ax.set_ylim(2.55,-.5);ax.grid(axis='x')
    axs[0].set_yticks(range(3),['Explicit','All states: random','All states: matched'])
    axs[0].set_xlabel('Later-era AUROC');axs[1].set_xlabel('Fusion - image AUROC')
    axs[1].axvline(0,color=GREY,ls=':',lw=.9)
    title(axs[0],'a','Absolute discrimination');title(axs[1],'b','Incremental fusion gain')
    fig.legend(handles=[Line2D([0],[0],marker='o',ls='',color=INK,label='Image'),Line2D([0],[0],marker='s',ls='',color=INK,label='Fusion')],
               ncols=2,frameon=False,loc='lower center',bbox_to_anchor=(.48,-.01))
    fig.subplots_adjust(left=.215,right=.985,top=.82,bottom=.30,wspace=.23)
    save(fig,out,'figure5_era')


def transfer(out,L):
    fig,axs=plt.subplots(1,2,figsize=(7.1,2.8))
    for ax,rep,label in zip(axs,REPS,REP_LABELS):
        data=np.array([[L.get('core.json','transfer',rep,'results',train+'_fusion__against_'+ref,'auroc','point')
                        for ref in ('report','coded')] for train in ('report','coded')])
        ax.imshow(data,cmap='Blues',vmin=.5,vmax=.9)
        for i in range(2):
            for j in range(2):ax.text(j,i,f'{data[i,j]:.3f}',ha='center',va='center',fontsize=12,
                                    color='white' if data[i,j]>.75 else INK)
        ax.set_xticks([0,1],['Report','Coded']);ax.set_yticks([0,1],['Report','Coded'])
        ax.set_xlabel('Scoring reference');ax.set_ylabel('Training target');ax.set_title(label,weight='bold')
    fig.subplots_adjust(left=.095,right=.985,top=.85,bottom=.20,wspace=.4)
    save(fig,out,'figureS1_reference_transfer')


def source_controls(out,L):
    add='strength_20260913/addback.json';cross='strength_20260913/class_source.json'
    fig=plt.figure(figsize=(7.1,4.8))
    gs=fig.add_gridspec(2,2,width_ratios=[1.05,1],height_ratios=[1.13,1])
    a=fig.add_subplot(gs[0,0]);b=fig.add_subplot(gs[0,1]);c=fig.add_subplot(gs[1,:])
    arms=['explicit','explicit_uncertain','explicit_blank','prevalence_matched']
    for mod,off,color,marker in [('image',-.12,BLUE,'o'),('fusion',.12,ORANGE,'s')]:
        for i,arm in enumerate(arms):
            d=L.get(add,'groups',arm+'__'+mod,'auroc')
            interval(a,d['point'],d['ci95'],i+off,color,marker)
    a.set_yticks(range(4),['Explicit','Explicit + uncertain','Explicit + blank','All states: matched'])
    a.set_ylim(3.55,-.6);a.set_xlim(.66,.825);a.set_xlabel('Absolute AUROC');a.grid(axis='x')
    title(a,'a','Report-state admission')
    data=np.array([[L.get(cross,'groups',f'p{i}n{j}__fusion','auroc','point') for j in range(2)] for i in range(2)])
    b.imshow(data,cmap='Blues',vmin=.70,vmax=.82,aspect='auto')
    for i in range(2):
        for j in range(2):
            im=L.get(cross,'groups',f'p{i}n{j}__image','auroc','point')
            gain=L.get(cross,'groups',f'p{i}n{j}__gain','auroc','point')
            assert abs(gain-(data[i,j]-im))<1e-14
            txtcolor='white' if data[i,j]>.76 else INK
            b.text(j,i-.22,f'{data[i,j]:.4f}',ha='center',va='center',fontsize=11,weight='bold',color=txtcolor)
            b.text(j,i+.04,f'Image {im:.4f}',ha='center',va='center',fontsize=7.8,color=txtcolor)
            b.text(j,i+.27,f'Gain {gain:.4f}',ha='center',va='center',fontsize=7.8,color=txtcolor)
    b.set_xticks([0,1],['Explicit','Broad']);b.set_yticks([0,1],['Explicit','Broad'])
    b.set_xlabel('Coded-negative source');b.set_ylabel('Coded-positive source',labelpad=4)
    title(b,'b','Crossed sources: fusion AUROC')
    effects=['positive_at_n0','positive_at_n1','negative_at_p0','negative_at_p1']
    labels=['Broaden positives | explicit negatives','Broaden positives | broad negatives',
            'Broaden negatives | explicit positives','Broaden negatives | broad positives']
    for metric,off,color,marker in [('auroc',-.12,BLUE,'o'),('average_precision',.12,ORANGE,'s')]:
        for i,key in enumerate(effects):
            d=L.get(cross,'contrasts',key+'__fusion',metric)
            interval(c,d['point'],d['ci95'],i+off,color,marker)
    c.set_yticks(range(4),labels,fontsize=7.9);c.set_ylim(3.6,-.6);c.set_xlim(-.017,.045)
    c.axvline(0,color=GREY,ls=':',lw=.9);c.grid(axis='x');c.set_xlabel('Change in absolute fusion performance')
    title(c,'c','All four class-source simple effects')
    c.legend(handles=[Line2D([0],[0],marker='o',ls='',color=BLUE,label='AUROC'),Line2D([0],[0],marker='s',ls='',color=ORANGE,label='Average precision')],
             frameon=False,ncols=2,loc='upper right',bbox_to_anchor=(1,-.34),fontsize=8)
    a.legend(handles=[Line2D([0],[0],marker='o',ls='',color=BLUE,label='Image'),Line2D([0],[0],marker='s',ls='',color=ORANGE,label='Fusion')],
             frameon=False,ncols=2,loc='upper center',bbox_to_anchor=(.5,-.30),fontsize=8)
    fig.subplots_adjust(left=.255,right=.985,top=.94,bottom=.19,hspace=.90,wspace=.60)
    save(fig,out,'figure7_selection_controls')


def pairs(out,L):
    source='strength_20260913/pair_accounting.json'
    weights=np.array(L.get(source,'cell_weights'))
    contrib=np.array(L.get(source,'contrasts','fusion','contributions'))
    assert abs(weights.sum()-1)<1e-14
    assert abs(contrib.sum()-.0316371303874815)<1e-14
    data=[L.derive('pair_mass_percent',100*weights,'Pair weights multiplied by 100.'),
          L.derive('pair_contribution_pp',100*contrib,'AUROC contributions multiplied by 100 to express percentage points.')]
    fig,axs=plt.subplots(1,2,figsize=(7.1,3.35))
    short=['Explicit +','Explicit -','Uncertain','Blank']
    for k,ax in enumerate(axs):
        v=data[k];vmax=v.max() if k==0 else abs(v).max()
        m=ax.imshow(v,cmap='Blues' if k==0 else 'RdBu_r',vmin=0 if k==0 else -vmax,vmax=vmax)
        for i in range(4):
            for j in range(4):
                cell=f'{v[i,j]:.2f}' if k==0 else f'{v[i,j]:.3f}'
                if cell=='-0.000':cell='0.000'
                ax.text(j,i,cell,ha='center',va='center',fontsize=8,
                        color='white' if abs(v[i,j])>.62*vmax else INK)
        ax.set_xticks(range(4),short,rotation=28,ha='right');ax.set_yticks(range(4),short)
        ax.set_xlabel('Report state: coded-negative study')
        title(ax,'a' if k==0 else 'b','Pair opportunity (%)' if k==0 else 'AUROC contribution (pp)')
    axs[0].set_ylabel('Report state: coded-positive study')
    fig.subplots_adjust(left=.12,right=.985,top=.87,bottom=.25,wspace=.43)
    save(fig,out,'figure8_pair_accounting')


def build(bundle,out):
    out=Path(out)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    setup();ledger=Ledger(bundle)
    for plot in (composition,curves,contrasts,era,transfer,source_controls,pairs):plot(out,ledger)
    record=ledger.verify()
    record['builder_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    record['figure_hashes']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir()) if p.is_file()}
    (out/'PLOT_VALUES.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'figures':7,'source_entries_checked':record['verified_source_entries'],'out':str(out)}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',required=True,type=Path);parser.add_argument('--out',required=True,type=Path)
    args=parser.parse_args();build(args.bundle,args.out)
