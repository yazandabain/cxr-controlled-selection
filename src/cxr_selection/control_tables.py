"""Exploratory control tables from public aggregates only."""
import json
import numpy as np
from .aggregate_tables import table
STATES = ('Explicit +', 'Explicit -', 'Uncertain', 'Blank')


def longtable(path, caption, label, columns, header, rows):
    text = [r'{\small', r'\begin{longtable}{'+columns+'}', r'\caption{'+caption+r'}\label{'+label+r'}\\',
            r'\toprule '+header+r'\\\midrule\endfirsthead', r'\toprule '+header+r'\\\midrule\endhead',
            r'\midrule\multicolumn{'+str(header.count('&')+1)+r'}{r}{Continued on next page}\\\endfoot',
            r'\bottomrule\endlastfoot', *[row+r'\\' for row in rows], r'\end{longtable}', '}']
    table(path.parent, path.stem, [item.strip() for item in header.split('&')],
          [[item.strip() for item in row.split('&')] for row in rows])
    path.write_text('\n'.join(text)+'\n', encoding='utf-8')

def interval(value, key='ci95'):
    lo, hi = value[key]
    def f(x):
        if 0 < abs(x) < .00005: return f'{x:.6f}'
        return f'{x:.4f}'
    return '$'+f(value['point'])+r'\;['+f(lo)+', '+f(hi)+']$'

def build_controls(add, cross, pair, budget, out):
    w = np.asarray(pair['cell_weights'])
    arms = ['explicit','explicit_uncertain','explicit_blank','prevalence_matched']
    names = ['Explicit only','Explicit + uncertain','Explicit + blank','All states, matched']
    row_records=[]
    for data, filename, cap, prefix, labels in (
        (add,'table_strength_addback','All report-state pools, R2 logistic at N=3,072. Values are mean per-fit metrics with pointwise fixed-model patient 95\\% intervals. Gain is fusion minus image.', 'addback', dict(zip(arms,names))),
        (cross,'table_strength_class_cells','All crossed coded-class cells, R2 logistic at N=3,072. Sources are listed as coded-positive / coded-negative eligibility; the explicit and broad pools may overlap. Values include pointwise fixed-model patient 95\\% intervals.', 'classcells',{'p0n0':'Explicit / explicit','p1n0':'Broad / explicit','p0n1':'Explicit / broad','p1n1':'Broad / broad'})):
        rows=[]
        for arm,label in labels.items():
            for mod in ('image','fusion','gain'):
                v=data['groups'][arm+'__'+mod]; rows.append(label+' & '+mod.capitalize()+' & '+interval(v['auroc'])+' & '+interval(v['average_precision']))
                row_records.append({'table':filename,'group':arm+'__'+mod,'values':v})
        longtable(out/f'{filename}.tex',cap,'tab:s'+prefix,'p{.29\\textwidth}p{.09\\textwidth}p{.25\\textwidth}p{.25\\textwidth}', 'Training source & Head & AUROC [95\\% CI] & AP [95\\% CI]',rows)

    for data, filename, cap, label, names in (
        (add,'table_strength_addback_contrasts','Every specified add-back contrast. All intervals are exploratory, pointwise and conditional on fitted models.', 'saddcontrasts',
         {'prevalence_matched_minus_explicit':'All minus E','explicit_uncertain_minus_explicit':'E+U minus E','explicit_blank_minus_explicit':'E+B minus E','explicit_uncertain_minus_explicit_blank':'E+U minus E+B'}),
        (cross,'table_strength_class_contrasts','Every crossed-class simple effect, interaction and diagonal total. All intervals are exploratory and pointwise. P0/N0 use explicit-source members; P1/N1 use broad-source members. Small nonzero limits are shown with additional precision.', 'sclasscontrasts',
         {'positive_at_n0':'P1N0 minus P0N0','positive_at_n1':'P1N1 minus P0N1','negative_at_p0':'P0N1 minus P0N0','negative_at_p1':'P1N1 minus P1N0','interaction':'Interaction','total':'P1N1 minus P0N0'})):
        rows=[]
        for key,title in names.items():
            for mod in ('fusion','image','gain'):
                v=data['contrasts'][key+'__'+mod]; rows.append(title+' & '+mod.capitalize()+' & '+interval(v['auroc'])+' & '+interval(v['average_precision']))
                row_records.append({'table':filename,'contrast':key+'__'+mod,'values':v})
        longtable(out/f'{filename}.tex',cap,'tab:'+label,'p{.26\\textwidth}p{.09\\textwidth}p{.27\\textwidth}p{.26\\textwidth}', 'Contrast & Head & $\\Delta$AUROC [95\\% CI] & $\\Delta$AP [95\\% CI]',rows)
    rows=[]
    for rep,label in [('score14','R1'),('chex128','R2')]:
        for n in (512,1536,3072):
            positive=budget['groups'][rep+'_'+str(n)]['training_positive_range']
            for metric,mlabel in [('auroc','AUROC'),('average_precision','AP')]:
                v=budget['contrasts'][rep+'_'+str(n)+'_'+metric]
                rows.append(f'{label} & {n:,} & {positive[0]}--{positive[1]} & {mlabel} & '+interval(v,'pointwise_ci95')+' & '+interval(v,'joint_band95'))
                row_records.append({'table':'table_strength_budgets','contrast':rep+'_'+str(n)+'_'+metric,'values':v})
    longtable(out/'table_strength_budgets.tex','Class-matched fusion minus the full explicit fit (3,937 patients, 745 coded positives). All twelve comparisons are shown. The last column uses a joint approximate 95\\% band over the complete family. Budgets omit common preprocessing and tuning information.', 'tab:sbudgets', 'llrlp{.24\\textwidth}p{.24\\textwidth}', 'Rep. & N & Positives & Metric & Difference [pointwise CI] & Difference [joint band]',rows)
    rows=[]
    for i, ps in enumerate(STATES):
        for j, ns in enumerate(STATES):
            values=[pair['contrasts'][mod]['contributions'][i][j] for mod in ('fusion','image','gain')]
            rows.append(f'{ps} & {ns} & {100*w[i,j]:.2f} & '+ ' & '.join(f'${v:.6f}$' for v in values))
            row_records.append({'table':'table_strength_pairs','positive_state':ps,'negative_state':ns,'weight':float(w[i,j]),'contrasts':values})
    longtable(out/'table_strength_pairs.tex','All sixteen report-state pair contributions. The weight is the share of all coded-positive/coded-negative comparisons. Contributions are in AUROC units and sum to the pooled contrast. Complete pointwise cell intervals are retained in the aggregate JSON; no selected-cell significance claim is made.', 'tab:spairs','llrrrr','Positive state & Negative state & Weight (\\%) & $\\Delta_F$ & $\\Delta_I$ & $\\Delta_G$',rows)
    (out/'table_values.json').write_text(json.dumps(row_records,indent=2)+'\n',encoding='utf-8')
