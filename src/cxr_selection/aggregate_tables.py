"""Core summary tables from public aggregate results only."""
import csv
import json
from pathlib import Path

COLORS = dict.fromkeys(('explicit', 'random', 'prevalence_matched'))
FAMILY = {'logistic': 'Logistic', 'xgboost': 'Boosted trees'}

def tex_escape(x):
    for old,new in [('&',r'\&'),('%',r'\%'),('_',r'\_')]: x=x.replace(old,new)
    return x

def num(x): return f'{x:.3f}'

def interval(x,ci): return f'{x:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]'

def table(out, name, headers, rows, align=None):
    with (out / (name + '.csv')).open('w', encoding='utf-8', newline='') as stream:
        csv.writer(stream).writerows([headers, *rows])
    text = '| ' + ' | '.join(headers) + ' |\n| ' + ' | '.join(['---'] * len(headers)) + ' |\n'
    text += '\n'.join('| ' + ' | '.join(map(str, row)) + ' |' for row in rows) + '\n'
    (out / (name + '.md')).write_text(text, encoding='utf-8')
    tex = r'\begin{tabular}{' + (align or 'l' + 'r' * (len(headers)-1)) + '}\n' + r'\toprule' + '\n'
    tex += ' & '.join(headers) + r' \\' + '\n' + r'\midrule' + '\n'
    tex += '\n'.join(' & '.join(map(str, row)) + r' \\' for row in rows)
    tex += '\n' + r'\bottomrule' + '\n' + r'\end{tabular}' + '\n'
    (out / (name + '.tex')).write_text(tex, encoding='utf-8')


def build_core(bundle, out):
    summaries, description = bundle['campaigns'], bundle['description']
    era, refit = bundle['era'], bundle['refit']
    transfers, populations = bundle['transfer'], bundle['populations']
    rows=[]
    for part in ('train','tune','test'):
        d=description['panels'][part]; rows.append([part.title(),f"{d['all']['studies']:,}",f"{d['all']['patients']:,}",f"{d['all']['coded_positive']:,}",f"{d['explicit']['studies']:,}",f"{d['explicit']['coded_positive']:,}"])
    table(out,'table1_panels',['Panel','Studies','Patients','Coded+','Explicit','Explicit coded+'],rows)
    rows=[]
    for key,s in summaries.items():
        rep,family=key.split('_');prefix=f"{'Scores' if rep=='score14' else 'CheXpert'} / {FAMILY[family]}"
        for arm in COLORS:
            g=s['groups'];z=s['contrasts'][arm+'__gain']
            rows.append([prefix,{'explicit':'Explicit','random':'Random','prevalence_matched':'Class-matched'}[arm],
                *[num(g['3072__'+arm+'__'+mod]['auroc']) for mod in ('image','triage','fusion')],interval(z['auroc'],z['ci95']['auroc'])])
    table(out,'table2_main',['Representation / head','Selection','Image','Triage','Fusion','Gain [95\\% CI]'],rows,'llrrrr')
    rows=[]
    for key,s in summaries.items():
        rep,family=key.split('_');f=s['contrasts']['prevalence_matched_minus_explicit__fusion'];g=s['contrasts']['prevalence_matched_minus_explicit__gain']
        rows.append([('Scores' if rep=='score14' else 'CheXpert')+' / '+FAMILY[family],interval(f['auroc'],f['ci95']['auroc']),interval(g['auroc'],g['ci95']['auroc'])])
    table(out,'table3_contrasts',['Representation / head',r'$\Delta_F$ [95\% CI]',r'$\Delta_G$ [95\% CI]'],rows,'lrr')
    rows=[]
    for key,s in summaries.items():
        rep,family=key.split('_')
        for arm in COLORS:
            g=s['groups'];z=s['contrasts'][arm+'__gain'];rows.append([('Scores' if rep=='score14' else 'CheXpert')+'/'+FAMILY[family],{'explicit':'Explicit','random':'Random','prevalence_matched':'Matched'}[arm],
                *[num(g['3072__'+arm+'__'+mod]['average_precision']) for mod in ('image','triage','fusion')],interval(z['average_precision'],z['ci95']['average_precision'])])
    table(out,'tableS2_ap',['Representation / head','Selection','Image AP','Triage AP','Fusion AP','AP gain [95\\% CI]'],[[tex_escape(str(c)) if j<2 else c for j,c in enumerate(row)] for row in rows],'llrrrr')
    rows=[]
    for rep,s in transfers.items():
        for train in ('report','coded'):
            for ref in ('report','coded'):
                z=s['results']; rows.append(['R1 scores' if rep=='score14' else 'R2 CheXpert',train,ref,*[num(z[train+'_'+mod+'__against_'+ref]['auroc']['point']) for mod in ('image','triage','fusion')],interval(z[train+'_gain__against_'+ref]['auroc']['point'],z[train+'_gain__against_'+ref]['auroc']['ci95'])])
    table(out,'tableS3_reference_transfer',['Representation','Train target','Reference','Image','Triage','Fusion','Gain [95\\% CI]'],rows,'lllrrrr')
    rows=[]
    for key,s in populations.items():
        for name,z in s['populations'].items():
            f=z['contrasts']['prevalence_matched_minus_explicit__fusion']['auroc'];g=z['contrasts']['prevalence_matched_minus_explicit__gain']['auroc']
            human_key=key.replace('score14','Scores').replace('chex128','CheXpert').replace('_logistic',' / LR').replace('_xgboost',' / trees')
            human_name={'explicit_report':'Explicit','other_report_states':'Other states','one_study_per_patient':'One per patient'}[name]
            rows.append([human_key,human_name,str(z['n']),str(z['coded_positive']),interval(f['point'],f['ci95']),interval(g['point'],g['ci95'])])
    table(out,'tableS4_populations',['Setting','Test view','N','Coded+',r'$\Delta_F$ [95\% CI]',r'$\Delta_G$ [95\% CI]'],rows,'llrrrr')
    rows=[]
    for key,z in refit['results'].items():
        if key.startswith('matched_minus_explicit'): rows.append([key.replace('matched_minus_explicit_','').replace('__auroc',' AUROC').replace('__average_precision',' AP').capitalize().replace('auroc','AUROC').replace('ap','AP'),interval(z['anchor'],z['ci95'])])
    table(out,'tableS5_refit',['Single-anchor contrast','Estimate [refitting 95\\% interval]'],rows,'lr')
    rows=[]
    for key,s in summaries.items():
        def order(item):
            parts=item[0].split('__');return (int(parts[0]) if parts[0].isdigit() else 999999,parts[1],{'image':0,'triage':1,'fusion':2}[parts[2]])
        for name,z in sorted(s['groups'].items(),key=order):
            human_key=key.replace('score14','Scores').replace('chex128','CheXpert').replace('_logistic',' / LR').replace('_xgboost',' / trees')
            human_name=name.replace('full__full_','Full / ').replace('prevalence_matched','Matched').replace('__',' / ')
            rows.append([human_key,human_name,str(z['fits']),num(z['auroc']),num(z['average_precision']),str(z['training_positive_range'][0])+'--'+str(z['training_positive_range'][1])])
    table(out,'tableS6_complete_grid',['Setting','Group','Fits','AUROC','AP','Training positives'],rows,'llrrrr')
