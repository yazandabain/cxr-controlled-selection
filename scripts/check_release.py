"""Fail on unexpected files or obvious sensitive content, including Git history.

This narrow allowlist supplements human review. It does not prove data compliance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path


FILES = {
    'requirements-ci.txt',
    'THIRD_PARTY.md', 'third_party/DEJAVU_LICENSE.txt',
    '.gitattributes', '.gitignore', 'README.md', 'NOTICE.md', 'CITATION.cff', 'pyproject.toml',
    'docs/DATA_ACCESS.md', 'docs/METHODS.md', 'examples/synthetic_study.py',
    'scripts/check_release.py',
    'src/cxr_selection/__init__.py', 'src/cxr_selection/models.py',
    'src/cxr_selection/evaluate.py', 'src/cxr_selection/addback.py',
    'src/cxr_selection/class_source_control.py', 'src/cxr_selection/pair_accounting.py',
    'src/cxr_selection/budget_comparison.py', 'src/cxr_selection/group_contrast.py',
    'src/cxr_selection/refit_sensitivity.py', 'src/cxr_selection/target_transfer.py',
    'tests/test_models.py', 'tests/test_evaluate.py', 'tests/test_addback.py',
    'tests/test_class_source_control.py', 'tests/test_pair_accounting.py',
    'tests/test_budget_comparison.py', 'tests/test_group_contrast.py',
    'tests/test_companion.py', 'tests/test_release.py',
    'docs/REPRODUCIBILITY.md', '.github/workflows/ci.yml',
    'src/cxr_selection/publication_figures.py', 'src/cxr_selection/aggregate_tables.py',
    'src/cxr_selection/control_tables.py', 'scripts/build_assets.py',
    'tests/test_publication_figures.py', 'tests/test_assets.py',
    'results/aggregates/core.json', 'results/aggregates/group_contrast.json',
    'results/aggregates/strength_20260913/addback.json',
    'results/aggregates/strength_20260913/class_source.json',
    'results/aggregates/strength_20260913/budget_comparison.json',
    'results/aggregates/strength_20260913/pair_accounting.json',
}
FIGURES = ('figure1_design', 'figure2_selection', 'figure3_learning_curves',
           'figure4_controlled_contrasts', 'figure5_era', 'figure7_selection_controls',
           'figure8_pair_accounting', 'figureS1_reference_transfer')
TABLES = ('table1_panels', 'table2_main', 'table3_contrasts', 'tableS1_characteristics',
          'tableS2_ap', 'tableS3_reference_transfer', 'tableS4_populations',
          'tableS5_refit', 'tableS6_complete_grid', 'table_strength_addback',
          'table_strength_addback_contrasts', 'table_strength_class_cells',
          'table_strength_class_contrasts', 'table_strength_budgets', 'table_strength_pairs')
FILES.update(f'figures/{name}.{suffix}' for name in FIGURES for suffix in ('svg', 'pdf'))
FILES.add('figures/figure1_design.drawio')
FILES.update(f'results/tables/{name}.{suffix}' for name in TABLES for suffix in ('csv', 'md'))
# These are rendered aggregate heatmap tiles, not radiographs or clinical images.
HEATMAP_IMAGES = {'figure7_selection_controls': 1, 'figure8_pair_accounting': 2,
                  'figureS1_reference_transfer': 2}
PROHIBITED_KEYS = {'subject_id', 'study_id', 'stay_id', 'hadm_id', 'dicom_id', 'image_id',
                   'patient_ids', 'study_ids', 'predictions', 'coefficients', 'intercept_',
                   'components_', 'train_indices', 'memberships', 'embeddings', 'features',
                   'bootstrap_arrays', 'model_state', 'model_bytes'}
PATTERNS = {
    'credential': r'(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|AKIA[A-Z0-9]{16}|sk-[A-Za-z0-9]{20,})',
    'private key': r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
    'personal local path': r'(?i)(?:[a-z]:[\\/]+Users[\\/]+|[a-z]:[\\/]+MultiModalFinal|[a-z]:[\\/]+CXRControlledSelectionPrivate)',
    'personal email': r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}',
}


def issues_in_text(text, label):
    return [f'{label}: {kind}' for kind, pattern in PATTERNS.items() if re.search(pattern, text)]


def git(root, *args, binary=False):
    result = subprocess.run(['git', '-c', f'safe.directory={root.as_posix()}', '-C', str(root), *args],
                            capture_output=True, check=True)
    return result.stdout if binary else result.stdout.decode('utf-8')


def inspect_git_metadata(raw, label):
    """Allow no-reply identity fields, not arbitrary text in annotations."""
    if b'\0' in raw or len(raw) > 1_000_000:
        return [f'{label}: unexpected metadata encoding/size']
    try:
        text = raw.decode('utf-8')
    except UnicodeDecodeError:
        return [f'{label}: not UTF-8 metadata']
    header, separator, message = text.partition('\n\n')
    if not separator:
        return [f'{label}: missing metadata/message separator']
    header = re.sub(
        r'(?m)^(author|committer|tagger) (.+) <[A-Za-z0-9+._-]+@users\.noreply\.github\.com> ',
        r'\1 \2 <GitHub no-reply> ', header,
    )
    return issues_in_text(header + separator + message, label)


def check_tags(root):
    """Inspect reachable annotated tag objects, including nested annotations."""
    issues, annotations, seen = [], [], set()
    refs = git(root, 'for-each-ref', '--format=%(refname)\t%(objecttype)\t%(objectname)')
    for line in refs.splitlines():
        reference, kind, oid = line.split('\t')
        issues.extend(issues_in_text(reference, 'Git ref name'))
        if reference.startswith('refs/tags/') and kind not in {'commit', 'tag'}:
            issues.append(f'Git ref {reference}: tag target is not a commit or annotation')
        while kind == 'tag' and oid not in seen:
            seen.add(oid)
            label = f'Git tag object {oid[:8]}'
            raw = git(root, 'cat-file', 'tag', oid, binary=True)
            issues.extend(inspect_git_metadata(raw, label))
            try:
                header = raw.decode('utf-8').partition('\n\n')[0]
            except UnicodeDecodeError:
                break
            fields = {}
            for entry in header.splitlines():
                key, separator, value = entry.partition(' ')
                if not separator or key in fields:
                    issues.append(f'{label}: malformed/duplicate tag header')
                fields[key] = value
            target = fields.get('object', '')
            target_type = fields.get('type', '')
            if not re.fullmatch(r'(?:[0-9a-f]{40}|[0-9a-f]{64})', target):
                issues.append(f'{label}: invalid target object ID')
                break
            if not fields.get('tag') or not re.fullmatch(
                r'.+ <[^<>\r\n]+> [0-9]+ [+-][0-9]{4}', fields.get('tagger', ''),
            ):
                issues.append(f'{label}: missing/invalid tag name or tagger metadata')
            actual_type = git(root, 'cat-file', '-t', target).strip()
            annotations.append({'object': oid, 'target': target, 'target_type': actual_type})
            if actual_type != target_type or actual_type not in {'commit', 'tag'}:
                issues.append(f'{label}: invalid or non-commit target type')
                break
            oid, kind = target, actual_type
    return {'annotated_tags': annotations, 'issues': issues}


def inspect_aggregate(value, trail=''):
    issues = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key.lower() in PROHIBITED_KEYS or key.endswith('_sha256') or key == 'hashes':
                issues.append(f'{trail}/{key}: restricted/provenance payload key')
            issues.extend(inspect_aggregate(item, trail + '/' + key))
    elif isinstance(value, list):
        if len(value) > 20:
            issues.append(f'{trail}: array exceeds reviewed aggregate dimensions')
        for i, item in enumerate(value):
            issues.extend(inspect_aggregate(item, trail + '/' + str(i)))
    return issues


def inspect_bytes(raw, label, file_path=None):
    suffix = Path(file_path or label).suffix.lower()
    stem = Path(file_path or label).stem
    if suffix == '.pdf':
        import fitz
        issues = []
        with fitz.open(stream=raw, filetype='pdf') as doc:
            if len(doc) != 1 or doc.embfile_count() or doc.is_encrypted:
                issues.append(f'{label}: unexpected PDF pages/attachments/encryption')
            text = str(doc.metadata)
            images = 0
            for page in doc:
                text += page.get_text()
                images += len(page.get_images(full=True))
                if list(page.widgets()) or page.get_links():
                    issues.append(f'{label}: PDF form/link content')
            if images != HEATMAP_IMAGES.get(stem, 0):
                issues.append(f'{label}: unexpected PDF image count')
            for xref in range(1, doc.xref_length()):
                obj = doc.xref_object(xref)
                if re.search(r'/(JavaScript|JS|Launch|EmbeddedFile)\b', obj):
                    issues.append(f'{label}: active/embedded PDF object')
            issues.extend(issues_in_text(text, label))
        return issues
    if b'\0' in raw:
        return [f'{label}: binary content']
    try:
        text = raw.decode('utf-8')
    except UnicodeDecodeError:
        return [f'{label}: not UTF-8 text']
    if len(raw) > 1_000_000:
        return [f'{label}: exceeds the reviewed text-file size boundary']
    issues = issues_in_text(text, label)
    if suffix == '.json':
        issues.extend(f'{label}: {issue}' for issue in inspect_aggregate(json.loads(text)))
    if suffix in {'.svg', '.drawio'}:
        standard_svg_doctype = r'<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN"\s+"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">'
        xml_text = re.sub(standard_svg_doctype, '', text) if suffix == '.svg' else text
        if '<!DOCTYPE' in xml_text or '<!ENTITY' in xml_text:
            issues.append(f'{label}: external XML declaration')
        document = ET.fromstring(text)
        image_count = 0
        for node in document.iter():
            tag = node.tag.rsplit('}', 1)[-1].lower()
            if tag == 'image':
                image_count += 1
            if tag in {'script', 'foreignobject'} or (tag == 'image' and stem not in HEATMAP_IMAGES):
                issues.append(f'{label}: active/embedded SVG content')
            for key, value in node.attrib.items():
                attr = key.rsplit('}', 1)[-1].lower()
                embedded_heatmap = tag == 'image' and stem in HEATMAP_IMAGES and value.startswith('data:image/png;base64,')
                if attr in {'href', 'link', 'src'} and not value.startswith('#') and not embedded_heatmap:
                    issues.append(f'{label}: external graphic link')
                if attr == 'style' and re.search(r'image=|javascript:|https?://', value, re.I):
                    issues.append(f'{label}: graphic image/external content')
        if suffix == '.drawio' and (document.tag != 'mxfile' or not document.findall('./diagram/mxGraphModel/root/mxCell')):
            issues.append(f'{label}: expected uncompressed editable draw.io source')
        if image_count != HEATMAP_IMAGES.get(stem, 0):
            issues.append(f'{label}: unexpected SVG image count')
    return issues


def check(root, check_git=True):
    root = Path(root).resolve()
    issues, inventory = [], []
    paths = []
    for folder, directories, filenames in os.walk(root, followlinks=False):
        for name in list(directories):
            child = Path(folder) / name
            if child.is_symlink():
                issues.append(f'{child.relative_to(root)}: symlink not allowed')
                directories.remove(name)
            elif name in {'.git', '.venv', '__pycache__'} or name.endswith('.egg-info') or (Path(folder) == root and name in {'build', 'dist'}):
                directories.remove(name)
        paths.extend(Path(folder) / name for name in filenames)
    for path in sorted(paths):
        relative = path.relative_to(root).as_posix()
        if '.git' in path.relative_to(root).parts:
            continue
        if path.is_symlink():
            issues.append(f'{relative}: symlink not allowed')
            continue
        if path.is_dir():
            continue
        if '__pycache__' in path.relative_to(root).parts and path.suffix == '.pyc':
            continue
        if relative not in FILES:
            issues.append(f'{relative}: outside reviewed inventory')
            continue
        raw = path.read_bytes()
        issues.extend(inspect_bytes(raw, relative))
        inventory.append({'path': relative, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()})
    missing = FILES - {item['path'] for item in inventory}
    issues.extend(f'{name}: missing reviewed file' for name in sorted(missing))
    historical_blobs, commits, annotated_tags = 0, 0, []
    if check_git:
        if not (root / '.git').is_dir():
            issues.append('No local Git repository; history not checked')
        else:
            tag_review = check_tags(root)
            annotated_tags = tag_review['annotated_tags']
            issues.extend(tag_review['issues'])
            seen = set()
            for commit in git(root, 'rev-list', '--all').splitlines():
                commits += 1
                issues.extend(inspect_git_metadata(
                    git(root, 'cat-file', 'commit', commit, binary=True),
                    f'Git commit metadata {commit[:8]}',
                ))
                for line in git(root, 'ls-tree', '-r', commit).splitlines():
                    fields, relative = line.split('\t', 1)
                    mode, kind, oid = fields.split()
                    if relative not in FILES or mode != '100644' or kind != 'blob':
                        issues.append(f'Git {commit[:8]}: unexpected path/mode {relative}')
                    if oid not in seen:
                        issues.extend(inspect_bytes(git(root, 'cat-file', 'blob', oid, binary=True), f'Git blob {oid[:8]}', relative))
                        seen.add(oid)
            historical_blobs = len(seen)
            staged = git(root, 'ls-files', '--stage').splitlines()
            for line in staged:
                fields, relative = line.split('\t', 1)
                mode, oid, stage = fields.split()
                if relative not in FILES or mode != '100644' or stage != '0':
                    issues.append(f'Git index: unexpected path/mode {relative}')
                issues.extend(inspect_bytes(git(root, 'cat-file', 'blob', oid, binary=True), f'Git index {relative}', relative))
    return {'files': inventory, 'commits': commits, 'historical_blobs': historical_blobs,
            'annotated_tags': annotated_tags, 'issues': issues}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--no-git', action='store_true')
    args = parser.parse_args()
    result = check(args.root, check_git=not args.no_git)
    print(json.dumps(result, indent=2))
    raise SystemExit(bool(result['issues']))
