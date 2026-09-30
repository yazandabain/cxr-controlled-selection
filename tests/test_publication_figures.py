"""Independent checks of final aggregate-only graphics and diagram assets."""
import json
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np
from cxr_selection.publication_figures import Ledger, build

ROOT=Path(__file__).resolve().parents[1]


class FinalFigureTests(unittest.TestCase):
    def test_retained_class_counts(self):
        d=json.loads((ROOT/'results/aggregates/core.json').read_text())['description']['panels']['train']['states']
        self.assertEqual(sum(x['coded_positive'] for x in d.values()),2264)
        self.assertEqual(sum(x['studies']-x['coded_positive'] for x in d.values()),35563)
        explicit=[d['explicit_negative'],d['explicit_positive']]
        self.assertEqual(sum(x['studies']-x['coded_positive'] for x in explicit),3192)
        self.assertEqual(sum(x['coded_positive'] for x in explicit),745)

    def test_rebuild_and_check_every_source_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'figures'
            build(ROOT/'results/aggregates',out)
            self.assertEqual(len(list(out.glob('*.pdf'))),7)
            self.assertEqual(len(list(out.glob('*.svg'))),7)
            record=json.loads((out/'PLOT_VALUES.json').read_text())
            self.assertGreaterEqual(record['verified_source_entries'],90)
            # Independently retrieve every value recorded by the renderer.
            for item in record['source_values'].values():
                obj=json.loads((ROOT/'results/aggregates'/item['source']).read_text())
                for key in item['path']:obj=obj[key]
                self.assertEqual(obj,item['value'])
            source=json.loads((ROOT/'results/aggregates/strength_20260913/class_source.json').read_text())
            states=['explicit_positive','explicit_negative','uncertain','blank']
            expected=[]
            for cell,coded in [('p0n1','1'),('p1n0','1'),('p1n0','0'),('p0n1','0')]:
                shares=[]
                for row in source['new_training_composition'][cell]:
                    counts=row['counts'][coded];n=sum(counts.values())
                    shares.append([counts[state]/n for state in states])
                expected.append(np.mean(shares,axis=0))
            actual=record['derived_values']['retained_fitting_composition']['value']
            np.testing.assert_allclose(actual,expected,atol=1e-14,rtol=0)
            pairs=record['derived_values']['pair_contribution_pp']['value']
            self.assertAlmostEqual(np.asarray(pairs).sum(),3.16371303874815,places=12)
            with self.assertRaises(FileExistsError):build(ROOT/'results/aggregates',out)

    def test_diagram_is_editable_and_local(self):
        path=ROOT/'figures/figure1_design.drawio'
        doc=ET.parse(path).getroot()
        self.assertEqual(doc.tag,'mxfile')
        cells=doc.findall('./diagram/mxGraphModel/root/mxCell')
        self.assertGreater(len(cells),40)
        for cell in cells:
            self.assertNotIn('link',cell.attrib)
            self.assertNotIn('image=',cell.get('style',''))
        text=' '.join(c.get('value','') for c in cells)
        for value in ['81,084','37,827','4,726','8,972','4,731','686','3,937']:
            self.assertIn(value,text)
        self.assertIn('coded class counts',text)
        svg=(ROOT/'figures/figure1_design.svg').read_text()
        self.assertNotIn('foreignObject',svg)
        self.assertNotIn('xlink:href="http',svg)


if __name__=='__main__':unittest.main()
