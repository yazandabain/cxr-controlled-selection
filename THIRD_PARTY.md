# Third-party material

No dependency source, pretrained weights, dataset files, journal templates or downloaded icon packs are vendored in this repository.

## Figure fonts

The quantitative figure PDFs embed subsets of DejaVu Sans and DejaVu Sans Bold. Their separate Bitstream Vera and Arev permissions are reproduced in [third_party/DEJAVU_LICENSE.txt](third_party/DEJAVU_LICENSE.txt), extracted from the exact installed font's license record. No standalone font file is included. These font terms are not a license for the analysis code or the study results.

The conceptual figure PDF contains Arial subsets for document display. The embedded fonts' OpenType embedding flags were checked. Arial is not supplied as a standalone or web font. The editable draw.io and SVG files reference font names without bundling font files. See [Microsoft's document-embedding guidance](https://learn.microsoft.com/en-us/typography/fonts/font-faq).

## Dependencies and build tools

NumPy, pandas, SciPy, scikit-learn, XGBoost, Matplotlib, PyMuPDF and PyYAML are installed as external dependencies and remain subject to their own licenses. In particular, PyMuPDF's AGPL/commercial terms remain separate; using it for release checks does not mean its source has been relicensed here.

draw.io was used to author and export the conceptual figure. Its application and source are not distributed. GitHub Actions are referenced by pinned commits and are not vendored.

Authored source-code licensing authority is not yet confirmed. There is no blanket repository license, including over these separate third-party components.
