FIGURES

Figure_1.tex through Figure_8.tex are the final editable figure sources.
Each source includes its plot data, captions and panel keys. The supplied
PDFs retain vector paths; PNGs are rendered at 300 dpi.

From this folder, compile a source with pdfLaTeX twice, for example:
  pdflatex -interaction=nonstopmode -halt-on-error Figure_5.tex
  pdflatex -interaction=nonstopmode -halt-on-error Figure_5.tex

Required TeX packages include amsmath, amssymb, bm, booktabs, array, xcolor,
graphicx, tikz, pgfplots and url. Compile in a separate directory to preserve
the delivered cropped PDFs. These source documents retain the article
layout used for their exports. Captions use the manuscript numbering.

To regenerate the five validation figure blocks from full numerical fields:
  python numerics/validation/figures.py
Run that command from the complete archive root. The resulting blocks are
written beside the plotting script; the eight standalone sources here are
ready-to-compile figure documents. The full data are in the release ZIP.
