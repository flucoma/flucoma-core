
# Alpha-01:
date: December 5th, 2019

## New Features:
- Breaking change: all dataset interfaces are streamlined, please check the help-files if you have coded something personally. This is likely to change again as we tweak the workflow, so feedback welcome.

## New Examples:
- All plenary example updated
- All PA's naive learning patches included
(still underdocumented)
- One-liner references in SC, imported as 'ugly temp ref' in Max for the interpolation and hybridisation objects (soon to be made clearer and less ugly)
- SC: simple examples ported

## Bug Fixes:
- most, but those below still stand ;-)

## Known Bugs:
- bufnmfcross sound quality on large files could be less smeary
- bufnmfcross: selecting boundaries of both input buffers is not implemented yet
- SC workflow is slow because of many language-server sync (more streamlined workflow coming)
- audiotransport is still quantised by bin (phase interpolation is not trivial)
- audiotransport windows are restricted to fft sizes for now
- MAX: (buf)audiotransport and bufnmfcross should be their own standalone object soon
- MAX: bufnmfcross: progress bar does not work in blocking 0
- instacrash with fluid.kmeans [fit validdataset wordinsteadofnumberofclusters]
- instacrash with fluid.dataset [getPoint] on an empty dataset
- Max: a few ref missing
