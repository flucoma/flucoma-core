
# Alpha-01:
date: December 2nd, 2019

## New Features:
- Breaking change: all dataset interfaces are streamlined, please check the help-files if you have coded something personally. This is likely to change again as we tweak the workflow, so feedback welcome.

## New Examples:
- All plenary example updated
- All PA's naive learning patches included
(still underdocumented)

## Bug Fixes:
- most?

## Known Bugs:
- nmfcross sound quality on large files could be less smeary
- SC workflow is slow because of many language-server sync (more streamlined workflow coming)
- transport is still quantised by bin (phase interpolation is not trivial)
- (buf)audiotransport and bufnmfcross should be their own standalone object
- bufnmfcross: progress bar does not work in blocking 0
- bufnmfcross: selecting boundaries of both input buffers
- fluid.normalize will give 0 if 2 items are the same
- instacrash with fluid.kmeans [fit validdataset wordinsteadofnumberofclusters]
- instacrash with fluid.dataset [getPoint] on an empty dataset
