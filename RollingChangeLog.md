# Alpha-04:
date:

## New Objects:

## New Features:

## New Examples:

## Bug Fixes:

## Known Bugs Still Unfixed:


====

# Alpha-03:
date: 31 Jan 2019

## New Features:
- (Max) init folder to create key shortcuts (shift-F to create a fluid. object)
- (Max) bufview now has bipolar mode (and a helpfile)
- transientslice and buftransientslice : now a minSlice argument to define the smallest slice available.

## New Examples:
- bufcompose macros and folder iterations
- updating dict (mode 1) example in the helpfile
- commented nmf in 'real time'
- commented impact of FFT size on nmf

## Bug Fixes:
- Max’s buffer classes small bug which caused crash in specific instantiation order
- Max’s nmfmatch~ is now behaving with filter buffer changes
- (Max) bufview works for small buffers
- bufnmf is now returning updated filter buffers in mode 1
- transient and buftransient - quality improved significantly with debounce fix

===

# Alpha-02a:
date: November 17th, 2018

## New Examples:
- new didactic example in bufnmf and nmfmatch

## Bug Fixes:
- NMFMatch bug fix - cleaner activations and more stable behaviour

## Known Bugs Still Unfixed:
- debounce in all transient-based code

===

# Alpha-02:
date: November 5th, 2018

## New Objects:
- NMFMatch: real-time matching against fixed dictionaries as templates.
- (Max Only): new GUI for multichannel buffer viewing (in NMFMatch~ helpfile)

## New Features:
- New parameter for NoveltySlice - a smoothing filter for the novelty curve
- Breaking change for all Buf* objects (Max Only) - no more support for multibuffer~ - please use multichannel buffers

## New Examples:
- BufCompose help file: new example using MS and FIR

## Bug Fixes:
- all help files: many typos and bugs sorted
- NoveltySlice: the detection is now normalising
- BufCompose rewrite

## Known Bugs Still Unfixed:
- Transient, TransientSlice, BufTransient, BufTransientSlice - the debounce is not behaving as expected
