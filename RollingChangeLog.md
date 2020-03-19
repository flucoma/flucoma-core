# 1.0.0-RC1
date: 

## New Features:
* BREAKING CHANGE: (buf)melbands how has normalised amplitude output option, on by default, which is independent of window and fft size
* (buf)NoveltySlice now has a minimum slice length parameter
* BREAKING CHANGE: (buf)sines now have new parameter names, and a new option to select which algorithm is used to track the sines. It also has improved sound quality and more refined thresholding.
* BREAKING CHANGE: (buf)AmpSlice is now 2 objects, (buf)AmpGate for the absolute, and (buf)AmpSlice for the relative
* (buf)AmpSlice and (buf)AmpGate has 0Hz highpassfreq to bypass the high pass filter.

## Bug Fixes:
* (Pd on Windows) now works ;-)
* BREAKING CHANGE: buf* processes now use the buffer's sampling rate (or the SR attribute for Pd)
* BREAKING CHANGE: (buf)ampslice now works much faster and consistently but thresholds and times will have to be tweaked
* (Pd) BREAKING CHANGE: all non-real-time (buf) objects have lost the tilde (~) in their name to be consistent with the language conventions.
* BREAKING CHANGE: (buf)OnsetSlice latency reporting (rt) and offset (buf) are now more accurate but will have changed for similar threshold
* BREAKING CHANGE: (buf)NoveltySlice latency reporting (rt) and offset (buf) are now more accurate but will have changed for similar threshold
* (buf)HPSS speed is improved
* Many edge cases were found and sorted.
* (Max) parameters updates now behave all the time in real-time
* (SC) BREAKING CHANGE: bufnmf's audio output destination buffer is renamed properly to resynth

## Improvements:
* (Pd+CLI) documentation is now more accurate and consistently formatted
* cross reference between documents (see also)
* credits, references and acknowledgements are streamlined and consistent
* (Pd) better code for pinknoise abstraction

## New Example:
* (SC+Pd) most Max examples are now ported to the two other CCEs
* (SC) Gerard's GUI demos of algorithms are now available

## Known Bugs:
* bufnmf progress output in multithreading mode (blocking 0) does not output progress.
* (Mac) Notarization is not implemented yet since most CCEs are working on workarounds

===
# beta-02: multithreading!
date: 9 October 2019

## New Features:
* Max+Pd+SC: multithreading for all buf* objects! Make sure to read the tutorial on how to make the best use of various threading options you now have.
* Pd: stereo examples of fluid.buf* objects are done
* CLI: BREAKING CHANGE: new executable naming convention, because we want it to be fluid.
* CLI: CSV file output type
* Max: new audio player in some help files, adapted from C74's [demosound]
* pd (0.50.1+) launching of html ref from the helpfiles

## Bug Fixes:
* (Max) Reference pages are more complete (with all messages)
* (Max) Help patches better at making sound straight out of the box
* (CLI) Some audio file format were wrong polarity, now sorted
* (CLI) Mangled wav files on windows sorted
* all: many little tweaks here and there, more stable and consistent documentation

## New Example:
* (all) nb_of_slice is updated and more intelligent

## Known Bugs:
* BufNoveltySlice might generate garbage strange values in the first frame
* AmpSlice is noisy when some parameters are changed
* (Max/PD) Successive use of some fluid.buf* objects with buffers of differing sample rates will stick at first sample rate received

===
# beta-01: some fixes and more
date: 26 August 2019

## New Features:
- (Pd) all helpfiles now completed (except a few stereo examples of fluid.buf* objects - placeholders are empty [pd] patchers)
- (max) clickable overview

## Bug Fixes:
- (max) hpss help of maskingmode 1 and 2 sorted
- (max) bufpitch help is resizing
- strange communication between instances of bufnmf~ now zapped
- (pd) sample rate of buf* descriptor objects is now assumed (see helpfiles of bufPitch, bufMFCC, bufMelbands, bufSpectralShape)
- (pd) crash on some patches closings now should all be zapped
- (linux) strange names of Pd and Linux now sorted
- no more NaNs and crashes with edge cases and digital silence on NMF !
- HPSS does not crackle on param changes anymore
- BufOnsetSlice accepts all values for maxFFTSize without crashing
- (Pd) now gives a decent error when not providing enough 'channels' in 'multichannel' arrays instead of crashing

## Known Bugs:
- BufNoveltySlice might generate garbage strange values in the first frame
- AmpSlice is noisy when some parameters are changed

===
# beta-00: the big plunge
date: 8 July 2019

## New Features:
- overview in SuperCollider, PureData and CLI
- Max help overall review (uneven in completion)
- support for 3 OSs (Mac, Windows, Linux) and 4 CCEs (Max, PureData, SuperCollider, CLI)

## Known Bugs:
- HPSS still cracks when percussive filter is violently moved up
- NMF still creates NaNs in some edge cases, for instance when trying to factorise digital silence
- BufOnsetSlice with a maxFFTSize value of less than 1024 will crash
- BufNoveltySlice might generate garbage strange values in the first frame
- AmpSlice is noisy when some parameters are changed
- (Pd) help files are not finished yet!
- (Pd) not providing enough 'channels' in 'multichannel' arrays will crash

===
# Alpha-08: 2 new objects, and last interface change
date: 15 June 2019

## New Objects:
- NoveltySlice: a realtime version of the buffer based algo!
- AmpSeg/BufAmpSeg: an amplitude based segmentation powertool

## New Features:
- BREAKING CHANGE: (buf)onsetslice: "function" is now "metric"
- bufnoveltyslice: now segmenting on other features (mfccs, pitch, etc)
- BREAKING CHANGE: the threshold of bufnoveltyslice are now more stabble but will change some of the values.
- BREAKING CHANGE: (BufOnsetSlice, BufTransientSlice, BufNoveltySlice) the indices buffer does not return the query boundaries anymore, just valid detected onsets.
- (buf)pitch now has 'minFreq', 'maxFreq' and 'unit' (MIDI conversion)

## New Examples:
- (SC) working 2-passes-folder-load-bufcompose
- (SC) proper MFCC example (thanks to Sam)
- (max: removed dependencies on descriptor~) (SC: new example) now using fluid.bufpectralshape and fluid.bufstats in all *NMF*

## Bug Fixes:
- many again!

## Known Bugs:
- HPSS still cracks when percussive filter is violently move up
- NMF still creates NaNs in some edge cases

===
# Alpha-07: post-plenary-interface-update: hopefully last major parameter names
date: 4 June 2019

## New Objects:
(SC + CLI): all the alpha-06 ones!

## New Features:
- BREAKING CHANGES: parameter/attributes/messages interface unification
  - all nmf: "rank" is now "components", "numIters" is now "iterations"
  - all slicers with "debounce" now use "minSliceLength"
  - (buf)transient* "debounce" is now "clumpLength", and "minSlice" is "minSliceLength",
  - all "winSize" are now "windowSize"
  - "(max)numCoefs" is now "(max)numCoeffs"
- BREAKING CHANGE: spectralshape: now in Hertz

## Bug Fixes:
(MAX) bang gimme problem

===
# Alpha-06: yet again, new (descriptor) objects
date: 20 May 2019 - Max Mac only, plenary attendees focused

## New Objects:
- BufStats: computes various statistics on time-series (as buffer channel) and their time derivative
- MelBands/BufMelBands: an approximation of human listening of pitch
- MFCC/BufMFCC: a sturdy spectral shape descriptor
- Loudness/BufLoudness: EBU-128 standard capable loudness descriptor

## New Features:
- (MAX) skeleton of reference to allow attributes and arguments autocompletion (with quirks) to help coding

## New Examples:
(MAX) Just-In-Time NMF-based classifier

## Bug Fixes:
many, but not all :-)

## Known Bugs:
plenty of quirks to iron out, but let us know if you find any you have not flagged before

===
# Alpha-05: yet more new cool objects
date: 9 May 2019

## New Objects:
- bufSpectralShape: the buffer version of the same object
- NMFFilter: running real-time nmf on seeding bases passed as a buffer
- Pitch/BufPitch: pitch and confidence descriptor (3 different algorithms)

## New Features:
- (MAX) reset message in buf* objects now reset to instantiation values rather than defaults
- (CLI) basic help is working (-h) to read the name of the parameters

## Bug Fixes:
- (SC) fftSize below 4 are now rejected as advertised
- (SC) fixed the error handling of all buf* objects - getting a useful message instead of errors on the error string

## Known Bugs:
- MAX+CLI: buf* will be capped by maxfftsize default
- HPSS: percussive filter above 35 generate noises and glitches
- (buf)transient will hang the application if the order is set too high
- small noise when starting fft processes, sometimes (hard to reproduce)

===
# Alpha-04a: quick bug fix and stereo toy examples of buf* processes
date: 7/4/2019

## New Examples:
- stereo toy examples to test and showcase behaviour of stereo buf* processes.

## Bug Fixes:
- stereo inputs to buf* slicing code were crashing the processes. now fixed.
- (MAX) NMF example folder updated

## Known Bugs Still Unfixed:
- MAX: autocompletion of @arguments and in/outlet assist strings are on their way, but not yet implemented.
- HPSS: percussive filter above 35 generate noises and glitches

===
# Alpha-04: Post-Refactor-Major-BreakAll-Release
date: 4/4/2019

## New Environment:
- EXPERIMENTAL: early release of a basic command-line interface for the buffer objects. Check the readme for the (very small) set of indications of how it works.

## New Objects:
- OnsetSlice and BufOnsetSlice: a collection of 10 different onset detection functions for real-time and buffers.
- SpectralShape: a real-time spectral descriptor object, giving the 7 most common spectral shape descriptors (centroid, spread, skewness, etc.)

## New Features:
- BREAKING CHANGE: ‘process' method is significantly different - it doesn't exist anymore in Max (bang)
- BREAKING CHANGE (MAX): all parameters are now attributes
- All parameters of are now modulatable - new maxima parameters to set the ranges of those modulations are instantiation only.
- BREAKING CHANGE (MAX): fft parameters are now a single list
- BREAKING CHANGE: all parameters have been renamed in a consistent, verbose way.
- Warning attribute for all objects (in max, window, in SC turn verbose up (see manual)): reports warnings when attribute values try to go out of range
- BREAKING CHANGE: bufcompose is now single-in, single-out and has clearer syntax (and state and reset in MAX)
- SC: Buf* have a completion action, as well as clean update of the modified buffers.

## New Examples:
- both: 2-pass mass bufcompose for a folder, that makes it 60x faster in Max and 4x in SC (we're working on the latter)

## Bug Fixes:
- complete C++ code rewrite under the hood, therefore many, many fixes and some likely new bugs

## Known Bugs Still Unfixed:
- MAX: autocompletion of @arguments and in/outlet assist strings are on their way, but not yet implemented.
- HPSS: percussive filter above 35 generate noises and glitches

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
