# 1.0.0-TB2.beta3:
Date: Oct SOON, 2021

## CCE
- PureData has parity with other CCE's except for:
	- `load` and `dump` methods for DataSet and LabelSet (workaround in place)
	- Limited help-files. These will be updated A.S.A.P and posted as they are updated.

## Objects
- New Objects
	- FluidGrid: coerce data in a FluidDataSet into a two-dimensional grid
	- FluidStats: compute real-time statistics (mean and stddev) on data streams		
	- MAX: FluidPlotter: Rapidly plot two-dimensional datasets
	- MAX+PD `fluid.buf2list` and `fluid.list2buf` are now compiled objects. Be aware of their new attributes in Max
	- SC: FluidBufToKr and FluidKrToBuf pseudo-Ugens help with the dataset to descriptor to dataset workflow
- New Sounds
	- trombone long tones, and oboe multiphonics, now included.

## Interfaces and Behaviours
- FluidMFCC has a new attribute: "startCoef" which modifies which coefficients are returned by the algorithm. A common use case is to discard the first coefficient.
- MAX+PD: All data objects can be named to share persistent state. For example, `fluid.kdtree~ my-tree` in Max can be used everywhere throughout a patch to reference the same k-d tree named `my-tree`.

## Fixes
- Improved performance with FluidAmpGate
- Crashes caused by (tapin - tapout == 0) on FluidMLP
- Crashes caused by when loading MLP models from JSON
- FluidLoudness no longer spikes to the maximum on digital silence
- Many other fixes, see the `git` log for details.
- BREAKING: BufSelectEvery's method `channelhop`is now `chanhop`to be consistent

## Tooling
- Compilation on Arch Linux is now made possible again by fixing linking against the cutting edge of Boost.
- Building no longer uses AVX instructions by default to allow for an easier experience compiling for M1 Mac and ARM processors.
- Max compilation can now uses the 8.2 SDK

===
# Beta-02:
date: June 2nd, 2021

- New object: (buf)chroma for pitch class histograms
- New overview of all objects
- Log frequency and power magnitude options in SpectralShape

- Log output option to melbands
- Change of MFCC outs
- change of unit of rolloff and new parameter to set the value (instead of fixed to 95%)

- Some protection against thread conflicts in the dataset objects
- Pitch new default when no pitch detected (0 Hz or -999midi)
- Fixed Cosine distances in Onset and MDS
- Fix in PCA for size=dimcount
- Max: basic hints for inlets and outlets

#### Known bugs:
Windows (Max + SC): Pitch's Yin algo yields a few error frames at its start, but not every time. This also affects Novelty by Pitch. We're chasing.

Linux (SC): SpectralShape has infrequent random outputs. We're chasing that one too.

Windows (Max): bufthreaddemo is not working well. The real threading is working but the explanation object is not.

Linux (SC): example 13 (massive parallelisation) is not working.


===
# TB2 Beta 01
April 2021

## All Hosts

### New Objects
* BufSTFT: Run a STFT or ISTFT on a buffer
* BufSelect and BufSelectEvery: cherry picking from Buffers

### Enhancements
* Descriptors objects: Now have adjustable padding (none, window / 2, window - hop)
* DataSet: one step conversion to / from Buffer + access to IDs as LabelSet
* KMeans: Easy access to means and distances; incremental fitting
* PCA: allow transformation to same dimensions as input
* DataSet: new setPoint message for updating extant points, or creating new point if needed
* Slicers: Improved latency calculations and accuracy of Buffer versions
* BufFlatten: Now handles Buffer sub-ranges
* BufScale: Add clipping option
* BufNMFCross: Improve sound, add progress update and continuity control

### Fixes   
* MLP: handle inconsistent dimensions when incremental fitting
* LabelSet: Fix JSON
* MLP: Fix early stopping
* Standardization + Normalization: Fix divide by zero
* MLP: Fix concurrency issue when fitting and transforming on different threads
* Pitch: Avoid negative confidence values

## Supercollider

### Enhancements
* FluidDataSetWr: New interface to use DataSet::setPoint (breaking change)
* KMeans: documentation
* BufStats: documentation

### Fixes
* Argument passing in .kr for some objects
* Build settings on Catalina
* Parameter passing on initialisation for some objects
* Some problems when using internal server
* Some inefficiencies with UGens

## Known issues
* FluidBufThreadDemo can be very slow on Windows (Max and SC)
* SC: Behaviour of FreeSelfWhenDone with FluidBuf* *kr objects is unpredictable on Linux

## Max

### Enhancements
* KMeans help: new examples
* Improved indexing of reference documentation, especially on Max 7
* Explanation of FFT settings' impact in fluid.audiotransport~ help

### Fixes
* UMAP helpfile error
* RobustScale helpfile error
* Build settings on Catalina
* Cross-referencing between dimension reduction documentation

===
# TB2 Alpha 08
January 20 2021

## Bugs fixed:
* Compilation for Windows and Linux
* Realtime memory leak in Supercollider
* Use of SC .kr classes in SynthDefs
* NMFMorph window-related modulation artifacts

===

# TB2 Alpha 07
December 22 2020

## Improvements:

* KDTree is much faster
* AudioTransport is more accurate

## New:
* new UMAP dim redux
* new RobustScaler

# Breaking changes:
* Max: native support of dictionaries (removing the need for dict.(de)serialise~)
* Max: behaviour of read and write are like native objects (brings menu if no path provided)
* SC: a new sync dance

# Non-breaking changes:
* SC: no more name to DataSet and LabelSet

===

# TB2 Alpha 06
September 24 2020

# New objects:
* BufThresh
* BufScale

# New Features:
* weights and outliersCutoff to bufstats
* transformjoin to datasetquery
* radius limit to kdtree

# New examples:
* 10-weights in stats
* 10a-outliers in stats
* 11-joining and weighing datasets (advanced!)

# Known bugs:
* large dictionary dumps are still problematic in Max
* negative confidence in pitch is still there (will be noticed more with the new examples)

===

# TB2 Alpha 05
August 6 2020

## New examples:

* Max: working autoencoder as dataredux and interpolation up
* Max: tutorial on simple regression feature comparison
* SC: working autoencoder as dataredux (the other part will come soon here)

## Breaking changes:

### MLPregressor

* outputLayer is now tapOut. Its numbering is now different: 0 counting from the input layer, so 1 is the first hidden layer and -1 is the default (the last, like all our interface).
* it has a new friend, tapIn, which allows you to feed the data for predict and predictpoint in the middle somewhere. 0 is the input, and 1 is the first hidden layer

### Normalize and Standardize

* now have an ‘inverse’ parameters for transform and transformPoint to allow query from the transformed space to the original.

### PCA

Now returns/passes the variance, aka the fidelity of the new representation for a given number of dimensions.

### SC-kr

Most TB1 objects have a blocking mode in KR which allow to keep them on the main buffer thread of the server (faster for small jobs avoiding large memory copying)

## Bugfixes:

* all json load/save/states bugs/oddities reported
* Max: cluttering when buffer resizing
* bufnmf parameter check order
* general buf resize sanity check
* Max: buf resize of dataset is done just in low priority thread (faking mode 2 otherwise)
* SC: most dataset objects in KR are much more efficient
* SC: kr bus assignation

## Known Bugs

Should mostly be edge cases. Feel free to test against your reports and update the issues.

===

# TB2 Alpha 04
July 17 2020

===
# TB2 Alpha 03
June 20 2020

## Major changes in the SC interface.
 Most of your code will break but this is for the best. To make it short, read the help files and example code. They rock.

* we can now populate dataset 3 ways:  via Dictionaries, via single buffer triggered from language (Routine, the old way, but faster) and all on the server-side (via Done.kr and triggers). And it is mixed and matched too!
* added utilities to build corpus change their format
* all help files have examples
* the learning examples are updated too

## Max
* unified dataset builder helpers format
* helpfiles have had a lot of typos/loading issues now sorted
* super mega bug in TB1 that nobody found before because it was so edge-case… now sorted :wink:

## All
* 3 new sounds as an example (feel free to let us know what is missing, and who knows, you might even want to contribute)
* [breaking change] human-readable JSON

===

# TB2 Alpha02
May 21 2020

## A few notes:

    Keep your version of TB1 RC1. The TB2 download has everything, but is from another (much more daring) branch of development. We should have fixed all the bugs but we are not certain it is gig ready!

    The many, many interface break should moan in your CCE window. The most important one is that datasets do not need a fixed size as argument, so most arguments have disappeared from objects. This is a lot more flexible as you will discover.

## New stuff:

    As requested, we have commented the tutorials in Max - the SC people should read them too (they will be translated soon) as it is simple Max yet introduce interesting database concepts with links to scikitlearn documentation.

## Quite Improved:

    Helpfiles - most of them are great, some ok, a few stubs but in those cases the learning examples are there to help
    so many bugs fixed! Gazillions!

## New possibilities:

    dimensionality reduction algorithms (PCA and MDS (with 7 metrics)
    making subsets of dataset via datasetquery
    new buffer manipulation utilities in Max
    weighted KNN classification / regression
    dumping datasets (dump is all as json, print is a useful sample as string)

## New behaviours:

    the said weighted KNN classification / regression is now default
    major interface unification inspired by scikit learn’s syntax to help tap into their learning resources

## SC specific:

    Redesign of language classes and server plugin so that Dataset, Labelset and the various models (KDTree etc) persist for the whole server lifetime, and don’t vanish on cmd+.
    Dataset and Labelset will now throw an exception if you try and create two instances with the same name. However, all instances are cached, similar to Buffer, and there are now class level ‘at’ functions so you can retrieve an extant instance using its name.
    Work on a much streamlined approach for populating and querying without so much back and forth between language and server is ongoing, and will be our main focus in the next alpha. Experimental approaches are currently at the promising-but-dangerous stage.

## Known Issues:

    Max: we have not got autocompletion working for the new objects yet, nor documentation. (You will get some JS errors with the help files because of that, no worries)

===
# TB2 Alpha-01:

date: December 5th, 2019

## New Features:

* Breaking change: all dataset interfaces are streamlined, please check the help-files if you have coded something personally. This is likely to change again as we tweak the workflow, so feedback welcome.

## New Examples:

* All plenary example updated
* All PA’s naive learning patches included
(still underdocumented)
* One-liner references in SC, imported as ‘ugly temp ref’ in Max for the interpolation and hybridisation objects (soon to be made clearer and less ugly)
* SC: simple examples ported

## Known Bugs:

* bufnmfcross sound quality on large files could be less smeary
* bufnmfcross: selecting boundaries of both input buffers is not implemented yet
* SC workflow is slow because of many language-server sync (more streamlined workflow coming)
* audiotransport is still quantised by bin (phase interpolation is not trivial)
* audiotransport windows are restricted to fft sizes for now
* MAX: (buf)audiotransport and bufnmfcross should be their own standalone object soon
* MAX: bufnmfcross: progress bar does not work in blocking 0
* instacrash with fluid.dataset [getPoint] on an empty dataset
* Max: a few ref missing
* Max windows compile on its way soon

===
# 1.0.0-RC1
date: 31 March 2020

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

## Known Bugs/Issues:
* (Mac) Notarization is not implemented yet, since most CCEs are working on workarounds...
* (SC) Current implementation of non-realtime objects does not work with remote servers
* (Win - SC) calling cancel on threaded non-realtime processes crashes scsynth

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
