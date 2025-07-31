/*
Part of the Fluid Corpus Manipulation Project (http://www.flucoma.org/)
Copyright University of Huddersfield.
Licensed under the BSD-3 License.
See license.md file in the project root for full license information.
This project has received funding from the European Research Council (ERC)
under the European Union’s Horizon 2020 research and innovation programme
(grant agreement No 725899).
*/

/* Optionally using linear programming method from
Neri, J., and Depalle, P., "Fast Partial Tracking of Audio with Real-Time
Capability through Linear Programming". Proceedings of DAFx-2018.
*/

#pragma once

#include "Munkres.hpp"
#include "../../data/FluidIndex.hpp"
#include "../../data/FluidMemory.hpp"
#include <Eigen/Core>
#include <cmath>
#include <queue>

namespace fluid {
namespace algorithm {

struct SinePeak
{
  double freq;
  double logMag;
  bool   assigned;
};

struct SineTrack
{

  SineTrack(Allocator& alloc) : peaks(alloc) {}

  SineTrack(rt::vector<SinePeak>&& p, index s, index e, bool a, bool ass,
            index t)
      : peaks{p}, startFrame{s}, endFrame{e}, active{a}, assigned{ass}, trackId{
                                                                            t}
  {}

  rt::vector<SinePeak> peaks;

  index startFrame;
  index endFrame;
  bool  active;
  bool  assigned;
  index trackId;
};

class PartialTracking
{
  using ArrayXd = Eigen::ArrayXd;
  template <typename T>
  using vector = rt::vector<T>;

public:
  PartialTracking(Allocator& alloc)
      : mTracks(alloc), mPrevPeaks(0, alloc), mPrevTracks(0, alloc)
  {}

  void init()
  {
    using namespace std;

    mCurrentFrame = 0;
    mTracks.clear();
    mPrevPeaks.clear();
    mPrevTracks.clear();
    mZetaA = 0;
    mZetaF = 0;
    mDelta = 0;
    mPrevMaxAmp = 0;
    mLastTrackId = 1;
    mInitialized = true;
  }

  index minTrackLength() { return mMinTrackLength; }

  void processFrame(vector<SinePeak>& peaks, double maxAmp,
                    index minTrackLength, double birthLowThreshold,
                    double birthHighThreshold, index method, double zetaA,
                    double zetaF, double delta, Allocator& alloc)
  {
    assert(mInitialized);
    mMinTrackLength = minTrackLength;
    mBirthLowThreshold = birthLowThreshold;
    mBirthHighThreshold = birthHighThreshold;
    mBirthRange = mBirthLowThreshold - mBirthHighThreshold;
    if (zetaA != mZetaA || zetaF != mZetaF || delta != mDelta)
    {
      mZetaA = zetaA;
      mZetaF = zetaF;
      mDelta = delta;
      updateVariances();
    }
    if (method == 0)
      assignGreedy(peaks, maxAmp, alloc);
    else
      assignMunkres(peaks, maxAmp, alloc);
    mCurrentFrame++;
  }

  void prune()
  {
    auto iterator =
        std::remove_if(mTracks.begin(), mTracks.end(), [&](SineTrack track) {
          return (track.endFrame >= 0 &&
                  track.endFrame <= mCurrentFrame - mMinTrackLength);
        });
    mTracks.erase(iterator, mTracks.end());
  }


  vector<SinePeak> getActivePeaks(Allocator& alloc)
  {
    vector<SinePeak> sinePeaks(0, alloc);
    index            latencyFrame = mCurrentFrame - mMinTrackLength;
    if (latencyFrame < 0) return sinePeaks;
    for (auto&& track : mTracks)
    {
      if (track.startFrame > latencyFrame) continue;
      if (track.endFrame >= 0 && track.endFrame <= latencyFrame) continue;
      if (track.endFrame >= 0 &&
          track.endFrame - track.startFrame < mMinTrackLength)
        continue;
      sinePeaks.push_back(
          track.peaks[asUnsigned(latencyFrame - track.startFrame)]);
    }
    return sinePeaks;
  }

private:
  void updateVariances()
  {
    using namespace std;
    mVarA = -pow(mZetaA, 2) * log((mDelta - 1) / (mDelta - 2));
    mVarF = -pow(mZetaF, 2) * log((mDelta - 1) / (mDelta - 2));
  }

  void assignMunkres(vector<SinePeak>& sinePeaks, double maxAmp,
                     Allocator& alloc)
  {
    using namespace Eigen;
    using namespace std;

    typedef Array<int, Dynamic, Dynamic> ArrayXXb;
    for (auto&& track : mTracks) { track.assigned = false; }

    if (mPrevPeaks.empty())
    {
      mPrevPeaks = sinePeaks;
      mPrevTracks = vector<index>(sinePeaks.size(), 0, alloc);
      return;
    }

    index                   N = asSigned(mPrevPeaks.size());
    index                   M = asSigned(sinePeaks.size());
    ScopedEigenMap<ArrayXd> peakFreqs(M, alloc);
    ScopedEigenMap<ArrayXd> peakAmps(M, alloc);
    ScopedEigenMap<ArrayXd> prevFreqs(N, alloc);
    ScopedEigenMap<ArrayXd> prevAmps(N, alloc);
    vector<index>           trackAssignment(asUnsigned(M), -1, alloc);
    if (sinePeaks.size() > 0)
    {
      for (index i = 0; i < M; i++)
      {
        peakFreqs(i) = sinePeaks[asUnsigned(i)].freq;
        peakAmps(i) = sinePeaks[asUnsigned(i)].logMag;
      }
      for (index i = 0; i < N; i++)
      {
        prevFreqs(i) = mPrevPeaks[asUnsigned(i)].freq;
        prevAmps(i) = mPrevPeaks[asUnsigned(i)].logMag;
      }
      ScopedEigenMap<ArrayXXd> deltaF(N, M, alloc);
      deltaF.setZero();
      deltaF.colwise() = prevFreqs;
      for (index i = 0; i < N; i++) { deltaF.row(i) -= peakFreqs; }
      ScopedEigenMap<ArrayXXd> deltaA(N, M, alloc);
      deltaA.setZero();
      deltaA.colwise() = prevAmps;
      for (index i = 0; i < N; i++) { deltaA.row(i) -= peakAmps; }

      ScopedEigenMap<ArrayXXd> usefulCost(N, M, alloc);
      usefulCost =
          1 - (-deltaF.square() / mVarF - deltaA.square() / mVarA).exp();
      ScopedEigenMap<ArrayXXd> spuriousCost(N, M, alloc);
      spuriousCost = 1 - (1 - mDelta) * usefulCost;
      ScopedEigenMap<ArrayXXd> cost(N, M, alloc);
      ScopedEigenMap<ArrayXXb> useful(N, M, alloc);
      for (index i = 0; i < N; i++)
      {
        for (index j = 0; j < M; j++)
        {
          if (usefulCost(i, j) < spuriousCost(i, j))
          {
            cost(i, j) = std::abs(usefulCost(i, j));
            useful(i, j) = true;
          }
          else
          {
            cost(i, j) = spuriousCost(i, j);
            useful(i, j) = false;
          }
        }
      }
      ScopedEigenMap<ArrayXi> assignment(N, alloc);
      Munkres                 munkres(N, M, alloc);
      munkres.process(cost, assignment, alloc);
      for (index i = 0; i < N; i++)
      {
        index p = assignment(i);
        bool  aboveBirthThreshold =
            mPrevPeaks[asUnsigned(i)].logMag >
            birthThreshold(mPrevPeaks[asUnsigned(i)], mPrevMaxAmp);
        if (assignment(i) >= useful.cols()) continue;
        if (useful(i, assignment(i)) && mPrevTracks[asUnsigned(i)] > 0 &&
            mPrevPeaks[asUnsigned(i)].assigned)
        {
          for (auto& t : mTracks)
          {
            if (t.trackId == mPrevTracks[asUnsigned(i)])
            {
              trackAssignment[asUnsigned(p)] = t.trackId;
              sinePeaks[asUnsigned(p)].assigned = true;
              t.assigned = true;
              t.peaks.push_back(sinePeaks[asUnsigned(p)]);
            }
          }
        }
        else if (aboveBirthThreshold && useful(i, assignment(i)) &&
                 !mPrevPeaks[asUnsigned(i)].assigned)
        {
          mLastTrackId = mLastTrackId + 1;
          auto newTrack = SineTrack{vector<SinePeak>(2, alloc),
                                    mCurrentFrame - 1,
                                    -1,
                                    true,
                                    true,
                                    mLastTrackId};
          newTrack.peaks[0] = mPrevPeaks[asUnsigned(i)];
          newTrack.peaks[1] = sinePeaks[asUnsigned(p)];
          mTracks.push_back(newTrack);
          sinePeaks[asUnsigned(p)].assigned = true;
          trackAssignment[asUnsigned(p)] = newTrack.trackId;
        }
      }
    }
    // dying tracks
    for (auto&& track : mTracks)
    {
      if (track.active && !track.assigned)
      {
        track.active = false;
        track.endFrame = mCurrentFrame;
      }
    }
    mPrevTracks = trackAssignment;
    mPrevPeaks = sinePeaks;
    mPrevMaxAmp = maxAmp;
  }

  double birthThreshold(SinePeak peak, double maxAmp)
  {
    return maxAmp + mBirthLowThreshold - mBirthRange +
           mBirthRange * std::pow(0.0075, peak.freq / 20000.0);
  }

  void assignGreedy(vector<SinePeak>& sinePeaks, double maxAmp,
                    Allocator& alloc)
  {
    using namespace std;
    rt::vector<tuple<double, SineTrack*, SinePeak*>> distances(0, alloc);
    for (auto&& track : mTracks) { track.assigned = false; }
    for (auto& track : mTracks)
    {
      if (track.active)
      {
        for (auto&& peak : sinePeaks)
        {
          double dist =
              1 - exp(-pow(track.peaks.back().freq - peak.freq, 2) / mVarF -
                      pow(track.peaks.back().logMag - peak.logMag, 2) / mVarA);
          distances.push_back(std::make_tuple(dist, &track, &peak));
        }
      }
    }

    sort(distances.begin(), distances.end(),
         [](tuple<double, SineTrack*, SinePeak*> const& t1,
            tuple<double, SineTrack*, SinePeak*> const& t2) {
           return get<0>(t1) < get<0>(t2);
         });

    for (auto&& pairing : distances)
    {
      if (!get<1>(pairing)->assigned && !get<2>(pairing)->assigned &&
          get<0>(pairing) <
              (1 - (1 - mDelta) * get<0>(pairing))) // useful vs spurious
      {
        get<1>(pairing)->peaks.push_back(*get<2>(pairing));
        get<1>(pairing)->assigned = true;
        get<2>(pairing)->assigned = true;
      }
    }
    // new tracks
    index nBorn = 0, nDead = 0;
    for (auto&& peak : sinePeaks)
    {
      if (!peak.assigned && peak.logMag > birthThreshold(peak, maxAmp))
      {
        nBorn++;
        mTracks.push_back(SineTrack{vector<SinePeak>(1, peak, alloc),
                                    static_cast<int>(mCurrentFrame), -1, true,
                                    true, mLastTrackId++});
      }
    }
    // diying tracks
    for (auto&& track : mTracks)
    {
      if (track.active && !track.assigned)
      {
        nDead++;
        track.active = false;
        track.endFrame = mCurrentFrame;
      }
    }
  }

  index             mMinTrackLength{15};
  index             mCurrentFrame{0};
  vector<SineTrack> mTracks;
  bool              mInitialized{false};
  vector<SinePeak>  mPrevPeaks;
  vector<index>     mPrevTracks;
  //  Munkres           mMunkres;
  double mZetaA{0};
  double mVarA{0};
  double mZetaF{0};
  double mVarF{0};
  double mDelta{0};
  double mPrevMaxAmp{0};
  index  mLastTrackId{1};
  double mBirthLowThreshold{-24.};
  double mBirthHighThreshold{-60.};
  double mBirthRange{36.};
};
} // namespace algorithm
} // namespace fluid
