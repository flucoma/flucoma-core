/*
Part of the Fluid Corpus Manipulation Project (http://www.flucoma.org/)
Copyright 2017-2019 University of Huddersfield.
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

#include "../util/Munkres.hpp"
#include <Eigen/Core>
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
  std::vector<SinePeak> peaks;
  int    startFrame;
  int    endFrame;
  bool   active;
  bool   assigned;
  size_t trackId;
};

class PartialTracking
{
  using ArrayXd = Eigen::ArrayXd;
  template <typename T>
  using vector = std::vector<T>;

public:
  void init()
  {
    mCurrentFrame = 0;
    mTracks = std::vector<SineTrack>();
    updateVariances();
    mInitialized = true;
  }

  int minTrackLength() { return mMinTrackLength; }

  void processFrame(vector<SinePeak> peaks, double maxAmp)
  {
    assert(mInitialized);
    if (mMethod == 0)
      assignGreedy(peaks, maxAmp);
    else
      assignMunkres(peaks, maxAmp);
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

  void setMinTrackLength(int minTrackLength)
  {
    mMinTrackLength = minTrackLength;
  }

  int getMinTrackLength()
  {
    return mMinTrackLength;
  }


  void setBirthLowThreshold(double threshold)
  {
    mBirthLowThreshold = threshold;
    mBirthRange = mBirthLowThreshold - mBirthHighThreshold;
  }

  void setBirthHighThreshold(double threshold)
  {
    mBirthHighThreshold = threshold;
    mBirthRange = mBirthLowThreshold - mBirthHighThreshold;
  }

  void setMethod(int method) { mMethod = method; }

  void updateVariances()
  {
    mVarA = -pow(mZetaA, 2) * log((mDelta - 1) / (mDelta - 2));
    mVarF = -pow(mZetaF, 2) * log((mDelta - 1) / (mDelta - 2));
  }

  void setZetaA(double zetaA)
  {
    mZetaA = zetaA;
    updateVariances();
  }

  void setZetaF(double zetaF)
  {
    mZetaF = zetaF;
    updateVariances();
  }

  void setDelta(double delta)
  {
    mDelta = delta;
    updateVariances();
  }

  bool initialized() { return mInitialized; }

  vector<SinePeak> getActivePeaks()
  {
    vector<SinePeak> sinePeaks;
    int              latencyFrame = mCurrentFrame - mMinTrackLength;
    if (latencyFrame < 0) return sinePeaks;
    for (auto&& track : mTracks)
    {
      if (track.startFrame > latencyFrame) continue;
      if (track.endFrame >= 0 && track.endFrame <= latencyFrame) continue;
      if (track.endFrame >= 0 &&
          track.endFrame - track.startFrame < mMinTrackLength)
        continue;
      sinePeaks.push_back(track.peaks[latencyFrame - track.startFrame]);
    }
    return sinePeaks;
  }

private:
  void assignMunkres(vector<SinePeak> sinePeaks, double maxAmp)
  {
    using namespace Eigen;
    typedef Array<bool, Dynamic, Dynamic> ArrayXXb;
    for (auto&& track : mTracks) { track.assigned = false; }

    if (mPrevPeaks.empty())
    {
      mPrevPeaks = sinePeaks;
      mPrevTracks = vector<size_t>(sinePeaks.size(), 0);
      return;
    }

    int            N = mPrevPeaks.size();
    int            M = sinePeaks.size();
    ArrayXd        peakFreqs(M);
    ArrayXd        peakAmps(M);
    ArrayXd        prevFreqs(N);
    ArrayXd        prevAmps(N);
    vector<size_t> trackAssignment(M, -1);
    if (sinePeaks.size() > 0)
    {
      for (int i = 0; i < M; i++)
      {
        peakFreqs(i) = sinePeaks[i].freq;
        peakAmps(i) = sinePeaks[i].logMag;
      }
      for (int i = 0; i < N; i++)
      {
        prevFreqs(i) = mPrevPeaks[i].freq;
        prevAmps(i) = mPrevPeaks[i].logMag;
      }
      ArrayXXd deltaF = ArrayXXd::Zero(N, M);
      deltaF.colwise() = prevFreqs;
      for (int i = 0; i < N; i++) { deltaF.row(i) -= peakFreqs; }
      ArrayXXd deltaA = ArrayXXd::Zero(N, M);
      deltaA.colwise() = prevAmps;
      for (int i = 0; i < N; i++) { deltaA.row(i) -= peakAmps; }

      ArrayXXd usefulCost =
          1 - (-deltaF.square() / mVarF - deltaA.square() / mVarA).exp();
      ArrayXXd spuriousCost = 1 - (1 - mDelta) * usefulCost;
      ArrayXXd cost(N, M);
      ArrayXXb useful(N, M);
      for (int i = 0; i < N; i++)
      {
        for (int j = 0; j < M; j++)
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
      ArrayXi assignment(N);
      mMunkres.init(N, M);
      mMunkres.process(cost, assignment);
      for (int i = 0; i < N; i++)
      {
        int  p = assignment(i);
        bool aboveBirthThreshold =
            mPrevPeaks[i].logMag > birthThreshold(mPrevPeaks[i], mPrevMaxAmp);
        if (assignment(i) >= useful.cols()) continue;
        if (useful(i, assignment(i)) && mPrevTracks[i] > 0 &&
            mPrevPeaks[i].assigned)
        {
          for (auto& t : mTracks)
          {
            if (t.trackId == mPrevTracks[i])
            {
              trackAssignment[p] = t.trackId;
              sinePeaks[p].assigned = true;
              t.assigned = true;
              t.peaks.push_back(sinePeaks[p]);
            }
          }
        }
        else if (aboveBirthThreshold && useful(i, assignment(i)) &&
                 !mPrevPeaks[i].assigned)
        {
          mLastTrackId = mLastTrackId + 1;
          auto newTrack =
              SineTrack{vector<SinePeak>{mPrevPeaks[i], sinePeaks[p]},
                        mCurrentFrame - 1,
                        -1,
                        true,
                        true,
                        mLastTrackId};
          mTracks.push_back(newTrack);
          sinePeaks[p].assigned = true;
          trackAssignment[p] = newTrack.trackId;
        }
      }
    }
    // diying tracks
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

  void assignGreedy(vector<SinePeak> sinePeaks, double maxAmp)
  {
    using namespace std;
    vector<tuple<double, SineTrack*, SinePeak*>> distances;
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
    int nBorn = 0, nDead = 0;
    for (auto&& peak : sinePeaks)
    {
      if (!peak.assigned && peak.logMag > birthThreshold(peak, maxAmp))
      {
        nBorn++;
        mTracks.push_back(SineTrack{vector<SinePeak>{peak},
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

  int               mMinTrackLength{15};
  size_t            mCurrentFrame{0};
  vector<SineTrack> mTracks;
  bool              mInitialized{false};
  vector<SinePeak>  mPrevPeaks;
  vector<size_t>    mPrevTracks;
  Munkres           mMunkres;
  int               mMethod;
  double            mZetaA;
  double            mVarA;
  double            mZetaF;
  double            mVarF;
  double            mDelta;
  double            mPrevMaxAmp{0};
  size_t            mLastTrackId{1};
  double            mBirthLowThreshold{-24.};
  double            mBirthHighThreshold{-60.};
  double            mBirthRange{36.};
};
} // namespace algorithm
} // namespace fluid
