#include "../core/global.h"
#include "../core/fileutils.h"
#include "../core/makedir.h"
#include "../core/config_parser.h"
#include "../core/elo.h"
#include "../core/timer.h"
#include "../dataio/sgf.h"
#include "../search/asyncbot.h"
#include "../search/patternbonustable.h"
#include "../program/setup.h"
#include "../program/play.h"
#include "../command/commandline.h"
#include "../core/test.h"
#include "../main.h"

#include <algorithm>
#include <csignal>
#include <cmath>

using namespace std;


static std::atomic<bool> sigReceived(false);
static std::atomic<bool> shouldStop(false);
static void signalHandler(int signal)
{
  if(signal == SIGINT || signal == SIGTERM) {
    sigReceived.store(true);
    shouldStop.store(true);
  }
}

struct MatchEloConfig {
  bool enabled;
  int64_t reportPeriodGames;
  int baselineBotIdx;
  int candidateBotIdx;
  double priorGames;
  bool sprtEnabled;
  double sprtElo0;
  double sprtElo1;
  double sprtAlpha;
  double sprtBeta;
};

struct MatchEloGameRecord {
  double firstWins;
  double secondWins;
  double draws;

  MatchEloGameRecord()
    : firstWins(0.0), secondWins(0.0), draws(0.0)
  {}
};

class MatchEloTracker {
 public:
  MatchEloTracker(const vector<string>& botNames, const MatchEloConfig& cfg)
    : names(botNames),
      config(cfg),
      numBots((int)botNames.size()),
      records((size_t)numBots * (size_t)numBots),
      scoredGames(0),
      drawGames(0),
      noResultGames(0),
      unfinishedGames(0)
  {}

  void addGame(const FinishedGameData& gameData) {
    if(!gameData.endHist.isGameFinished) {
      unfinishedGames += 1;
      return;
    }
    if(gameData.endHist.isNoResult) {
      noResultGames += 1;
      return;
    }

    int bIdx = gameData.bIdx;
    int wIdx = gameData.wIdx;
    if(bIdx < 0 || bIdx >= numBots || wIdx < 0 || wIdx >= numBots)
      return;

    MatchEloGameRecord& record = records[(size_t)bIdx * (size_t)numBots + (size_t)wIdx];
    if(gameData.endHist.winner == P_BLACK) {
      record.firstWins += 1.0;
    }
    else if(gameData.endHist.winner == P_WHITE) {
      record.secondWins += 1.0;
    }
    else {
      record.firstWins += 0.5;
      record.secondWins += 0.5;
      record.draws += 1.0;
      drawGames += 1;
    }
    scoredGames += 1;
  }

  int64_t getScoredGames() const {
    return scoredGames;
  }

  string report(const std::map<string,double>& timeUsedByBotMap, const std::map<string,double>& movesByBotMap) const {
    ostringstream out;
    out << "Match Elo report after " << scoredGames << " scored games";
    if(drawGames > 0 || noResultGames > 0 || unfinishedGames > 0) {
      out << " (draws " << drawGames
          << ", noResult skipped " << noResultGames
          << ", unfinished skipped " << unfinishedGames << ")";
    }
    out << "\n";

    if(scoredGames <= 0) {
      out << "No scored games yet";
      return out.str();
    }

    vector<ComputeElos::WLRecord> winMatrix((size_t)numBots * (size_t)numBots);
    vector<double> gamesByBot(numBots,0.0);
    for(int i = 0; i<numBots; i++) {
      for(int j = 0; j<numBots; j++) {
        const MatchEloGameRecord& record = records[(size_t)i * (size_t)numBots + (size_t)j];
        winMatrix[(size_t)i * (size_t)numBots + (size_t)j] = ComputeElos::WLRecord(record.firstWins,record.secondWins);
        gamesByBot[i] += record.firstWins + record.secondWins;
        gamesByBot[j] += record.firstWins + record.secondWins;
      }
    }

    double priorWL = 0.5 * config.priorGames;
    vector<double> elos = ComputeElos::computeElos(winMatrix.data(),numBots,priorWL,1000,0.0001,NULL);
    vector<double> stdevs = ComputeElos::computeApproxEloStdevs(elos,winMatrix.data(),numBots,priorWL);

    vector<int> order;
    for(int i = 0; i<numBots; i++)
      order.push_back(i);
    std::sort(order.begin(),order.end(),[&](int a, int b) {
      return elos[a] > elos[b];
    });

    out << "Standings:";
    for(int idx: order) {
      const string& name = names[idx];
      double avgMoveTime = 0.0;
      bool hasAvgMoveTime = false;
      auto timeIter = timeUsedByBotMap.find(name);
      auto movesIter = movesByBotMap.find(name);
      if(timeIter != timeUsedByBotMap.end() && movesIter != movesByBotMap.end() && movesIter->second > 0.0) {
        avgMoveTime = timeIter->second / movesIter->second;
        hasAvgMoveTime = true;
      }

      out << "\n  "
          << Global::strprintf("%-18s Elo %+7.1f +/- %5.1f  games %.0f", name.c_str(), elos[idx], stdevs[idx], gamesByBot[idx]);
      if(hasAvgMoveTime)
        out << "  avgMoveSec " << Global::strprintf("%.4f", avgMoveTime);
    }

    if(config.baselineBotIdx >= 0 && config.baselineBotIdx < numBots &&
       config.candidateBotIdx >= 0 && config.candidateBotIdx < numBots &&
       config.baselineBotIdx != config.candidateBotIdx) {
      out << "\n" << reportPair(config.candidateBotIdx,config.baselineBotIdx);
    }
    return out.str();
  }

 private:
  const vector<string> names;
  const MatchEloConfig config;
  const int numBots;
  vector<MatchEloGameRecord> records;
  int64_t scoredGames;
  int64_t drawGames;
  int64_t noResultGames;
  int64_t unfinishedGames;

  MatchEloGameRecord getCombinedRecord(int first, int second) const {
    const MatchEloGameRecord& direct = records[(size_t)first * (size_t)numBots + (size_t)second];
    const MatchEloGameRecord& reverse = records[(size_t)second * (size_t)numBots + (size_t)first];

    MatchEloGameRecord combined;
    combined.firstWins = direct.firstWins + reverse.secondWins;
    combined.secondWins = direct.secondWins + reverse.firstWins;
    combined.draws = direct.draws + reverse.draws;
    return combined;
  }

  static double logisticWinProb(double eloDiff) {
    return 1.0 / (1.0 + pow(10.0,-eloDiff / 400.0));
  }

  string reportPair(int first, int second) const {
    MatchEloGameRecord record = getCombinedRecord(first,second);
    double total = record.firstWins + record.secondWins;
    ostringstream out;
    out << "Pair " << names[first] << " vs " << names[second] << ": ";
    if(total <= 0.0) {
      out << "no games";
      return out.str();
    }

    double adjustedTotal = total + config.priorGames;
    double adjustedScore = record.firstWins + 0.5 * config.priorGames;
    double winProb = adjustedScore / adjustedTotal;
    winProb = std::min(1.0 - 1e-12, std::max(1e-12, winProb));
    double elo = 400.0 * log10(winProb / (1.0 - winProb));
    double eloStderr = 400.0 / log(10.0) * sqrt(1.0 / std::max(1e-12, adjustedTotal * winProb * (1.0 - winProb)));
    double los = 0.5 * (1.0 + erf(elo / std::max(1e-12, eloStderr) / sqrt(2.0)));

    out << Global::strprintf(
      "%.1f/%.0f (%.2f%%), Elo %+7.1f +/- %.1f, LOS %.2f%%, W-L-D %.1f-%.1f-%.0f",
      record.firstWins,
      total,
      100.0 * record.firstWins / total,
      elo,
      eloStderr,
      100.0 * los,
      record.firstWins - 0.5 * record.draws,
      record.secondWins - 0.5 * record.draws,
      record.draws
    );

    if(config.sprtEnabled) {
      double p0 = std::min(1.0 - 1e-12, std::max(1e-12, logisticWinProb(config.sprtElo0)));
      double p1 = std::min(1.0 - 1e-12, std::max(1e-12, logisticWinProb(config.sprtElo1)));
      double llr = record.firstWins * log(p1 / p0) + record.secondWins * log((1.0 - p1) / (1.0 - p0));
      double upper = log((1.0 - config.sprtBeta) / config.sprtAlpha);
      double lower = log(config.sprtBeta / (1.0 - config.sprtAlpha));
      string status;
      if(llr >= upper)
        status = "accept H1";
      else if(llr <= lower)
        status = "accept H0";
      else
        status = "continue";
      out << "\n  SPRT [" << config.sprtElo0 << "," << config.sprtElo1 << "] "
          << "LLR " << Global::strprintf("%.3f",llr)
          << " bounds [" << Global::strprintf("%.3f",lower) << "," << Global::strprintf("%.3f",upper) << "] "
          << status;
    }
    return out.str();
  }
};

int MainCmds::match(const vector<string>& args) {
  Board::initHash();
  ScoreValue::initTables();
  Rand seedRand;

  ConfigParser cfg;
  string logFile;
  string sgfOutputDir;
  try {
    KataGoCommandLine cmd("Play different nets against each other with different search settings in a match or tournament.");
    cmd.addConfigFileArg("","match_example.cfg");

    TCLAP::ValueArg<string> logFileArg("","log-file","Log file to output to",false,string(),"FILE");
    TCLAP::ValueArg<string> sgfOutputDirArg("","sgf-output-dir","Dir to output sgf files",false,string(),"DIR");

    cmd.add(logFileArg);
    cmd.add(sgfOutputDirArg);

    cmd.setShortUsageArgLimit();
    cmd.addOverrideConfigArg();

    cmd.parseArgs(args);

    logFile = logFileArg.getValue();
    sgfOutputDir = sgfOutputDirArg.getValue();

    cmd.getConfig(cfg);
  }
  catch (TCLAP::ArgException &e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
    return 1;
  }

  Logger logger(&cfg);
  logger.addFile(logFile);

  logger.write("Match Engine starting...");
  logger.write(string("Git revision: ") + Version::getGitRevision());

  //Load per-bot search config, first, which also tells us how many bots we're running
  vector<SearchParams> paramss = Setup::loadParams(cfg,Setup::SETUP_FOR_MATCH);
  assert(paramss.size() > 0);
  int numBots = (int)paramss.size();

  //Figure out all pairs of bots that will be playing.
  std::vector<std::pair<int,int>> matchupsPerRound;
  {
    //Load a filter on what bots we actually want to run. By default, include everything.
    vector<bool> includeBot(numBots);
    if(cfg.contains("includeBots")) {
      vector<int> includeBotIdxs = cfg.getInts("includeBots",0,Setup::MAX_BOT_PARAMS_FROM_CFG);
      for(int i = 0; i<numBots; i++) {
        if(contains(includeBotIdxs,i))
          includeBot[i] = true;
      }
    }
    else {
      for(int i = 0; i<numBots; i++) {
        includeBot[i] = true;
      }
    }

    std::vector<int> secondaryBotIdxs;
    if(cfg.contains("secondaryBots"))
      secondaryBotIdxs = cfg.getInts("secondaryBots",0,Setup::MAX_BOT_PARAMS_FROM_CFG);
    for(int i = 0; i<secondaryBotIdxs.size(); i++)
      if(secondaryBotIdxs[i] < 0 || secondaryBotIdxs[i] >= numBots)
        throw StringError("secondaryBots value " + Global::intToString(secondaryBotIdxs[i]) + " is out of range, numBots is " + Global::intToString(numBots));

    for(int i = 0; i<numBots; i++) {
      if(!includeBot[i])
        continue;
      for(int j = 0; j<numBots; j++) {
        if(!includeBot[j])
          continue;
        if(i < j && !(contains(secondaryBotIdxs,i) && contains(secondaryBotIdxs,j))) {
          matchupsPerRound.emplace_back(i,j);
          matchupsPerRound.emplace_back(j,i);
        }
      }
    }

    if(cfg.contains("extraPairs")) {
      std::vector<std::pair<int,int>> pairs = cfg.getNonNegativeIntDashedPairs("extraPairs",0,numBots-1);
      for(const std::pair<int,int>& pair: pairs) {
        int p0 = pair.first;
        int p1 = pair.second;
        if(cfg.contains("extraPairsAreOneSidedBW") && cfg.getBool("extraPairsAreOneSidedBW")) {
          matchupsPerRound.emplace_back(p0,p1);
        }
        else {
          matchupsPerRound.emplace_back(p0,p1);
          matchupsPerRound.emplace_back(p1,p0);
        }
      }
    }
  }

  //Load the names of the bots and which model each bot is using
  vector<string> nnModelFilesByBot(numBots);
  vector<string> botNames(numBots);
  for(int i = 0; i<numBots; i++) {
    string idxStr = Global::intToString(i);

    if(cfg.contains("botName"+idxStr))
      botNames[i] = cfg.getString("botName"+idxStr);
    else if(numBots == 1)
      botNames[i] = cfg.getString("botName");
    else
      throw StringError("If more than one bot, must specify botName0, botName1,... individually");

    if(cfg.contains("nnModelFile"+idxStr))
      nnModelFilesByBot[i] = cfg.getString("nnModelFile"+idxStr);
    else
      nnModelFilesByBot[i] = cfg.getString("nnModelFile");
  }

  vector<bool> botIsUsed(numBots);
  for(const std::pair<int,int>& pair : matchupsPerRound) {
    botIsUsed[pair.first] = true;
    botIsUsed[pair.second] = true;
  }

  //Dedup and load each necessary model exactly once
  vector<string> nnModelFiles;
  vector<int> whichNNModel(numBots);
  for(int i = 0; i<numBots; i++) {
    if(!botIsUsed[i])
      continue;

    const string& desiredFile = nnModelFilesByBot[i];
    int alreadyFoundIdx = -1;
    for(int j = 0; j<nnModelFiles.size(); j++) {
      if(nnModelFiles[j] == desiredFile) {
        alreadyFoundIdx = j;
        break;
      }
    }
    if(alreadyFoundIdx != -1)
      whichNNModel[i] = alreadyFoundIdx;
    else {
      whichNNModel[i] = (int)nnModelFiles.size();
      nnModelFiles.push_back(desiredFile);
    }
  }

  //Load match runner settings
  int numGameThreads = cfg.getInt("numGameThreads",1,16384);
  const string gameSeedBase = Global::uint64ToHexString(seedRand.nextUInt64());

  //Work out an upper bound on how many concurrent nneval requests we could end up making.
  int expectedConcurrentEvals;
  {
    //Work out the max threads any one bot uses
    int maxBotThreads = 0;
    for(int i = 0; i<numBots; i++)
      if(paramss[i].numThreads > maxBotThreads)
        maxBotThreads = paramss[i].numThreads;
    //Mutiply by the number of concurrent games we could have
    expectedConcurrentEvals = maxBotThreads * numGameThreads;
  }

  //Initialize object for randomizing game settings and running games
  PlaySettings playSettings = PlaySettings::loadForMatch(cfg);
  GameRunner* gameRunner = new GameRunner(cfg, playSettings, logger);
  const int minBoardXSizeUsed = gameRunner->getGameInitializer()->getMinBoardXSize();
  const int minBoardYSizeUsed = gameRunner->getGameInitializer()->getMinBoardYSize();
  const int maxBoardXSizeUsed = gameRunner->getGameInitializer()->getMaxBoardXSize();
  const int maxBoardYSizeUsed = gameRunner->getGameInitializer()->getMaxBoardYSize();

  //Initialize neural net inference engine globals, and load models
  Setup::initializeSession(cfg);
  const vector<string>& nnModelNames = nnModelFiles;
  const int defaultMaxBatchSize = -1;
  const bool defaultRequireExactNNLen = minBoardXSizeUsed == maxBoardXSizeUsed && minBoardYSizeUsed == maxBoardYSizeUsed;
  const bool disableFP16 = false;
  const vector<string> expectedSha256s;
  vector<NNEvaluator*> nnEvals = Setup::initializeNNEvaluators(
    nnModelNames,nnModelFiles,expectedSha256s,cfg,logger,seedRand,expectedConcurrentEvals,
    maxBoardXSizeUsed,maxBoardYSizeUsed,defaultMaxBatchSize,defaultRequireExactNNLen,disableFP16,
    Setup::SETUP_FOR_MATCH
  );
  logger.write("Loaded neural net");

  vector<NNEvaluator*> nnEvalsByBot(numBots);
  for(int i = 0; i<numBots; i++) {
    if(!botIsUsed[i])
      continue;
    nnEvalsByBot[i] = nnEvals[whichNNModel[i]];
  }

  std::vector<std::unique_ptr<PatternBonusTable>> patternBonusTables = Setup::loadAvoidSgfPatternBonusTables(cfg,logger);
  testAssert(patternBonusTables.size() == numBots);

  //Initialize object for randomly pairing bots
  int64_t numGamesTotal = cfg.getInt64("numGamesTotal",1,((int64_t)1) << 62);
  MatchPairer* matchPairer = new MatchPairer(cfg,numBots,botNames,nnEvalsByBot,paramss,matchupsPerRound,numGamesTotal);

  MatchEloConfig matchEloConfig;
  matchEloConfig.enabled = cfg.contains("matchEloReport") ? cfg.getBool("matchEloReport") : false;
  matchEloConfig.reportPeriodGames = cfg.contains("matchEloReportPeriodGames") ? cfg.getInt64("matchEloReportPeriodGames",1,((int64_t)1) << 50) : cfg.getInt64("logGamesEvery",1,1000000);
  matchEloConfig.baselineBotIdx = cfg.contains("matchEloBaselineBot") ? cfg.getInt("matchEloBaselineBot",0,numBots-1) : 0;
  matchEloConfig.candidateBotIdx = cfg.contains("matchEloCandidateBot") ? cfg.getInt("matchEloCandidateBot",0,numBots-1) : (numBots > 1 ? 1 : 0);
  matchEloConfig.priorGames = cfg.contains("matchEloPriorGames") ? cfg.getDouble("matchEloPriorGames",0.0,1000000.0) : 1.0;
  matchEloConfig.sprtEnabled = cfg.contains("matchEloSprtEnabled") ? cfg.getBool("matchEloSprtEnabled") : false;
  matchEloConfig.sprtElo0 = cfg.contains("matchEloSprtElo0") ? cfg.getDouble("matchEloSprtElo0",-10000.0,10000.0) : 0.0;
  matchEloConfig.sprtElo1 = cfg.contains("matchEloSprtElo1") ? cfg.getDouble("matchEloSprtElo1",-10000.0,10000.0) : 5.0;
  matchEloConfig.sprtAlpha = cfg.contains("matchEloSprtAlpha") ? cfg.getDouble("matchEloSprtAlpha",1e-20,1.0-1e-20) : 0.05;
  matchEloConfig.sprtBeta = cfg.contains("matchEloSprtBeta") ? cfg.getDouble("matchEloSprtBeta",1e-20,1.0-1e-20) : 0.05;
  if(matchEloConfig.enabled && numBots < 2)
    throw StringError("matchEloReport requires numBots >= 2");
  if(matchEloConfig.sprtEnabled && matchEloConfig.sprtElo0 == matchEloConfig.sprtElo1)
    throw StringError("matchEloSprtElo0 and matchEloSprtElo1 must differ");

  //Check for unused config keys
  cfg.warnUnusedKeys(cerr,&logger);
  for(int i = 0; i<numBots; i++) {
    if(!botIsUsed[i])
      continue;
    Setup::maybeWarnHumanSLParams(paramss[i],nnEvalsByBot[i],NULL,cerr,&logger);
  }

  //Done loading!
  //------------------------------------------------------------------------------------
  logger.write("Loaded all config stuff, starting matches");
  if(!logger.isLoggingToStdout())
    cout << "Loaded all config stuff, starting matches" << endl;

  if(sgfOutputDir != string())
    MakeDir::make(sgfOutputDir);

  if(!std::atomic_is_lock_free(&shouldStop))
    throw StringError("shouldStop is not lock free, signal-quitting mechanism for terminating matches will NOT work!");
  std::signal(SIGINT, signalHandler);
  std::signal(SIGTERM, signalHandler);


  std::mutex statsMutex;
  int64_t gameCount = 0;
  std::map<string,double> timeUsedByBotMap;
  std::map<string,double> movesByBotMap;
  MatchEloTracker matchEloTracker(botNames,matchEloConfig);
  int64_t nextMatchEloReportGames = matchEloConfig.reportPeriodGames;
  int64_t lastMatchEloReportGames = 0;

  auto runMatchLoop = [
    &gameRunner,&matchPairer,&sgfOutputDir,&logger,&gameSeedBase,&patternBonusTables,
    &statsMutex, &gameCount, &timeUsedByBotMap, &movesByBotMap, &matchEloConfig, &matchEloTracker, &nextMatchEloReportGames, &lastMatchEloReportGames
  ](
    uint64_t threadHash
  ) {
    ofstream* sgfOut = NULL;
    if(sgfOutputDir.length() > 0) {
      sgfOut = new ofstream();
      FileUtils::open(*sgfOut, sgfOutputDir + "/" + Global::uint64ToHexString(threadHash) + ".sgfs");
    }
    auto shouldStopFunc = []() noexcept {
      return shouldStop.load();
    };
    WaitableFlag* shouldPause = nullptr;

    Rand thisLoopSeedRand;
    while(true) {
      if(shouldStop.load())
        break;

      FinishedGameData* gameData = NULL;

      MatchPairer::BotSpec botSpecB;
      MatchPairer::BotSpec botSpecW;
      if(matchPairer->getMatchup(botSpecB, botSpecW, logger)) {
        string seed = gameSeedBase + ":" + Global::uint64ToHexString(thisLoopSeedRand.nextUInt64());
        std::function<void(const MatchPairer::BotSpec&, Search*)> afterInitialization = [&patternBonusTables](const MatchPairer::BotSpec& spec, Search* search) {
          assert(spec.botIdx < patternBonusTables.size());
          search->setCopyOfExternalPatternBonusTable(patternBonusTables[spec.botIdx]);
        };
        gameData = gameRunner->runGame(
          seed, botSpecB, botSpecW, NULL, NULL, logger,
          shouldStopFunc, shouldPause, nullptr, afterInitialization, nullptr
        );
      }

      bool shouldContinue = gameData != NULL;
      if(gameData != NULL) {
        if(sgfOut != NULL) {
          WriteSgf::writeSgf(*sgfOut,gameData->bName,gameData->wName,gameData->endHist,gameData,false,true);
          (*sgfOut) << endl;
        }

        {
          std::lock_guard<std::mutex> lock(statsMutex);
          gameCount += 1;
          timeUsedByBotMap[gameData->bName] += gameData->bTimeUsed;
          timeUsedByBotMap[gameData->wName] += gameData->wTimeUsed;
          movesByBotMap[gameData->bName] += (double)gameData->bMoveCount;
          movesByBotMap[gameData->wName] += (double)gameData->wMoveCount;

          if(matchEloConfig.enabled) {
            matchEloTracker.addGame(*gameData);
            int64_t scoredGames = matchEloTracker.getScoredGames();
            if(scoredGames >= nextMatchEloReportGames) {
              logger.write(matchEloTracker.report(timeUsedByBotMap,movesByBotMap));
              lastMatchEloReportGames = scoredGames;
              do {
                nextMatchEloReportGames += matchEloConfig.reportPeriodGames;
              } while(scoredGames >= nextMatchEloReportGames);
            }
          }

          int64_t x = gameCount;
          while(x % 2 == 0 && x > 1) x /= 2;
          if(x == 1 || x == 3 || x == 5) {
            for(auto& pair : timeUsedByBotMap) {
              logger.write(
                "Avg move time used by " + pair.first + " " +
                Global::doubleToString(pair.second / movesByBotMap[pair.first]) + " " +
                Global::doubleToString(movesByBotMap[pair.first]) + " moves"
              );
            }
          }
        }

        delete gameData;
      }

      if(shouldStop.load())
        break;
      if(!shouldContinue)
        break;
    }
    if(sgfOut != NULL) {
      sgfOut->close();
      delete sgfOut;
    }
    logger.write("Match loop thread terminating");
  };
  auto runMatchLoopProtected = [&logger,&runMatchLoop](uint64_t threadHash) {
    Logger::logThreadUncaught("match loop", &logger, [&](){ runMatchLoop(threadHash); });
  };


  Rand hashRand;
  vector<std::thread> threads;
  threads.reserve(numGameThreads);
  for(int i = 0; i<numGameThreads; i++) {
    threads.emplace_back(runMatchLoopProtected, hashRand.nextUInt64());
  }
  for(int i = 0; i<threads.size(); i++)
    threads[i].join();

  if(matchEloConfig.enabled) {
    std::lock_guard<std::mutex> lock(statsMutex);
    if(matchEloTracker.getScoredGames() != lastMatchEloReportGames)
      logger.write(matchEloTracker.report(timeUsedByBotMap,movesByBotMap));
  }

  delete matchPairer;
  delete gameRunner;

  nnEvalsByBot.clear();
  for(int i = 0; i<nnEvals.size(); i++) {
    if(nnEvals[i] != NULL) {
      logger.write(nnEvals[i]->getModelFileName());
      logger.write("NN rows: " + Global::int64ToString(nnEvals[i]->numRowsProcessed()));
      logger.write("NN batches: " + Global::int64ToString(nnEvals[i]->numBatchesProcessed()));
      logger.write("NN avg batch size: " + Global::doubleToString(nnEvals[i]->averageProcessedBatchSize()));
      delete nnEvals[i];
    }
  }
  NeuralNet::globalCleanup();
  ScoreValue::freeTables();

  if(sigReceived.load())
    logger.write("Exited cleanly after signal");
  logger.write("All cleaned up, quitting");
  return 0;
}
