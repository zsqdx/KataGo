#ifndef SEARCH_SEARCHNODETABLE_H
#define SEARCH_SEARCHNODETABLE_H

#include "../core/global.h"
#include "../core/hash.h"
#include "../core/multithread.h"
#include "../game/board.h"
#include "../search/mutexpool.h"

#include <unordered_map>

struct SearchNode;

struct SearchNodeTableHash {
  size_t operator()(Hash128 hash) const;
};

struct SearchNodeTable {
  std::vector<std::unordered_map<Hash128,SearchNode*,SearchNodeTableHash>> entries;
  MutexPool* mutexPool;
  uint32_t numShards;

  SearchNodeTable(int numShardsPowerOfTwo);
  ~SearchNodeTable();

  uint32_t getIndex(uint64_t hash) const;
};

#endif


