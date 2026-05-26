#include "../search/searchnodetable.h"

#include "../core/rand.h"
#include "../search/localpattern.h"

SearchNodeTable::SearchNodeTable(int numShardsPowerOfTwo) {
  numShards = (uint32_t)1 << numShardsPowerOfTwo;
  mutexPool = new MutexPool(numShards);
  entries.resize(numShards);
}
SearchNodeTable::~SearchNodeTable() {
  delete mutexPool;
}

size_t SearchNodeTableHash::operator()(Hash128 hash) const {
  uint64_t mixed = Hash::rrmxmx(hash.hash0 ^ Hash::basicLCong2(hash.hash1));
  return (size_t)mixed;
}

uint32_t SearchNodeTable::getIndex(uint64_t hash) const {
  uint32_t mutexPoolMask = numShards-1; //Always a power of two
  return (uint32_t)(hash & mutexPoolMask);
}

