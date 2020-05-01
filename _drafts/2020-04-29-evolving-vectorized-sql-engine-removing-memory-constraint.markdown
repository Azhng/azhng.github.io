---
layout: post
title:  "Evolving Vectorized SQL Engine: Efficient Unordered Distinct"
date:   2020-04-29 23:04:45 -0400
---

In the upcoming CockroachDB 20.1 release, we will finally be removing the experimental flag for
our new Vectorized SQL Execution Engine. Currently, CockroachDB will run the query through the
new vectorized engine automatically using a set of heuristics, such as the size of the returning
result and estimated memory consumption.  Users can also use vectorized engine to execute all queries
using the following command in their SQL shell.

```
set vectorize=on;
```

The vectorized execution engine is an interesting piece of technology that we have be building to
significantly speed up SQL queries. In our early benchmarks, the new vectorized engine can be up to
[40x][1] faster than our existing row-at-a-time SQL engine. You can read more about how we built the
vectorized engine more in our previous [blog posts][2].

The new vectorized engine enabled us to build SQL operators that are friendly to modern CPU caching architecture
and branch predictor. These are the keys to how vectorized engine can greatly outperform our
existing row-at-a time engine. However, these improvement does come at a cost. Columnar memory layout
is great when it comes to speeding up the operations, but sometimes it makes it difficult to implement complex
operations such as SQL joins or groupings. Also we need to be careful with our implementations that we don't
take any shortcut, which can cause the vectorized engine to essentially perform like the old row-at-a-time
engine.

When first rolled out the new vectorized engine in CockroachDB 19.2 release. It was considered an experimental
feature at the time due to one key issues: memory constraints. Although most of the SQL operators in vectorized
engine performed streaming operations, some does require buffering inputs in memory. The vectorized engine at
the time did not have any ability to fall back to disk when the buffering operator runs out of memory. This is
a significant road block that prevented us from fully turn on the vectorized engine and limiting its use case
to the ones that can fit inside database the memory.

In order to make vectorized engine suitable for all use cases, the SQL execution team at Cockroach Labs laid out plans
to address the memory constraint in [this RFC][3] later last year (Note: this RFC is slightly out of date since when it
was first published, but the general idea remains the same). The idea was essentially to use a combination
of disk spilling infrastructure and algorithmic improvement to reduce memory buffering, and fall back to disk when
necessary. As of CockroachDB 20.1, we are happy to say that we believe vectorized engine is now suitable for
production usage.

For past four months, I was working as a backend engneering intern with incredibly talented engineers from the
SQL Execution team to unlock the full potential of vectorized engine for the upcoming CockroachDB 20.1 release.
During this time, I focused on improving the algorithm used in our buffering SQL operators to reduce memory
usage and speed up execution. In this blog post, I will be going over the improvement I made to the algorithm
that's behind unordered distinct operator and how the new algorithms address the deficiency of the existing
algorithms.

Note that it is strongly encourage to read our previous [blog post][1] before proceed with this blog post.

# Unordered Distinct

Unordered Distinct is a SQL operator that implements the SQL `SELECT DISTINCT` query. This operator removes the duplicated
tuples from the query result. As its name suggested, it is designed to work with unordered inputs. Its counter part, the
ordered distinct, operator is designed to work with ordered data set. The main difference is that ordered distinct operates
as a streaming operator, which means it consumes a constant amount of memory as it processes through all input tuples. This
is made possible due to the ordered characteristic of the input.

For unordered distinct operator, it has to keep track of all distinct tuples that it has come across throughout
its lifetime. Naively, the operator can store all unique tuples inside an in-memory key-value dictionary. For
all subsequent tuples it encounters, the operator only needs to perform a simple dictionary lookup in the dictionary
to determine if this tuple is unique or not. However, the naive implementation of this algorithm causes the vectorized
engine to behave essentially same as a row-at-a-time engine. Since within each iteration of the loop, the operator needs
to perform multiple steps of computation: 

1. Hash the key columns of the input tuples.
1. Dictionary lookup using the tuple's hash.
1. If the same hash key exist, perform actual comparison between the new tuple and the tuple with the same hash key in
   the dictionary to check if it is hash collision.
1. Perform dictionary insertion if the new tuple is actually unique.

Observe the computation steps within each iteration of the loop. It involves a hash computation, multiple branching
and value comparisons (which can be potentially very expensive depending on the SQL data types) and insertion (which
can potentially lead to memory allocation and copying). The point of vectorized engine is to reduce one big complicated
loop into multiple small and tight loop with very simple operation inside the loop body. This can improve the program's
cache friendliness and reduce branch mispredict inside CPU.

Therefore, as of CockroachDB 19.2, the vectorized unordered distinct operator was implemented using our own vectorized
hash table. It was implemented based on Marcin Zukowski's paper: "[Balacing Vectorized Query Execution iwth Bandwith-Optimized
Storage][4]". This data structure was the key to how we initially implemented our efficient vectorized hash joiner.
Definitely check out our previous [blog post][1] if you are interested in how does the vectorized hash table work.

However, since the original vectorized hash table was designed for hash join operator, it stored a lot of extra information
that was not necessary for unordered distinct operator. For example, in order to implement hash join, the hash table not
only need to store the distinct tuples, it also needs to store all tuples that are identical to that tuple. It also need
to compute data structures to efficiently traverse through all those identical tuples. But in unordered distinct
operator, none of those auxiliary data structure and computation is required, since the only thing we care about is
the first distinct tuple we encounter, then we can discard all subsequent tuples that are identical to the first one.
And this is the key insight of how we improve the performance of unordered distinct operator.

# Vectorized Hash Table

The existing hash table separates its construction into multiple stage. In the first stage, it consumes ***all*** of its
input entirely and buffer them all in memory. Then it hashes each tuple in the buffer, and based on the value of the hash,
the hash table constructs a linked list for each hash value to chain together all tuples that are hashed to the same key.
For the rest of this blog post, I will refer to these linked lists as hash chain. The simplified code snippet below
desmonstrates how we construct the hash chains:

``` golang
/*
 * ht.vals:       all buffered tuple.
 * ht.hashBuffer: stores the mapping of index of tuple in ht.vals -> hash value.
 * ht.first:      a previously allocated slice of []uint64 that stores mapping of hashCode -> index
 *                into ht.next.
 * ht.next:       stores the index of the next tuple that has the same hash in the linked list, if it
 *                is the end
 *                of the linked list, the value is 0.
 */
func (ht *hashTable) buildNextChains() {
	// Note that we use the convention where if the entry is the end of the linked
	// list, it has value of 0. This is the reason why ht.next[0] is not used.
	for id := 1; id <= ht.vals.Length(); id++ {
		hash := ht.hashBuffer[id]
		ht.next[id] = ht.first[hash]
		ht.first[hash] = uint64(id)
	}
}
```

Note that since we use the convention where `0` indicates the end of the linked list, we cannot use the index
of the tuple directly since it can creates a entry in the `ht.next` of value `0`. Therefore, we create the mapping
of `keyID = tuple index + 1` to avoid this confusion.

The algorithm shown above is simplified for better readability. In reality, we would use the `ht.next` buffer
to store `keyID` to `hash value` mapping initially, then reuse it later to store the `next` chain of the linked
list in order to avoid extra memory allocation. However, this is strictly for the purpose of performance
optimization and does not affect the algorithm correctness.

To better illustrate the algorithm, suppose we have the following input and its corresponding hash values as the
result of hashing each tuple in the input:

```
input: +---+----+-----+    hash buffer: +-------+------+
       | a |  b |  c  |                 | keyID | hash |
       +---+----+-----+                 +-------+------+
       | 2 |null|"2.0"|                 |   0   |  0   |  <- not used, reserved
       +---+----+-----+                 +-------+------+
       | 2 |1.0 |"1.0"|                 |   1   |  1   |
       +---+----+-----+                 +-------+------+
       | 2 |null|"2.0"|                 |   2   |  0   |  <- hash collision \
       +---+----+-----+                 +-------+------+                     *
       | 2 |2.0 |"2.0"|                 |   3   |  0   |  <- hash collision  |
       +---+----+-----+                 +-------+------+                     *
       | 2 |null|"2.0"|                 |   4   |  0   |  <- hash collision /
       +---+----+-----+                 +-------+------+
       | 2 |1.0 |"1.0"|                 |   5   |  1   |
       +---+----+-----+                 +-------+------+
```

After we constructs `ht.next` and `ht.first` using the algorithm from above, we get following slices:

```
first: +------+-------+  next: +-------+------+
       | hash | keyID |        | keyID | next |
       +------+-------+        +-------+------+
       |  0   |   5   |        |   0   | N/A  |  <- not use, reserved
       +------+-------+        +-------+------+
       |  1   |   6   |        |   1   |  0   |  <- tail of the linked list of hash value 0
       +------+-------+        +-------+------+
                               |   2   |  0   |  <- tail of the linked list of hash value 1
                               +-------+------+
                               |   3   |  1   |
                               +-------+------+
                               |   4   |  3   |
                               +-------+------+
                               |   5   |  4   |  <- head of the linked list of hash value 0
                               +-------+------+
                               |   6   |  2   |  <- head of the linked list of hash value 1
                               +-------+------+
```

Note that in the diagram, it shows that `ht.first` only ontains two entries. However in reality, hash
value can be any arbitrary number, therefore `ht.first` was initially allocated to be as large as the
entirety of the hash range to accommodate that. (Currently, our hash range is 2^16)

A key observations here is that: For each hash chain, the values of the `keyID`s within the it are 
monotonically decreasing. This property is not a very big deal for hash joiner, however this can caused
some trouble for unordered distinct. This is a very important property of this algorithm and we
will come back to this later.

The next step of the algorithm is to probe through each hash chain we have just built, and resolve any hash
collisions. This is a rather quite complicated step which is explored in depth in our previous [blog post][1].
For simplicity, I'm going to go over this step from a high level.

In this step, we compute a `head` boolean slice which represents a mapping from each keyID to whether if this keyID
is the head of the linked list which contains keyIDs of all tuples that are identical. Note that this is
different from the hash chain we mentioned previously. In hash chains, we use linked lists to point to all
tuples that has the same hash value. Now, the linked list contains all tuples whose value are actually equal.
This is some crucial information for the hash joiner, however it is completely useless for unordered distinct.

This is the first thing we can optimize the hash table. Once we found the head of the linked list, we can skip
building out the linked list. This simple optimization actually yields about 35 ~ 40% of increase of performance
for the unordered distinct operator.

```
name                                                      old speed      new speed     
UnorderedDistinct/numCols=1/nulls=false/numBatches=1-8     128MB/s ± 0%    181MB/s ± 0%
UnorderedDistinct/numCols=1/nulls=false/numBatches=16-8    107MB/s ± 0%    145MB/s ± 0%
UnorderedDistinct/numCols=1/nulls=false/numBatches=256-8  45.1MB/s ± 0%   82.6MB/s ± 0%
UnorderedDistinct/numCols=1/nulls=true/numBatches=1-8     71.5MB/s ± 0%  112.1MB/s ± 0%
UnorderedDistinct/numCols=1/nulls=true/numBatches=16-8    65.6MB/s ± 0%   98.9MB/s ± 0%
UnorderedDistinct/numCols=1/nulls=true/numBatches=256-8   22.8MB/s ± 0%   41.5MB/s ± 0%
UnorderedDistinct/numCols=2/nulls=false/numBatches=16-8    146MB/s ± 0%    190MB/s ± 0%
UnorderedDistinct/numCols=2/nulls=false/numBatches=256-8  50.5MB/s ± 0%  103.1MB/s ± 0%
UnorderedDistinct/numCols=2/nulls=true/numBatches=1-8     97.2MB/s ± 0%  131.6MB/s ± 0%
UnorderedDistinct/numCols=2/nulls=true/numBatches=16-8    82.0MB/s ± 0%  111.2MB/s ± 0%
```

Can we do even better than this? Of course!

Another key observation here is that in the initial stage of building the vectorized hash table, we buffer
all tuples into memory first, then afterwards we proceed to perform subsequent computations. As we have
mentioned previously, this was designed so that we will have all the information we needed for the hash
joiner. However, in the case of unordered distinct, we only need the distinct tuples. This is the key insight
into our second optimization: we only need to only store the distinct tuples in the hash table!

This optimization strategy also limits the memory consumption of unordered distinct to proportional to the number
of _distinct_ tuples in the input instead of the _total_ number of tuples in the input. Also since we only buffer
distinct tuples into memory, we reduce total number of comparisons we perform throughout the entire algorithm.

# Vectorized Hash Table Distinct Mode

In order to fully implement the distinct mode into vectorized hash table, we need to modify the algorithm
of building the hash table quite a bit. First, since the main goal is to eliminate the need to buffer the
entire input into memory, we can no longer consume the entire input and then buffer them in memory in the
beginning of the algorithm. Instead, we would need to build the hash table incrementally. The intuition is
quite simple, when we fetched a new batch of tuples, we first perform de-duplication within that batch. Once
all duplicated tuples are removed from the batch, we check the remaining tuples within that batch against
the already buffered tuples to see if there is any additional duplicated tuples. Note that since we only
insert batches that only contain unique tuples, we can be sure that there does not exist any duplicated tuples
within the buffered batch. Therefore, we can be sure that once the second duplication check is completed,
we can simply append all tuples remained in the new batch directly into the buffered batch, and the uniqueness
invariant will still hold.

To put this idea into code, we essentially have this:

``` golang
func (ht *hashTable) build() {
	for {
		batch := input.Next(ctx)
		if batch.Length() == 0 {
			break
		}

		/*
		 * Handling building hash chains.
		 * ...
		 */

		// We remove duplicated tuples that exist within this batch.
		batch.removeDuplicatedTuples(batch)

		numBuffered := ht.bufferedTuples.Length()
		// We only check duplicates when there is at least one buffered
		// tuple.
		if numBuffered > 0 {
    // We remove duplicated tuples that exist in already buffered tuples.
			batch.removeDuplicatedTuples(ht.bufferedTuples)
		}

		ht.bufferedTuples.Append(batch)
	}
}
```

Note that the code has been greatly simplified to illustrate the idea of the algorithm, here is the [link][5]
to the full implementation.

Now, if you are familiar with the hash collision handling scheme we implemented in our previous [blog post][1],
you will realize that the process of de-duplication and checking for hash collision is quite similar. In
fact, if you squint hard enough at the code, they are almost identical with slight modifications.

There are two phases in how we perform de-duplication, in the first phase, we remove duplicated tuples within
the batch itself. In this phase, for each tuple, we traverse through the hash chain that has been built previously,
it stops the linked list traversal as soon as it founds the first identical tuple in the hash chain, and that
first tuple will be our choice of the distinct tuple. The algorithm itself is quite straight forward, however,
if you are paying close attention, this optimiztion cannot completely guarantee the correctness of the algorithm.

Remeber that we mentioned previously, once we build the hash chain, the values of the `keyID`s inside the hash
chain will be in a monotonically decreasing order. In other words, this means the "head" of the hash chain is going
the be the tuple with largest `keyID`. Now also observe that with our de-duplication algorithm, we immediately
stop hash chain traversal once we encounter a tuple that has the identical value, and that tuple will be emitted
as the distinct tuple. Since the ordered of the hash chain is reversed, this means the `head`, a.k.a. the last
occurrence of the tuple inside a batch will be the one that gets emitted. So concretly, recall our previous example: 

```
first: +------+-------+  next: +-------+------+
       | hash | keyID |        | keyID | next |
       +------+-------+        +-------+------+
       |  0   |   5   |        |   0   | N/A  | <- not use, reserved
       +------+-------+        +-------+------+
       |  1   |   6   |        |   1   |  0   |  <- tail of the linked list of hash value 0
       +------+-------+        +-------+------+
                               |   2   |  0   |  <- tail of the linked list of hash value 1
                               +-------+------+
                               |   3   |  1   |
                               +-------+------+
                               |   4   |  3   |
                               +-------+------+
                               |   5   |  4   |  <- head of the linked list of hash value 0
                               +-------+------+
                               |   6   |  2   |  <- head of the linked list of hash value 1
                               +-------+------+
```

In this case, since we are starting to probe from `keyID=5` and `keyID=6`, we will eventually emit the following
`keyID`s: 5, 6 and 4. This seemingly to be compliant with the SQL standard, since the SQL standard does not require
the `SELECT DISTINCT` query to emit resulting tuple in any order without the `ORDERD BY` clause. However, inside
of CockroachDB's SQL engine, different SQL operators can be used as building blocks for more complicated
queries. Over the course of many releases, all our previous unordered distinct operators maintained this implicit
behaviour of emitting the first distinct tuple it encounters in the input, instead of the last tuple as we have
seen in our optimized algorithm. This behaviour is being depended on by CockroachDB's query planner when it decides
to use unordered distinct operator as the input for other SQL operators. Therefore, in order to maintain
compatibility with this assumption, we need to make sure that hash table emits the first distinct tuple it encounters,
instead of the last.

The solution towards to this problem is essentially reverse the ordered of the linked list. Since the de-duplication
process starts with the head of the hash chain, if we can ensure the value of `keyID`s within the hash chain is in
monotonically increasing order, we can be sure that the hash table will emit the first tuple it encountered. As the
result, we updated the `buildNextChains` function to the following:


``` golang
/*
 * Initially, all ht.first are set to 0.
 *
 * batch:         the batch of tuples that we need to build hash chain for.
 * ht.hashBuffer: stores the mapping of index of tuple in ht.vals -> hash value.
 * ht.first:      a previously allocated slice of []uint64 that stores mapping of hashCode -> index
 *                into ht.next.
 * ht.next:       stores the index of the next tuple that has the same hash in the linked list, if it
 *                is the end
 *                of the linked list, the value is 0.
 */
func (ht *hashTable) buildNextChains(batch coldata.Batch) {
	// Note that we use the convention where if the entry is the end of the linked
	// list, it has value of 0. This is the reason why ht.next[0] is not used.
	for id := batch.Length(); id >= 1; id-- {
		hash := ht.hashBuffer[id]
		firstKeyID := ht.first[id]
		if firstKeyID == 0 || id < firstKeyID {
			next[id] = first[hash]
			first[hash] = id
		} else {
			next[id] = next[firstKeyID]
			next[firstKeyID] = id
		}
	}
}
```

Here is the benchmark for the unordered distinct operator with updated algorithm:

```
name                                                                                        old speed      new speed      delta
Distinct/Unordered/hasNulls=false/newTupleProbability=0.001/rows=4096/cols=2/ordCols=0-24    154MB/s ± 5%   156MB/s ± 3%      ~     (p=0.460 n=10+8)
Distinct/Unordered/hasNulls=false/newTupleProbability=0.001/rows=4096/cols=4/ordCols=0-24    219MB/s ± 3%   236MB/s ±13%      ~     (p=0.143 n=10+10)
Distinct/Unordered/hasNulls=false/newTupleProbability=0.001/rows=65536/cols=2/ordCols=0-24   136MB/s ± 3%   384MB/s ± 2%  +181.65%  (p=0.000 n=10+10)
Distinct/Unordered/hasNulls=false/newTupleProbability=0.001/rows=65536/cols=4/ordCols=0-24   245MB/s ± 1%   518MB/s ± 2%  +111.55%  (p=0.000 n=9+10)
Distinct/Unordered/hasNulls=false/newTupleProbability=0.010/rows=4096/cols=2/ordCols=0-24    154MB/s ± 4%   155MB/s ± 3%      ~     (p=0.754 n=10+10)
Distinct/Unordered/hasNulls=false/newTupleProbability=0.010/rows=4096/cols=4/ordCols=0-24    214MB/s ± 4%   249MB/s ± 2%   +16.29%  (p=0.000 n=10+9)
Distinct/Unordered/hasNulls=false/newTupleProbability=0.010/rows=65536/cols=2/ordCols=0-24   181MB/s ± 3%   373MB/s ± 2%  +106.24%  (p=0.000 n=10+10)
Distinct/Unordered/hasNulls=false/newTupleProbability=0.010/rows=65536/cols=4/ordCols=0-24   225MB/s ± 6%   504MB/s ± 2%  +123.53%  (p=0.000 n=10+10)
Distinct/Unordered/hasNulls=false/newTupleProbability=0.100/rows=4096/cols=2/ordCols=0-24    150MB/s ± 4%   147MB/s ± 4%      ~     (p=0.143 n=10+10)
Distinct/Unordered/hasNulls=false/newTupleProbability=0.100/rows=4096/cols=4/ordCols=0-24    207MB/s ± 6%   204MB/s ± 6%      ~     (p=0.182 n=10+9)
Distinct/Unordered/hasNulls=false/newTupleProbability=0.100/rows=65536/cols=2/ordCols=0-24   165MB/s ± 2%   301MB/s ± 4%   +82.26%  (p=0.000 n=10+10)
Distinct/Unordered/hasNulls=false/newTupleProbability=0.100/rows=65536/cols=4/ordCols=0-24   201MB/s ± 3%   398MB/s ± 2%   +97.83%  (p=0.000 n=10+9)
Distinct/Unordered/hasNulls=true/newTupleProbability=0.001/rows=4096/cols=2/ordCols=0-24     113MB/s ± 4%   122MB/s ± 4%    +8.02%  (p=0.000 n=10+10)
Distinct/Unordered/hasNulls=true/newTupleProbability=0.001/rows=4096/cols=4/ordCols=0-24     141MB/s ± 3%   168MB/s ± 2%   +19.43%  (p=0.000 n=10+10)
Distinct/Unordered/hasNulls=true/newTupleProbability=0.001/rows=65536/cols=2/ordCols=0-24    138MB/s ± 4%   236MB/s ± 2%   +71.44%  (p=0.000 n=10+10)
Distinct/Unordered/hasNulls=true/newTupleProbability=0.001/rows=65536/cols=4/ordCols=0-24    156MB/s ± 3%   282MB/s ± 1%   +80.65%  (p=0.000 n=10+10)
Distinct/Unordered/hasNulls=true/newTupleProbability=0.010/rows=4096/cols=2/ordCols=0-24     109MB/s ± 5%   117MB/s ± 1%    +6.74%  (p=0.000 n=10+9)
Distinct/Unordered/hasNulls=true/newTupleProbability=0.010/rows=4096/cols=4/ordCols=0-24     143MB/s ± 3%   164MB/s ± 2%   +14.59%  (p=0.000 n=10+10)
Distinct/Unordered/hasNulls=true/newTupleProbability=0.010/rows=65536/cols=2/ordCols=0-24    100MB/s ± 3%   226MB/s ± 2%  +125.29%  (p=0.000 n=10+10)
Distinct/Unordered/hasNulls=true/newTupleProbability=0.010/rows=65536/cols=4/ordCols=0-24    142MB/s ± 3%   272MB/s ± 3%   +91.53%  (p=0.000 n=10+10)
Distinct/Unordered/hasNulls=true/newTupleProbability=0.100/rows=4096/cols=2/ordCols=0-24     108MB/s ± 3%   112MB/s ± 3%    +3.43%  (p=0.001 n=10+10)
Distinct/Unordered/hasNulls=true/newTupleProbability=0.100/rows=4096/cols=4/ordCols=0-24     132MB/s ± 3%   150MB/s ± 8%   +13.62%  (p=0.000 n=10+10)
Distinct/Unordered/hasNulls=true/newTupleProbability=0.100/rows=65536/cols=2/ordCols=0-24    107MB/s ± 5%   191MB/s ± 4%   +78.89%  (p=0.000 n=10+10)
Distinct/Unordered/hasNulls=true/newTupleProbability=0.100/rows=65536/cols=4/ordCols=0-24    125MB/s ± 2%   237MB/s ± 4%   +89.61%  (p=0.000 n=10+10)
```

We can see that depending on the characteristics of the input data, we can have up to 181% of
performance increase on top of our previous performance boost of ~35% to ~40%. Also, we are seeing
a drastic drop of memory foot print of the unordered distinct operator: 


```
name                                                                        old alloc/op   new alloc/op   delta
Distinct/Unordered/newTupleProbability=0.001/rows=4096/cols=2/ordCols=0-8      932kB ± 0%    1164kB ± 0%   +24.97%  (p=0.000 n=9+10)
Distinct/Unordered/newTupleProbability=0.001/rows=4096/cols=4/ordCols=0-8     1.16MB ± 0%    1.20MB ± 0%    +3.01%  (p=0.000 n=9+8)
Distinct/Unordered/newTupleProbability=0.001/rows=65536/cols=2/ordCols=0-8    8.17MB ± 0%    1.17MB ± 0%   -85.68%  (p=0.000 n=10+8)
Distinct/Unordered/newTupleProbability=0.001/rows=65536/cols=4/ordCols=0-8    14.0MB ± 0%     1.2MB ± 0%   -91.34%  (p=0.000 n=8+9)
Distinct/Unordered/newTupleProbability=0.010/rows=4096/cols=2/ordCols=0-8      932kB ± 0%    1166kB ± 0%   +25.14%  (p=0.000 n=8+8)
Distinct/Unordered/newTupleProbability=0.010/rows=4096/cols=4/ordCols=0-8     1.16MB ± 0%    1.20MB ± 0%    +3.34%  (p=0.000 n=6+9)
Distinct/Unordered/newTupleProbability=0.010/rows=65536/cols=2/ordCols=0-8    8.17MB ± 0%    1.20MB ± 0%   -85.29%  (p=0.000 n=8+9)
Distinct/Unordered/newTupleProbability=0.010/rows=65536/cols=4/ordCols=0-8    14.0MB ± 0%     1.3MB ± 0%   -90.92%  (p=0.000 n=8+8)
Distinct/Unordered/newTupleProbability=0.100/rows=4096/cols=2/ordCols=0-8      932kB ± 0%    1181kB ± 0%   +26.70%  (p=0.000 n=8+9)
Distinct/Unordered/newTupleProbability=0.100/rows=4096/cols=4/ordCols=0-8     1.16MB ± 0%    1.23MB ± 0%    +5.84%  (p=0.000 n=7+8)
Distinct/Unordered/newTupleProbability=0.100/rows=65536/cols=2/ordCols=0-8    8.17MB ± 0%    1.77MB ± 0%   -78.27%  (p=0.000 n=8+10)
Distinct/Unordered/newTupleProbability=0.100/rows=65536/cols=4/ordCols=0-8    14.0MB ± 0%     2.2MB ± 0%   -83.99%  (p=0.000 n=9+10)
```

I hope you enjoyed learning how our vectorized hash table work under the hood and how we
continously evolving our algorithms and implementations. What I have presented in this blog
post is only a tip of the iceberg of what we have worked on for the vectorized engine that's
coming in CockroachDB 20.1 release. I am very grateful for the opportunity to work with the
absolute brilliant engineers at Cockroach Labs and I want to really thank my mentor Alfonso,
my manager Jordan and my teammate Yahor for this amazing internship. Thank you Cockroach Labs,
it has been a great journey.

# P.S.

// TODO(azhng): hiring plug?

[1]: https://www.cockroachlabs.com/blog/vectorized-hash-joiner
[2]: https://www.cockroachlabs.com/blog/how-we-built-a-vectorized-execution-engine/
[3]: https://github.com/cockroachdb/cockroach/blob/master/docs/RFCS/20191113_vectorized_external_storage.md
[4]: https://dare.uva.nl/search?identifier=5ccbb60a-38b8-4eeb-858a-e7735dd37487
[5]: https://github.com/cockroachdb/cockroach/blob/master/pkg/sql/colexec/hashtable.go
