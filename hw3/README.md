## OpenMP Programming

### Part 1: Parallelizing Conjugate Gradient Method with OpenMP
Conjugate gradient method is an algorithm for the numerical solution of particular systems of linear equations. It is often used to solve partial differential equations, or applied on some optimization problems. You may get more information on [Wikipedia](http://en.wikipedia.org/wiki/Conjugate_gradient_method).

Please enter the `part1` folder:

```bash
$ cd part1
```
You can build and run the `CG` program by:

```bash
$ make; ./cg
```

In this assignment, you are asked to parallelize a serial implementation of the conjugate gradient method using OpenMP. The serial implementation is in `cg.c` and `cg_impl.c`. Please refer to the `README` file for more information about the code.

In order to parallelize the conjugate gradient method, you may want to use profiling tools to help you explain the performance difference before and after your modification. As it probably just works ineffectively if you simply add a parallel-for directive for each for loop, you may want to use profiling tools we mentioned in Programming Assignment ZERO to profile your program to better understand how much the improvement is from the code snippets you have modified.

### Part 2: Parallelizing PageRank Algorithm with OpenMP
![Page Rank](images/page_rank.png)

In this part, you will implement two graph processing algorithms: [breadth-first search](https://en.wikipedia.org/wiki/Breadth-first_search) (BFS) and a simple implementation of [page rank](https://en.wikipedia.org/wiki/PageRank). A good implementation of this assignment will be able to run these algorithms on graphs containing hundreds of millions of edges on a multi-core machine in only seconds.

Please enter the `part2` folder:

```bash
$ cd part2
```

#### 2.1 Background: Representing Graphs
The starter code operates on directed graphs, whose implementation you can find in `common/graph.h` and `common/graph_internal.h`. We recommend you begin by understanding the graph representation in these files. A graph is represented by an array of edges (both `outgoing_edges` and `incoming_edges`), where each edge is represented by an integer describing the id of the destination vertex. Edges are stored in the graph sorted by their source vertex, so the source vertex is implicit in the representation. This makes for a compact representation of the graph, and also allows it to be stored contiguously in memory. For example, to iterate over the outgoing edges for all nodes in the graph, you’d use the following code which makes use of convenient helper functions defined in `common/graph.h`(and implemented in `common/graph_internal.h`):

```cpp
for (int i=0; i<num_nodes(g); i++) {
    // Vertex is typedef'ed to an int. Vertex* points into g.outgoing_edges[]
    const Vertex* start = outgoing_begin(g, i);
    const Vertex* end = outgoing_end(g, i);
    for (const Vertex* v=start; v!=end; v++)
        printf("Edge %u %u\n", i, *v);
}
```

#### Task 1: Implementing Page Rank
As a simple warm up exercise to get comfortable using the graph data structures, and to get acquainted with a few OpenMP basics, we’d like you to begin by implementing a basic version of the well-known page rank algorithm.

Please take a look at the pseudocode provided to you in the function `pageRank()`, in the file `page_rank/page_rank.cpp`. You should implement the function, parallelizing the code with OpenMP. Just like any other algorithm, first identify independent work and any necessary synchronization.

You can run your code, checking correctness and performance against the staff reference solution using:

```bash
./pr <PATH_TO_GRAPH_DIRECTORY>/com-orkut_117m.graph
```

If you are working on our workstation machines, we’ve located a copy of the graph and some other graphs at /HW3/graphs/. You can also download the graphs from [here](http://sslab.cs.nctu.edu.tw/~acliu/all_graphs.tgz). 

Some interesting real-world graphs include:

- com-orkut_117m.graph
- oc-pokec_30m.graph
- rmat_200m.graph
- soc-livejournal1_68m.graph

Some useful synthetic, but large graphs include:

- random_500m.graph
- rmat_200m.graph

There are also some very small graphs for testing. If you look in the `/tools` directory of the starter code, you’ll notice a useful program called `graphTools.cpp` that can be used to make your own graphs as well.

By default, the `pr` program runs your page rank algorithm with an increasing number of threads (so you can assess speedup trends). However, since runtimes at low core counts can be long, you can explicitly specify the number of threads to only run your code under a single configuration.

```bash
./pr %GRAPH_FILENAME% 8
```

Your code should handle cases where there are no outgoing edges by distributing the probability mass on such vertices evenly among all the vertices in the graph. That is, your code should work as if there were edges from such a node to every node in the graph (including itself). The comments in the starter code describe how to handle this case.

You can also run our grader via: 

```bash
./pr_grader <PATH_TO_GRAPH_DIRECTORY>
```

which reports the correctness of your program and a performance score for four specific graphs.

You are highly recommended to read the paper, [“Direction-Optimizing Breadth-First Search”](https://parlab.eecs.berkeley.edu/sites/all/parlab/files/main.pdf), which proposed a hybrid method combining top-down and bottom-up algorithms for breadth-first search, before you continue the following sections. In Sections 2.3, 2.4, 2.5, you will need to finish a top-down, bottom-up, and hybrid method, respectively.

#### 2.3 Task 2: Parallel Breadth-First Search (“Top Down”)
Breadth-first search (BFS) is a common algorithm that you’ve almost certainly seen in a prior algorithms class.

Please familiarize yourself with the function `bfs_top_down()` in `breadth_first_search/bfs.cpp`, which contains a sequential implementation of BFS. The code uses BFS to compute the distance to vertex 0 for all vertices in the graph. You may wish to familiarize yourself with the graph structure defined in `common/graph.h` as well as the simple array data structure vertex_set (breadth_first_search/bfs.h), which is an array of vertices used to represent the current frontier of BFS.

You can run bfs using:
```bash
./bfs <PATH_TO_GRAPHS_DIRECTORY>/rmat_200m.graph
```
> as with page rank, bfs’s first argument is a graph file, and an optional second argument is the number of threads.

When you run bfs, you’ll see the execution time and the frontier size printed for each step in the algorithm. Correctness will pass for the top-down version (since we’ve given you a correct sequential implementation), but it will be slow. (Note that bfs will report failures for a “bottom-up” and “hybrid” versions of the algorithm, which you will implement later in this assignment.)

In this part of the assignment your job is to parallelize a top-down BFS. As with page rank, you’ll need to focus on identifying parallelism, as well as inserting the appropriate synchronization to ensure correctness. We wish to remind you that you should not expect to achieve near-perfect speedups on this problem.

##### Hints:

- Always start by considering what work can be done in parallel.
- Some part of the computation may need to be synchronized, for example, by wrapping the appropriate code within a critical region using `#pragma omp critical`. However, in this problem you can get by with a single atomic operation called `compare and swap`. You can read about [GCC’s implementation of compare and swap](http://gcc.gnu.org/onlinedocs/gcc-4.1.2/gcc/Atomic-Builtins.html), which is exposed to C code as the `function __sync_bool_compare_and_swap`. If you can figure out how to use `compare-and-swap` for this problem, you will achieve much higher performance than using a critical region.
- Are there conditions where it is possible to avoid using `compare_and_swap`? In other words, when you know in advance that the comparison will fail?
There is a preprocessor macro `VERBOSE` to make it easy to disable useful print per-step timings in your solution (see the top of `breadth_first_search/bfs.cpp`). In general, these printfs occur infrequently enough (only once per BFS step) that they do not notably impact performance, but if you want to disable the printfs during timing, you can use this `#define` as a convenience.

#### 2.4 Task 3: “Bottom-Up” BFS
Think about what behavior might cause a performance problem in the BFS implementation from Part 2.3. An alternative implementation of a breadth-first search step may be more efficient in these situations. Instead of iterating over all vertices in the frontier and marking all vertices adjacent to the frontier, it is possible to implement BFS by having each vertex check whether it should be added to the frontier! Basic pseudocode for the algorithm is as follows:

```cpp
for(each vertex v in graph)
    if(v has not been visited &&
       v shares an incoming edge with a vertex u on the frontier)
            add vertex v to frontier;
```

This algorithm is sometimes referred to as a “bottom-up” implementation of BFS, since each vertex looks “up the BFS tree” to find its ancestor. (As opposed to being found by its ancestor in a “top-down” fashion, as was done in Part 2.3.)

Please implement a bottom-up BFS to compute the shortest path to all the vertices in the graph from the root (see `bfs_bottom_up()` in `breadth_first_search/bfs.cpp`). Start by implementing a simple sequential version. Then parallelize your implementation.

##### Tips/Hints:

- It may be useful to think about how you represent the set of unvisited nodes. Do the top-down and bottom-up versions of the code lend themselves to different implementations?
- How do the synchronization requirements of the bottom-up BFS change?

#### 2.5 Task 4: Hybrid BFS
Notice that in some steps of the BFS, the “bottom-up” BFS is significantly faster than the top-down version. In other steps, the top-down version is significantly faster. This suggests a major performance improvement in your implementation, if you could dynamically choose between your “top-down” and “bottom-up” formulations based on the size of the frontier or other properties of the graph! If you want a solution competitive with the reference one, your implementation will likely have to implement this dynamic optimization. Please provide your solution in `bfs_hybrid()` in `breadth_first_search/bfs.cpp`.

##### Tips/Hints:

- If you used different representations of the frontier in Parts 2.3 and 2.4, you may have to convert between these representations in the hybrid solution. How might you efficiently convert between them? Is there an overhead in doing so?

### Reference
- [OpenMP Official Website](http://openmp.org/)
- [OpenMP5 SPEC](https://www.openmp.org/spec-html/5.0/openmp.html)
- [OpenMP Tutorial](https://computing.llnl.gov/tutorials/openMP/)
- [Clang11 OpenMP Support](https://releases.llvm.org/11.0.0/tools/clang/docs/OpenMPSupport.html)
- [Linux TIME Manual](https://linux.die.net/man/1/time)
- [在 Linux 上使用 Perf 做效能分析(入門篇)](https://tigercosmos.xyz/post/2020/08/system/perf-basic/)
- [Wikipedia: Parallel breadth-first search](https://en.wikipedia.org/wiki/Parallel_breadth-first_search)
- [使用 Compare and Swap 做到 Lock Free](https://tigercosmos.xyz/post/2020/10/system/cas-lock-free/)
- [並行程式設計: Lock-Free Programming](https://hackmd.io/@sysprog/concurrency-lockfree)
- [Atomics 操作](https://hackmd.io/OVPTyhEPTwSHumO28EpJnQ?view)