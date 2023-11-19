#include "page_rank.h"

#include <stdlib.h>
#include <cmath>
#include <omp.h>
#include <utility>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

// pageRank --
//
// g:           graph to process (see common/graph.h)
// solution:    array of per-vertex vertex scores (length of array is num_nodes(g))
// damping:     page-rank algorithm's damping parameter
// convergence: page-rank algorithm's convergence threshold
//
void pageRank(Graph g, double *solution, double damping, double convergence)
{

  // initialize vertex weights to uniform probability. Double
  // precision scores are used to avoid underflow for large graphs
  bool converged = false;
  int numNodes = num_nodes(g);
  double equal_prob = 1.0 / numNodes;
  double *solution_new = (double *)malloc(sizeof(double) * numNodes);

  double global_diff;
  double  no_outgoing_sum = 0.0;

  #pragma omp parallel for reduction(+:no_outgoing_sum)
  for (int i = 0 ; i < numNodes ; i++)
  {
    solution[i] = equal_prob;
    solution_new[i] = 0.0;
    if (outgoing_size(g, i) == 0)
      no_outgoing_sum += solution[i];
  }

   while (!converged) {
    // compute score_new[vi] for all nodes vi:
    #pragma omp parallel for
    for (int i = 0 ; i < numNodes ; i++) {
    
      const Vertex* start = incoming_begin(g, i);
      const Vertex* end = incoming_end(g, i);
      
      for (const Vertex* vj = start ; vj != end ; ++vj ) {
        solution_new[i] += (solution[*vj] / outgoing_size(g, *vj));
      }

      solution_new[i] = (damping * solution_new[i]) + (1.0 - damping) / numNodes;
      solution_new[i] += damping * no_outgoing_sum / numNodes;
    }

    // compute how much per-node scores have changed
    global_diff = 0.0, no_outgoing_sum = 0.0;

    #pragma omp parallel for reduction(+:global_diff, no_outgoing_sum)
    for (int i = 0 ; i < numNodes ; i++) {
      global_diff += fabs(solution_new[i] - solution[i]);
      solution[i] = solution_new[i];
      solution_new[i] = 0.0;
      if (outgoing_size(g, i) == 0)
        no_outgoing_sum += solution[i];
    }

    // quit once algorithm has converged
    converged = (global_diff < convergence);
  }
}
