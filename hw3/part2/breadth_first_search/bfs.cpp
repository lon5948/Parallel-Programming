#include "bfs.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstddef>
#include <omp.h>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

#define ROOT_NODE_ID 0
#define NOT_VISITED_MARKER -1

void vertex_set_clear(vertex_set *list)
{
    list->count = 0;
}

void vertex_set_init(vertex_set *list, int count)
{
    list->max_vertices = count;
    list->vertices = (int *)malloc(sizeof(int) * list->max_vertices);
    vertex_set_clear(list);
}

// Take one step of "top-down" BFS.  For each vertex on the frontier,
// follow all outgoing edges, and add all neighboring vertices to the
// new_frontier.
void top_down_step(
    Graph g,
    vertex_set *frontier,
    vertex_set *new_frontier,
    int *distances)
{
    #pragma omp parallel
    {
        int cnt = 0 ;
        Vertex* temp_frontier = (Vertex*)malloc(sizeof(Vertex) * g->num_nodes);

        #pragma omp for 
        for (int i = 0; i < frontier->count; i++)
        {

            int node = frontier->vertices[i];

            int start_edge = g->outgoing_starts[node];
            int end_edge = (node == g->num_nodes - 1)
                            ? g->num_edges
                            : g->outgoing_starts[node + 1];

            // attempt to add all neighbors to local frontier
            for (int neighbor = start_edge; neighbor < end_edge; neighbor++)
            {
                int outgoing = g->outgoing_edges[neighbor];
                if(distances[outgoing] == NOT_VISITED_MARKER && __sync_bool_compare_and_swap(&distances[outgoing], NOT_VISITED_MARKER, distances[node] + 1))
                {
                    temp_frontier[cnt] = outgoing;
                    cnt++;
                }
            }
        }
        // memcpy to new frontier
        #pragma omp critical
        {
            memcpy(new_frontier->vertices + new_frontier->count, temp_frontier, sizeof(int) * cnt);
            new_frontier->count += cnt;
        }

        free(temp_frontier);
    }
}

// Implements top-down BFS.
//
// Result of execution is that, for each node in the graph, the
// distance to the root is stored in sol.distances.
void bfs_top_down(Graph graph, solution *sol)
{

    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set *frontier = &list1;
    vertex_set *new_frontier = &list2;

    // initialize all nodes to NOT_VISITED
    #pragma omp parallel for
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    
    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    while (frontier->count != 0)
    {

        #ifdef VERBOSE
                double start_time = CycleTimer::currentSeconds();
        #endif

        vertex_set_clear(new_frontier);

        top_down_step(graph, frontier, new_frontier, sol->distances);

        #ifdef VERBOSE
                double end_time = CycleTimer::currentSeconds();
                printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
        #endif

        // swap pointers
        vertex_set *tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
    }
}

void bottom_up_step(
    Graph g,
    vertex_set *frontier,
    vertex_set *new_frontier,
    int *distances,
    int depth)
{   
    // for(each vertex v in graph)
    #pragma omp parallel for schedule(dynamic, 512)
    for (int i = 0; i < g->num_nodes; i++)
    {
        // if(v has not been visited &&
        if(distances[i] == NOT_VISITED_MARKER) 
        {
            int start_edge = g->incoming_starts[i];
            int end_edge = (i == g->num_nodes - 1)
                        ? g->num_edges
                        : g->incoming_starts[i + 1];

            for (int neighbor = start_edge; neighbor < end_edge; neighbor++)
            {
                int incoming = g->incoming_edges[neighbor];

                // v shares an incoming edge with a vertex u on the frontier)
                if(distances[incoming] == depth)
                {
                    // add vertex v to frontier;
                    distances[i] = depth + 1;
                    frontier->vertices[i] = 1;
                    new_frontier->vertices[__sync_fetch_and_add(&new_frontier->count, 1)] = i;
                    break; 
                }
            }
        }
    }
}

void bfs_bottom_up(Graph graph, solution *sol)
{
    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set *frontier = &list1;
    vertex_set *new_frontier = &list2;

    // initialize all nodes to NOT_VISITED
    #pragma omp parallel for
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    
    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    int depth = 0;
    while (frontier->count != 0)
    {

        #ifdef VERBOSE
                double start_time = CycleTimer::currentSeconds();
        #endif

        vertex_set_clear(new_frontier);

        bottom_up_step(graph, frontier, new_frontier, sol->distances, depth);

        #ifdef VERBOSE
                double end_time = CycleTimer::currentSeconds();
                printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
        #endif

        // swap pointers
        vertex_set *tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
        depth++;
    }
}

void bfs_hybrid(Graph graph, solution *sol)
{
    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set *frontier = &list1;
    vertex_set *new_frontier = &list2;

    // initialize all nodes to NOT_VISITED
    #pragma omp parallel for
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    
    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    int depth = 0;
    bool using_bot = 0;
    while (frontier->count != 0)
    {

        #ifdef VERBOSE
                double start_time = CycleTimer::currentSeconds();
        #endif

        vertex_set_clear(new_frontier);

		if((float)(frontier->count)/(float)(graph->num_nodes) < 0.1) {
			top_down_step(graph, frontier, new_frontier, sol->distances);
		}
		else {
			bottom_up_step(graph, frontier, new_frontier, sol->distances, depth);
		}

        #ifdef VERBOSE
                double end_time = CycleTimer::currentSeconds();
                printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
        #endif

        // swap pointers
        vertex_set *tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
        depth++;
    }
}
