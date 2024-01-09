import java.util.*;
import java.util.concurrent.*;
public class Graph {
    private final int numOfVertex;
    private final Integer[][] verticesOfGraph;
    private static int threadsNumber = 1;
    public Graph(int numOfVertex){
        this.numOfVertex = numOfVertex;
        this.verticesOfGraph = new Integer[numOfVertex][numOfVertex];
        for (int i = 0; i < numOfVertex; i++) {
            for (int j = 0; j < numOfVertex; j++) {
                if (i == j) {
                    verticesOfGraph[i][j] = 0;
                } else {
                    verticesOfGraph[i][j] = Integer.MAX_VALUE;
                }
            }
        }
    }
    public Graph(Integer[][] verticesOfGraph){
        this.numOfVertex = verticesOfGraph.length;
        this.verticesOfGraph = new Integer[numOfVertex][numOfVertex];
        for (int i = 0; i < numOfVertex; i++) {
            for (int j = 0; j < numOfVertex; j++) {
                this.verticesOfGraph[i][j] = verticesOfGraph[i][j];
            }
        }
    }
    public void setEdge(int source, int destination, int weight){
        if (source == destination) return;
        this.verticesOfGraph[source][destination] = weight;
    }
    public void removeEdge(int source, int destination){
        if (source == destination) return;
        this.verticesOfGraph[source][destination] = Integer.MAX_VALUE;
    }
    public void fillGraph(int numberOfEdges){
        if (numberOfEdges > this.numOfVertex * this.numOfVertex - this.numOfVertex)
            throw new IllegalArgumentException("Number of edges is too big");
        if (numberOfEdges < this.numOfVertex)
            throw new IllegalArgumentException("Number of edges is too small");

        for (int i = 0; i < numberOfEdges; i++){
            int source = (int) (Math.random() * this.numOfVertex);
            int destination = (int) (Math.random() * this.numOfVertex);
            while (this.verticesOfGraph[source][destination] != Integer.MAX_VALUE){
                source = (int) (Math.random() * this.numOfVertex);
                destination = (int) (Math.random() * this.numOfVertex);
            }
            int weight = (int) (Math.random() * 100 + 1);
            this.setEdge(source, destination, weight);
        }
    }
    public Integer[][] getVertices(){
        return this.verticesOfGraph;
    }
    public static int getThreadsNumber(){
        return threadsNumber;
    }
    public static void setThreadsNumber(int threadsNumber){
        Graph.threadsNumber = threadsNumber;
    }
    public Integer[] DijkstraAlgorithm(int source) {
        Integer[] distances = new Integer[numOfVertex];
        Arrays.fill(distances, Integer.MAX_VALUE);
        distances[source] = 0;

        PriorityQueue<VertexDistancePair> priorityQueue = new PriorityQueue<>();
        priorityQueue.add(new VertexDistancePair(source, 0));

        while (!priorityQueue.isEmpty()) {
            VertexDistancePair current = priorityQueue.poll();
            int u = current.vertex;

            if (current.distance > distances[u]) {
                continue;
            }

            for (int v = 0; v < numOfVertex; v++) {
                int edgeWeight = verticesOfGraph[u][v];

                if (edgeWeight != Integer.MAX_VALUE && distances[u] != Integer.MAX_VALUE
                        && distances[u] + edgeWeight < distances[v]) {
                    distances[v] = distances[u] + edgeWeight;
                    priorityQueue.add(new VertexDistancePair(v, distances[v]));
                }
            }
        }

        return distances;
    }
    private static class VertexDistancePair implements Comparable<VertexDistancePair> {
        int vertex;
        int distance;

        VertexDistancePair(int vertex, int distance) {
            this.vertex = vertex;
            this.distance = distance;
        }

        @Override
        public int compareTo(VertexDistancePair other) {
            return Integer.compare(this.distance, other.distance);
        }
    }
    public Integer[] DijkstraAlgorithmParallel(int source) {
        Integer[] distances = new Integer[numOfVertex];
        Arrays.fill(distances, Integer.MAX_VALUE);
        distances[source] = 0;

        ForkJoinPool pool = new ForkJoinPool(threadsNumber);
        pool.invoke(new DijkstraTask(0, numOfVertex, source, distances));

        return distances;
    }
    private class DijkstraTask extends RecursiveTask<Void> {
        private static int THRESHOLD = 100;
        private final int start;
        private final int end;
        private final int source;
        private final Integer[] distances;

        DijkstraTask(int start, int end, int source, Integer[] distances) {
            this.start = start;
            this.end = end;
            this.source = source;
            this.distances = distances;
        }

        @Override
        protected Void compute() {
            if (end - start <= THRESHOLD) {
                DijkstraSequential(start, end);
            } else {
                int mid = start + (end - start) / 2;
                invokeAll(
                        new DijkstraTask(start, mid, source, distances),
                        new DijkstraTask(mid, end, source, distances)
                );
            }
            return null;
        }

        private void DijkstraSequential(int start, int end) {
            PriorityQueue<VertexDistancePair> priorityQueue = new PriorityQueue<>();
            priorityQueue.add(new VertexDistancePair(source, 0));

            while (!priorityQueue.isEmpty()) {
                VertexDistancePair current = priorityQueue.poll();
                int u = current.vertex;

                // чи відома відстань до вершини u та чи є можливість скоротити відстань до вершини v через вершину u.
                for (int v = start; v < end; v++) {
                    if (verticesOfGraph[u][v] != Integer.MAX_VALUE && distances[u] != Integer.MAX_VALUE
                            && distances[u] + verticesOfGraph[u][v] < distances[v]) {
                        distances[v] = distances[u] + verticesOfGraph[u][v];
                        priorityQueue.add(new VertexDistancePair(v, distances[v]));
                    }
                }
            }
        }
    }
}
