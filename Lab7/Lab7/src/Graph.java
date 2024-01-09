import java.util.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.concurrent.*;

public class Graph {
    record Edge(int source, int destination, int weight) {
    }
    private final int numOfVertex;
    private final ArrayList<Edge> edges = new ArrayList<>();
    private static int threadsNumber = 1;
    public Graph(int numOfVertex){
        this.numOfVertex = numOfVertex;
    }
    public void setEdge(int source, int destination, int weight){
        if (source == destination) return;
        this.removeEdge(source, destination);
        this.edges.add(new Edge(source, destination, weight));
    }
    public void removeEdge(int source, int destination){
        if (source == destination) return;
        for (int i = 0; i < this.edges.size(); i++){
            if (this.edges.get(i).source == source && this.edges.get(i).destination == destination ||
                    this.edges.get(i).source == destination && this.edges.get(i).destination == source){
                this.edges.remove(i);
                return;
            }
        }
    }
    private boolean isEdge(int source, int destination){
        if (source == destination) return false;
        for (Edge edge : this.edges) {
            if (edge.source == source && edge.destination == destination) return true;
            if (edge.source == destination && edge.destination == source) return true;
        }
        return false;
    }
    public int getNumberOfVertices() {
        return numOfVertex;
    }
    public int getNumberOfEdges() {
        return this.edges.size();
    }
    public void fillGraph(int numberOfEdges) {
        if (numberOfEdges > (this.numOfVertex * this.numOfVertex - this.numOfVertex) / 2)
            throw new IllegalArgumentException("Number of edges is too big");
        if (numberOfEdges < this.numOfVertex - 1)
            throw new IllegalArgumentException("Number of edges is too small");

        int edgesLeft = numberOfEdges;
        for (int i = 0; i < this.numOfVertex - 1; i++){
            int weight = (int) (Math.random() * 100 + 1);
            this.setEdge(i, i + 1, weight);
            edgesLeft--;
        }

        while (edgesLeft > 0){
            int source = (int) (Math.random() * this.numOfVertex);
            int destination = (int) (Math.random() * this.numOfVertex);
            while (source == destination || this.isEdge(source, destination)){
                source = (int) (Math.random() * this.numOfVertex);
                destination = (int) (Math.random() * this.numOfVertex);
            }
            int weight = (int) (Math.random() * 100 + 1);
            this.setEdge(source, destination, weight);
            edgesLeft--;
        }
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
    @Override
    public String toString() {
        StringBuilder stringBuilder = new StringBuilder();
        stringBuilder.append("Edges:\n");
        for (Edge edge : this.edges) {
            stringBuilder.append(edge.source).append(" -> ").append(edge.destination).append(" (").append(edge.weight).append(")\n");
        }
        return stringBuilder.toString();
    }
    record ReturnObject(ArrayList<Edge> edges, int sumOfPaths) {
        @Override
        public String toString() {
            StringBuilder stringBuilder = new StringBuilder();
            stringBuilder.append("Edges:\n");
            for (Edge edge : this.edges) {
                stringBuilder.append(edge.source).append(" -> ").append(edge.destination).append(" (").append(edge.weight).append(")\n");
            }
            stringBuilder.append("Sum of paths: ").append(this.sumOfPaths);
            return stringBuilder.toString();
        }
        public ArrayList<Edge> getEdges() {
            return edges;
        }
        public int getSumOfPaths() {
            return sumOfPaths;
        }
    }
    public static int getThreadsNumber(){
        return threadsNumber;
    }
    public static void setThreadsNumber(int threadsNumber){
        Graph.threadsNumber = threadsNumber;
    }
    public ReturnObject PrimsAlgorithm(int startVertex) {
        ArrayList<Edge> selectedEdges = new ArrayList<>();
        int sumOfPaths = 0;

        ArrayList<ArrayList<VertexDistancePair>> adjacencyList = new ArrayList<>(); // сусідні вершини для кожної вершини
        for (int i = 0; i < this.numOfVertex; i++) adjacencyList.add(new ArrayList<>());

        for (Edge edge : this.edges) {
            adjacencyList.get(edge.source).add(new VertexDistancePair(edge.destination, edge.weight));
            adjacencyList.get(edge.destination).add(new VertexDistancePair(edge.source, edge.weight));
        }

        PriorityQueue<VertexDistancePair> priorityQueue = new PriorityQueue<>();
        priorityQueue.add(new VertexDistancePair(startVertex, 0));

        boolean[] visited = new boolean[this.numOfVertex];
        int[] parent = new int[this.numOfVertex];
        int[] minDistance = new int[this.numOfVertex];
        Arrays.fill(minDistance, Integer.MAX_VALUE);
        minDistance[startVertex] = 0;

        while (!priorityQueue.isEmpty()) {
            VertexDistancePair current = priorityQueue.remove();
            int currentVertex = current.vertex;
            if (visited[currentVertex]) continue;

            visited[currentVertex] = true;
            int currentWeight = current.distance;
            sumOfPaths += currentWeight;

            if (currentVertex != startVertex) {
                int parentVertex = parent[currentVertex];
                selectedEdges.add(new Edge(parentVertex, currentVertex, currentWeight));
            }

            for (VertexDistancePair pair : adjacencyList.get(currentVertex)) {
                if (!visited[pair.vertex] && pair.distance < minDistance[pair.vertex]) {
                    priorityQueue.add(new VertexDistancePair(pair.vertex, pair.distance));
                    parent[pair.vertex] = currentVertex;
                    minDistance[pair.vertex] = pair.distance;
                }
            }
        }

        return new ReturnObject(selectedEdges, sumOfPaths);
    }
    public ReturnObject PrimsAlgorithmParallel(int startVertex) {
        ArrayList<Edge> selectedEdges = new ArrayList<>();
        int sumOfPaths = 0;

        ArrayList<ArrayList<VertexDistancePair>> adjacencyList = new ArrayList<>();
        for (int i = 0; i < this.numOfVertex; i++) adjacencyList.add(new ArrayList<>());

        for (Edge edge : this.edges) {
            adjacencyList.get(edge.source).add(new VertexDistancePair(edge.destination, edge.weight));
            adjacencyList.get(edge.destination).add(new VertexDistancePair(edge.source, edge.weight));
        }

        PriorityQueue<VertexDistancePair> priorityQueue = new PriorityQueue<>();
        priorityQueue.add(new VertexDistancePair(startVertex, 0));

        boolean[] visited = new boolean[this.numOfVertex];
        int[] parent = new int[this.numOfVertex];
        int[] minDistance = new int[this.numOfVertex];
        Arrays.fill(minDistance, Integer.MAX_VALUE);
        minDistance[startVertex] = 0;

        ForkJoinPool forkJoinPool = new ForkJoinPool(threadsNumber);

        class ParallelProcessNeighbors extends RecursiveAction {
            int vertex;

            ParallelProcessNeighbors(int vertex) {
                this.vertex = vertex;
            }

            @Override
            protected void compute() {
                for (VertexDistancePair pair : adjacencyList.get(vertex)) {
                    if (!visited[pair.vertex] && pair.distance < minDistance[pair.vertex]) {
                        priorityQueue.add(new VertexDistancePair(pair.vertex, pair.distance));
                        parent[pair.vertex] = vertex;
                        minDistance[pair.vertex] = pair.distance;
                    }
                }
            }
        }

        while (!priorityQueue.isEmpty()) {
            VertexDistancePair current = priorityQueue.remove();
            int currentVertex = current.vertex;
            if (visited[currentVertex]) continue;

            visited[currentVertex] = true;
            int currentWeight = current.distance;
            sumOfPaths += currentWeight;

            if (currentVertex != startVertex) {
                int parentVertex = parent[currentVertex];
                selectedEdges.add(new Edge(parentVertex, currentVertex, currentWeight));
            }

            ParallelProcessNeighbors task = new ParallelProcessNeighbors(currentVertex);
            forkJoinPool.invoke(task);
        }
        return new ReturnObject(selectedEdges, sumOfPaths);
    }
}

