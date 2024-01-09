import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
public class Graph {
    private final int numOfVertex;
    private final Integer[][] verticesOfGraph;
    private static int thredsNumber = 1;

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
    public void fillGraph(int numOfEdges){
        if (numOfEdges > this.numOfVertex * this.numOfVertex - this.numOfVertex)
            throw new IllegalArgumentException("Number of edges is too big");
        if (numOfEdges < this.numOfVertex)
            throw new IllegalArgumentException("Number of edges is too small");

        for (int i = 0; i < numOfEdges; i++){
            int from = (int) (Math.random() * this.numOfVertex);
            int to = (int) (Math.random() * this.numOfVertex);
            while (this.verticesOfGraph[from][to] != Integer.MAX_VALUE){
                from = (int) (Math.random() * this.numOfVertex);
                to = (int) (Math.random() * this.numOfVertex);
            }
            int weight = (int) (Math.random() * 100 + 1);
            this.setEdge(from, to, weight);
        }
    }
    public void setEdge(int from, int to, int weight) {
        this.verticesOfGraph[from][to] = weight;
    }
    public void removeEdge(int from, int to) {
        this.verticesOfGraph[from][to] = Integer.MAX_VALUE;
    }
    public static void setThreadsNumber(int threadsNumber) {
        thredsNumber = threadsNumber;
    }
    Integer[][] getVerticesOfGraph(){
        return this.verticesOfGraph;
    }
    //A_k[i, j] = min(A_k-1[i, j], A_k-1[i, k] + A_k-1[k, j])
    public Integer[][] FloydAlgorithm(){
        if (verticesOfGraph == null || this.numOfVertex <= 0) {
            throw new IllegalArgumentException("Invalid graph or number of vertices.");
        }

        if (verticesOfGraph.length != this.numOfVertex || verticesOfGraph[0].length != this.numOfVertex) {
            throw new IllegalArgumentException("Graph matrix size doesn't match the number of vertices.");
        }
        for (int k = 0; k < this.numOfVertex; k++) {
            for (int i = 0; i < this.numOfVertex; i++) {
                for (int j = 0; j < this.numOfVertex; j++) {
                    if (verticesOfGraph[i][k] != Integer.MAX_VALUE && verticesOfGraph[k][j] != Integer.MAX_VALUE) {
                        int throughK = verticesOfGraph[i][k] + verticesOfGraph[k][j];
                        if (throughK < verticesOfGraph[i][j]) {
                            verticesOfGraph[i][j] = throughK;
                        }
                    }
                }
            }
        }
        return verticesOfGraph;
    }
    public Integer[][] FloydAlgorithmParallel() {
        if (verticesOfGraph == null || this.numOfVertex <= 0) {
            throw new IllegalArgumentException("Invalid graph or number of vertices.");
        }

        if (verticesOfGraph.length != this.numOfVertex || verticesOfGraph[0].length != this.numOfVertex) {
            throw new IllegalArgumentException("Graph matrix size doesn't match the number of vertices.");
        }
        ExecutorService executor = Executors.newFixedThreadPool(this.thredsNumber);

        for (int k = 0; k < this.numOfVertex; k++) {
            final int kFinal = k;
            for (int i = 0; i < this.numOfVertex; i++) {
                final int iFinal = i;
                executor.submit(() -> {
                    for (int j = 0; j < this.numOfVertex; j++) {
                        if (verticesOfGraph[iFinal][kFinal] != Integer.MAX_VALUE && verticesOfGraph[kFinal][j] != Integer.MAX_VALUE) {
                            int throughK = verticesOfGraph[iFinal][kFinal] + verticesOfGraph[kFinal][j];
                            if (throughK < verticesOfGraph[iFinal][j]) {
                                verticesOfGraph[iFinal][j] = throughK;
                            }
                        }
                    }
                });
            }
        }

        executor.shutdown();
        try {
            executor.awaitTermination(1, TimeUnit.MINUTES);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        return verticesOfGraph;
    }
}
