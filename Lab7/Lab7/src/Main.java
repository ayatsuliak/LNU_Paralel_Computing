import java.util.Scanner;

public class Main {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);

        System.out.print("Enter the number of vertices: ");
        int numOfVertex = scanner.nextInt();

        System.out.print("Enter the number of edges: ");
        int numOfEdges = scanner.nextInt();

        System.out.print("Enter the number of start vertex: ");
        int startVertex = scanner.nextInt();

        Graph graph = new Graph(numOfVertex);
        graph.fillGraph(numOfEdges);
        long startTime, endTime, durationSeq, durationPar;
        startTime = System.nanoTime();
        graph.PrimsAlgorithm(startVertex);
        endTime = System.nanoTime();
        durationSeq = (endTime - startTime);
        System.out.println("Sequential time: " + durationSeq + " nanoseconds\n");
        int[] threadNumbers = {2, 3, 4, 8, 16, 32, 64};
        for (int threadNumber: threadNumbers) {
            Graph.setThreadsNumber(threadNumber);
            startTime = System.nanoTime();
            graph.PrimsAlgorithmParallel(50);
            endTime = System.nanoTime();
            durationPar = (endTime - startTime);
            System.out.println("Parallel time for " + threadNumber + " threads: " + durationPar + " nanoseconds");
            System.out.println("Speedup: " + (double) durationSeq / durationPar);
            System.out.println("Efficiency: " + (double) durationSeq / durationPar / threadNumber);
            System.out.println();
        }
    }
}