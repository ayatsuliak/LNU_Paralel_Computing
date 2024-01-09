import java.util.Scanner;
public class Main {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);

        System.out.print("Enter the number of vertices: ");
        int numOfVertex = scanner.nextInt();

        System.out.print("Enter the number of edges: ");
        int numOfEdges = scanner.nextInt();

        Graph graph = new Graph(numOfVertex);
        graph.fillGraph(numOfEdges);

        long startTime, endTime, durationSeq, durationPar;

        Integer[][] floydSeq;
        Integer[][] floydPar;

        startTime = System.nanoTime();
        floydSeq = graph.FloydAlgorithm();
        endTime = System.nanoTime();
        durationSeq = (endTime - startTime);
        System.out.println("Sequential time: " + durationSeq + " nanoseconds");
        System.out.println();


        int[] threadNumbers = new int[]{2, 3, 4, 8, 16};
        for (int threadNumber: threadNumbers) {
            Graph.setThreadsNumber(threadNumber);

            startTime = System.nanoTime();
            floydPar = graph.FloydAlgorithmParallel();
            endTime = System.nanoTime();
            durationPar = (endTime - startTime);
            System.out.println("Threads number: " + threadNumber);
            System.out.println("Parallel time: " + durationPar + " nanoseconds");
            System.out.println("Speed up: " + (double) durationSeq / durationPar);
            System.out.println("Efficiency: " + (double) durationSeq / durationPar / threadNumber);
            System.out.println();
        }
    }
}