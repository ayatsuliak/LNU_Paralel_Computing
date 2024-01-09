import java.util.Random;
import java.util.Scanner;

public class Main {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);

        System.out.print("Enter the number of rows of the matrix: ");
        int rows = scanner.nextInt();

        System.out.print("Enter the number of columns of the matrix: ");
        int columns = scanner.nextInt();

        System.out.print("Enter the number of threads: ");
        int numThreads = scanner.nextInt();

        Matrix matrix1 = new Matrix(rows, columns);
        Matrix matrix2 = new Matrix(rows, columns);

        // ДОДАВАННЯ
        /*matrix1.fillMatrix();
        matrix1.setTheardsNumber(numThreads);
        //matrix1.printMatrix();
        //System.out.print("------------------------------\n");
        matrix2.fillMatrix();
        //matrix2.printMatrix();
        //System.out.print("---------------Result---------------\n");

        long startTime = System.nanoTime();
        Matrix resultMatrix2 = matrix1.addMatrixByParallel(matrix2);
        long endTime = System.nanoTime();
        long durationPar = (endTime - startTime);
        //resultMatrix2.printMatrix();
        //System.out.print("---------------Result2---------------\n");
        startTime = System.nanoTime();
        Matrix resultMatrix = matrix1.addMatrix(matrix2);
        endTime = System.nanoTime();
        long durationSeq = (endTime - startTime);
        //resultMatrix.printMatrix();
        System.out.println();
        System.out.println("Time taken sequential: " + durationSeq + " nanoseconds");
        System.out.println("Time taken parallel: " + durationPar + " nanoseconds");
        System.out.println("Speedup: " + (double) durationSeq / durationPar);
        System.out.println("Efficiency: " + (double) durationSeq / durationPar / numThreads);
        System.out.println();*/


        // ВІДНІМАННЯ
        /*matrix1.fillMatrix();
        matrix1.setTheardsNumber(numThreads);
        //matrix1.printMatrix();
        //System.out.print("------------------------------\n");
        matrix2.fillMatrix();
        //matrix2.printMatrix();
        //System.out.print("---------------Result---------------\n");

        long startTime = System.nanoTime();
        Matrix resultMatrix2 = matrix1.subtractionMatrixByParallel(matrix2);
        long endTime = System.nanoTime();
        long durationPar = (endTime - startTime);
        //resultMatrix2.printMatrix();
        //System.out.print("---------------Result2---------------\n");
        startTime = System.nanoTime();
        Matrix resultMatrix = matrix1.subtractionMatrix(matrix2);
        endTime = System.nanoTime();
        long durationSeq = (endTime - startTime);
        //resultMatrix.printMatrix();
        System.out.println();
        System.out.println("Time taken sequential: " + durationSeq + " nanoseconds");
        System.out.println("Time taken parallel: " + durationPar + " nanoseconds");
        System.out.println("Speedup: " + (double) durationSeq / durationPar);
        System.out.println("Efficiency: " + (double) durationSeq / durationPar / numThreads);
        System.out.println();*/


        // МНОЖЕННЯ
        matrix1.fillMatrix();
        matrix1.setTheardsNumber(numThreads);
        //matrix1.printMatrix();
        //System.out.print("------------------------------\n");
        matrix2.fillMatrix();
        //matrix2.printMatrix();
        //System.out.print("---------------Result---------------\n");

        long startTime = System.nanoTime();
        Matrix resultMatrix2 = matrix1.multiplyMatrixByParallel(matrix2);
        long endTime = System.nanoTime();
        long durationPar = (endTime - startTime);
        //resultMatrix2.printMatrix();
        //System.out.print("---------------Result2---------------\n");
        startTime = System.nanoTime();
        Matrix resultMatrix = matrix1.multiplyMatrix(matrix2);
        endTime = System.nanoTime();
        long durationSeq = (endTime - startTime);
        //resultMatrix.printMatrix();
        System.out.println();
        System.out.println("Time taken sequential: " + durationSeq + " nanoseconds");
        System.out.println("Time taken parallel: " + durationPar + " nanoseconds");
        System.out.println("Speedup: " + (double) durationSeq / durationPar);
        System.out.println("Efficiency: " + (double) durationSeq / durationPar / numThreads);
        System.out.println();
    }
}