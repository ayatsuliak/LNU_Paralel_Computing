import java.util.Scanner;

public class Main
{
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);

        System.out.print("Enter the number of dimension: ");
        int dimension = scanner.nextInt();

        System.out.print("Enter the number of threads: ");
        int threadsNumber = scanner.nextInt();

        Slar slar = new Slar(dimension);
        slar.Fill();
        Slar.setTheardsNumber(threadsNumber);
        long startTime, endTime, durationSeq, durationPar;
        double speedUp, efficiency;
        double[] sequence;
        double[] paralel;
        startTime = System.nanoTime();
        sequence = slar.Result();
        endTime = System.nanoTime();
        durationSeq = (endTime - startTime);
        System.out.println("Sequential time: " + durationSeq + " nanoseconds");
        startTime = System.nanoTime();
        paralel = slar.ParalelResult();
        endTime = System.nanoTime();
        durationPar = (endTime - startTime);
        System.out.println("Paralel time: " + durationPar + " nanoseconds");
        speedUp = (double) durationSeq / durationPar;
        efficiency = speedUp / threadsNumber;
        System.out.println("Speed up: " + speedUp);
        System.out.println("Efficiency: " + efficiency);
    }
}