import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
public class Slar
{
    private final int dimension;
    private final int[][] matrix;
    private static int theardsNumber;
    public Slar(int dimension)
    {
        this.dimension = dimension;
        this.matrix = new int[dimension][dimension + 1];
    }
    public void Fill()
    {
        Random rand = new Random();
        int upperBound = 10;
        for(int i = 0; i < this.dimension; i++)
        {
            for(int j = 0; j < this.dimension; j++)
            {
                this.matrix[i][j] = (int) (Math.pow(-1, rand.nextInt(2)) * rand.nextInt(upperBound));
            }
        }
        for(int i = 0; i < this.dimension; i++)
        {
            this.matrix[i][this.dimension] = (int) (Math.pow(-1, rand.nextInt(2)) * rand.nextInt(upperBound));
        }
    }
    public void Fill(int[][] coef, int[] terms)
    {
        for (int i = 0; i < this.dimension; i++)
        {
            System.arraycopy(coef[i], 0, this.matrix[i], 0, this.dimension);
        }
        for(int i = 0; i < this.dimension; i++)
        {
            this.matrix[i][this.dimension] = terms[i];
        }
    }
    public void printSystem(){
        for(int i = 0; i < this.dimension; i++)
        {
            for(int j = 0; j < this.dimension; j++)
            {
                System.out.print(this.matrix[i][j] + " ");
            }
            System.out.println("| " + this.matrix[i][this.dimension]);
        }
    }
    private void swapColumns(int[][] matrix, int col1, int col2)
    {
        for(int i = 0; i < matrix.length; i++)
        {
            int temp = matrix[i][col1];
            matrix[i][col1] = matrix[i][col2];
            matrix[i][col2] = temp;
        }
    }
    public static int getTheardsNumber()
    {
        return Slar.theardsNumber;
    }
    public static void setTheardsNumber(int theardsNumber)
    {
        Slar.theardsNumber = theardsNumber;
    }
    public int calculateDeterminant(int[][] matrix)
    {
        if (matrix.length == 1)
        {
            return matrix[0][0];
        }

        int det = 0;
        for (int col = 0; col < matrix.length; col++)
        {
            det += matrix[0][col] * cofactor(matrix, 0, col) * calculateDeterminant(subMatrix(matrix, 0, col));
        }

        return det;
    }
    // Метод для знаходження мінора (підматриці без рядка і стовпця)
    private int[][] subMatrix(int[][] matrix, int row, int col)
    {
        int[][] sub = new int[matrix.length - 1][matrix.length - 1];
        int subRow = 0;
        int subCol = 0;

        for (int i = 0; i < matrix.length; i++)
        {
            if (i != row)
            {
                subCol = 0;
                for (int j = 0; j < matrix.length; j++)
                {
                    if (j != col)
                    {
                        sub[subRow][subCol] = matrix[i][j];
                        subCol++;
                    }
                }
                subRow++;
            }
        }
        return sub;
    }
    // Метод для обчислення коефіцієнта сполученого з мінором
    private int cofactor(int[][] matrix, int row, int col)
    {
        if ((row + col) % 2 == 0)
        {
            return 1;
        } else
        {
            return -1;
        }
    }
    public double[] Result()
    {
        int det = this.calculateDeterminant(this.matrix);
        if(det == 0){
            throw new IllegalArgumentException("The system has no or infinite solution.");
        }

        double[] result = new double[this.dimension];
        for(int i = 0; i < this.dimension; i++){
            this.swapColumns(this.matrix, i, this.dimension);
            result[i] = (double) this.calculateDeterminant(this.matrix) / det;
            this.swapColumns(this.matrix, i, this.dimension);
        }
        return result;
    }
    public double[] ParalelResult()
    {
        int det = this.calculateDeterminant(this.matrix);
        if (det == 0) {
            throw new IllegalArgumentException("The system has no or infinite solution.");
        }

        double[] result = new double[this.dimension];
        ExecutorService executorService = Executors.newFixedThreadPool(Slar.getTheardsNumber()); //створення пулу потоків

        for (int i = 0; i < this.dimension; i++) {
            final int col = i;
            executorService.submit(() -> {
                int[][] tempMatrix = new int[this.dimension][this.dimension + 1];
                for (int j = 0; j < this.dimension; j++) {
                    System.arraycopy(this.matrix[j], 0, tempMatrix[j], 0, this.dimension);
                }
                for (int j = 0; j < this.dimension; j++) {
                    tempMatrix[j][col] = this.matrix[j][this.dimension];
                }

                int colDet = this.calculateDeterminant(tempMatrix);
                result[col] = (double) colDet / det;
            });
        }

        executorService.shutdown();
        try {
            executorService.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

        return result;
    }
}
