import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
public class Matrix
{
    private int rows;
    private int columns;
    private final int[][] matrix;
    private static int theardsNumber;

    Matrix(int rows, int columns)
    {
        this.rows = rows;
        this.columns = columns;
        this.matrix = new int[rows][columns];
    }
    Matrix(int[][] matrix)
    {
        this.matrix = matrix;
    }
    public void fillMatrix()
    {
        Random random = new Random();
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                this.matrix[i][j] = random.nextInt(1001);
            }
        }
    }
    public void printMatrix()
    {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                System.out.print(matrix[i][j] + " ");
            }
            System.out.println();
        }
    }
    public int getRows()
    {
        return this.rows;
    }
    public void setRows(int rows)
    {
        this.rows = rows;
    }
    public int getColumns()
    {
        return this.columns;
    }
    public void setColumns(int columns)
    {
        this.columns = columns;
    }
    public static int getTheardsNumber()
    {
        return Matrix.theardsNumber;
    }
    public static void setTheardsNumber(int theardsNumber)
    {
        Matrix.theardsNumber = theardsNumber;
    }
    public int getElement(int rows, int columns)
    {
        return this.matrix[rows][columns];
    }
    public void setElement(int rows, int columns, int value)
    {
        this.matrix[rows][columns] = value;
    }
    public Matrix addMatrix(Matrix other)
    {
        if (this.rows != other.rows || this.columns != other.columns)
        {
            throw new IllegalArgumentException("Matrix dimensions do not match for addition.");
        }
        Matrix resultMatrix = new Matrix(other.rows, other.columns);
        for (int i = 0; i < other.rows; i++)
        {
            for (int j = 0; j < other.columns; j++)
            {
                resultMatrix.setElement(i, j,  this.matrix[i][j] + other.getElement(i, j));
            }
        }
        return resultMatrix;
    }
    public Matrix addMatrixByParallel(Matrix other) {
        if (this.rows != other.rows || this.columns != other.columns) {
            throw new IllegalArgumentException("Matrix dimensions do not match for addition.");
        }

        Matrix resultMatrix = new Matrix(other.rows, other.columns);
        ExecutorService executorService = Executors.newFixedThreadPool(Matrix.getTheardsNumber());

        for (int i = 0; i < other.rows; i++) {
            final int row = i;
            executorService.execute(() -> {
                for (int j = 0; j < other.columns; j++) {
                    int sum = this.matrix[row][j] + other.getElement(row, j);
                    resultMatrix.setElement(row, j, sum);
                }
            });
        }

        executorService.shutdown();
        try {
            executorService.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

        return resultMatrix;
    }
    public Matrix subtractionMatrix(Matrix other)
    {
        if (this.rows != other.rows || this.columns != other.columns)
        {
            throw new IllegalArgumentException("Matrix dimensions do not match for subtraction.");
        }
        Matrix resultMatrix = new Matrix(other.rows, other.columns);
        for (int i = 0; i < other.rows; i++)
        {
            for (int j = 0; j < other.columns; j++)
            {
                resultMatrix.setElement(i, j,  this.matrix[i][j] - other.getElement(i, j));
            }
        }
        return resultMatrix;
    }
    public Matrix subtractionMatrixByParallel(Matrix other)
    {
        if (this.rows != other.rows || this.columns != other.columns) {
            throw new IllegalArgumentException("Matrix dimensions do not match for subtraction.");
        }

        Matrix resultMatrix = new Matrix(other.rows, other.columns);
        ExecutorService executorService = Executors.newFixedThreadPool(Matrix.getTheardsNumber());

        for (int i = 0; i < other.rows; i++) {
            final int row = i;
            executorService.execute(() -> {
                for (int j = 0; j < other.columns; j++) {
                    int sub = this.matrix[row][j] - other.getElement(row, j);
                    resultMatrix.setElement(row, j, sub);
                }
            });
        }

        executorService.shutdown();
        try {
            executorService.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

        return resultMatrix;
    }
    public Matrix multiplyMatrix(Matrix other)
    {
        if (this.columns != other.rows) {
            throw new IllegalArgumentException("Matrix dimensions do not match for multiplication.");
        }

        int resultRows = this.rows;
        int resultColumns = other.columns;
        Matrix resultMatrix = new Matrix(resultRows, resultColumns);

        for (int i = 0; i < resultRows; i++) {
            for (int j = 0; j < resultColumns; j++) {
                int sum = 0;
                for (int k = 0; k < this.columns; k++) {
                    sum += this.getElement(i, k) * other.getElement(k, j);
                }
                resultMatrix.setElement(i, j, sum);
            }
        }

        return resultMatrix;
    }
    public Matrix multiplyMatrixByParallel(Matrix other) {
        if (this.columns != other.rows) {
            throw new IllegalArgumentException("Matrix dimensions do not match for multiplication.");
        }

        int resultRows = this.rows;
        int resultColumns = other.columns;
        Matrix resultMatrix = new Matrix(resultRows, resultColumns);
        ExecutorService executorService = Executors.newFixedThreadPool(Matrix.getTheardsNumber());

        for (int i = 0; i < resultRows; i++) {
            final int row = i;
            executorService.execute(() -> {
                for (int j = 0; j < resultColumns; j++) {
                    int sum = 0;
                    for (int k = 0; k < this.columns; k++) {
                        sum += this.matrix[row][k] * other.getElement(k, j);
                    }
                    resultMatrix.setElement(row, j, sum);
                }
            });
        }

        executorService.shutdown();
        try {
            executorService.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

        return resultMatrix;
    }
}
