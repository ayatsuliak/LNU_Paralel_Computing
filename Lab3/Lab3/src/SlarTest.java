import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class SlarTest {
    @Test
    void TheardsNumbersCheck()
    {
        Slar slar = new Slar(3);
        Slar.setTheardsNumber(4);
        assertEquals(4, Slar.getTheardsNumber());
    }
    @Test
    void calculateDeterminantCheck()
    {
        int[][] A = {
                {2, 3, 1},
                {3, -1, 2},
                {1, 4, -1}
        };
        int[] terms = {1, 1, 2};
        Slar slar = new Slar(3);
        slar.Fill(A, terms);
        assertEquals(14, slar.calculateDeterminant(A));
    }
    @Test
    void ResultCheck()
    {
        int[][] A = {
                {2, 3, 1},
                {3, -1, 2},
                {1, 4, -1}
        };
        int[] terms = {1, 1, 2};
        double[] futureResult = {1.0, 0.0, -1.0};
        Slar slar = new Slar(3);
        slar.Fill(A, terms);
        double[] result = slar.Result();
        assertArrayEquals(futureResult, result);
    }
    @Test
    void ParalelResultCheck()
    {
        int[][] A = {
                {2, 3, 1},
                {3, -1, 2},
                {1, 4, -1}
        };
        int[] terms = {1, 1, 2};
        double[] futureResult = {1.0, 0.0, -1.0};
        Slar slar = new Slar(3);
        slar.Fill(A, terms);
        double[] result = slar.ParalelResult();
        assertArrayEquals(futureResult, result);
    }
    @Test
    void ExceptionsCheck()
    {
        int[][] A = {
                {1, 2, 3},
                {2, 4, 6},
                {3, 6, 9}
        };
        int[] terms = {1, 2, 4};
        Slar slar = new Slar(3);
        slar.Fill(A, terms);
        assertThrows(IllegalArgumentException.class, slar::Result);
        assertThrows(IllegalArgumentException.class, slar::ParalelResult);
    }
}