import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
public class DijkstraTest {
    @Test
    void DijkstraAlgorithm(){
        Integer matrix[][] = new Integer[][]{
                {0, 6, Integer.MAX_VALUE, 1, Integer.MAX_VALUE},
                {6, 0, 5, 2, 2},
                {Integer.MAX_VALUE, 5, 0, Integer.MAX_VALUE, 5},
                {1, 2, Integer.MAX_VALUE, 0, 1},
                {Integer.MAX_VALUE, 2, 5, 1, 0}
        };
        Integer matrix2[] = new Integer[]{0, 3, 7, 1, 2};
        Graph graph = new Graph(matrix);
        Integer expected[] = graph.DijkstraAlgorithm(0);
        for (int i = 0; i < 3; i++){
            assertArrayEquals(expected, matrix2);
        }
    }
    @Test
    void DijkstraAlgorithmParallel(){
        Integer matrix[][] = new Integer[][]{
                {0, 6, Integer.MAX_VALUE, 1, Integer.MAX_VALUE},
                {6, 0, 5, 2, 2},
                {Integer.MAX_VALUE, 5, 0, Integer.MAX_VALUE, 5},
                {1, 2, Integer.MAX_VALUE, 0, 1},
                {Integer.MAX_VALUE, 2, 5, 1, 0}
        };
        Integer matrix2[] = new Integer[]{0, 3, 7, 1, 2};
        Graph graph = new Graph(matrix);
        Integer expected[] = graph.DijkstraAlgorithmParallel(0);
        assertArrayEquals(expected, matrix2);
    }
}
