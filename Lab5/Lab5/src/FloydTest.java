import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
public class FloydTest {
    @Test
    void setEdge(){
        Integer matrix[][] = new Integer[][]{
                {0, Integer.MAX_VALUE, 2},
                {1, 0, 3},
                {2, 3, 0}
        };
        Graph graph = new Graph(matrix);
        graph.setEdge(0, 1, 4);
        assertEquals(4, graph.getVerticesOfGraph()[0][1]);
    }
    @Test
    void removeEdge(){
        Integer matrix[][] = new Integer[][]{
                {0, Integer.MAX_VALUE, 2},
                {1, 0, 3},
                {2, 3, 0}
        };
        Graph graph = new Graph(matrix);
        graph.removeEdge(2, 2);
        assertEquals(Integer.MAX_VALUE, graph.getVerticesOfGraph()[2][2]);
    }
    @Test
    void fillGraph(){
        int n = 5;
        int edges = 10;
        Graph graph = new Graph(n);
        graph.fillGraph(edges);
        int edgeNum = 0;
        for (int i = 0; i < graph.getVerticesOfGraph().length; i++){
            for (int j = 0; j < graph.getVerticesOfGraph().length; j++){
                if (graph.getVerticesOfGraph()[i][j] != Integer.MAX_VALUE && graph.getVerticesOfGraph()[i][j] != 0) edgeNum++;
            }
        }
        assertEquals(edges, edgeNum);
        assertEquals(graph.getVerticesOfGraph().length, n);
    }
    @Test
    void FloydAlgorithm(){
        Integer matrix[][] = new Integer[][]{
                {0, Integer.MAX_VALUE, 3, Integer.MAX_VALUE},
                {2, 0, Integer.MAX_VALUE, Integer.MAX_VALUE},
                {Integer.MAX_VALUE, 7, 0, 1},
                {6, Integer.MAX_VALUE, Integer.MAX_VALUE, 0}
        };
        Integer matrix2[][] = new Integer[][]{
                {0, 10, 3, 4},
                {2, 0, 5, 6},
                {7, 7, 0, 1},
                {6, 16, 9, 0}
        };
        Graph graph = new Graph(matrix);
        Integer expected[][] = graph.FloydAlgorithm();
        for (int i = 0; i < 3; i++){
            assertArrayEquals(expected[i], matrix2[i]);
        }
    }
    @Test
    void FloydAlgorithmParallel(){
        Integer matrix[][] = new Integer[][]{
                {0, Integer.MAX_VALUE, 3, Integer.MAX_VALUE},
                {2, 0, Integer.MAX_VALUE, Integer.MAX_VALUE},
                {Integer.MAX_VALUE, 7, 0, 1},
                {6, Integer.MAX_VALUE, Integer.MAX_VALUE, 0}
        };
        Integer matrix2[][] = new Integer[][]{
                {0, 10, 3, 4},
                {2, 0, 5, 6},
                {7, 7, 0, 1},
                {6, 16, 9, 0}
        };
        Graph graph = new Graph(matrix);
        Integer expected[][] = graph.FloydAlgorithmParallel();
        for (int i = 0; i < 3; i++){
            assertArrayEquals(expected[i], matrix2[i]);
        }
    }
}
