import org.junit.jupiter.api.Test;

import java.util.ArrayList;

import static org.junit.jupiter.api.Assertions.*;
public class PrimeTest {
    @Test
    public void PrimeAlgorithm(){
        Graph graph = new Graph(9);

        graph.setEdge(0, 1, 4);
        graph.setEdge(0, 7, 8);
        graph.setEdge(1, 2, 8);
        graph.setEdge(1, 7, 11);
        graph.setEdge(2, 3, 7);
        graph.setEdge(2, 8, 2);
        graph.setEdge(2, 5, 4);
        graph.setEdge(3, 4, 9);
        graph.setEdge(3, 5, 14);
        graph.setEdge(4, 5, 10);
        graph.setEdge(5, 6, 2);
        graph.setEdge(6, 7, 1);
        graph.setEdge(6, 8, 6);
        graph.setEdge(7, 8, 7);

        ArrayList<Graph.Edge> expected = new ArrayList<>();
        expected.add(new Graph.Edge(0, 1, 4));
        expected.add(new Graph.Edge(0, 7, 8));
        expected.add(new Graph.Edge(7, 6, 1));
        expected.add(new Graph.Edge(6, 5, 2));
        expected.add(new Graph.Edge(5, 2, 4));
        expected.add(new Graph.Edge(2, 8, 2));
        expected.add(new Graph.Edge(2, 3, 7));
        expected.add(new Graph.Edge(3, 4, 9));

        Graph.ReturnObject returnObject = graph.PrimsAlgorithm(0);
        assertEquals(37, returnObject.getSumOfPaths());
        assertEquals(8, returnObject.getEdges().size());
        assertEquals(expected, returnObject.getEdges());
    }
    @Test
    public void PrimeAlgorithmParallel(){
        Graph graph = new Graph(9);

        graph.setEdge(0, 1, 4);
        graph.setEdge(0, 7, 8);
        graph.setEdge(1, 2, 8);
        graph.setEdge(1, 7, 11);
        graph.setEdge(2, 3, 7);
        graph.setEdge(2, 8, 2);
        graph.setEdge(2, 5, 4);
        graph.setEdge(3, 4, 9);
        graph.setEdge(3, 5, 14);
        graph.setEdge(4, 5, 10);
        graph.setEdge(5, 6, 2);
        graph.setEdge(6, 7, 1);
        graph.setEdge(6, 8, 6);
        graph.setEdge(7, 8, 7);

        ArrayList<Graph.Edge> expected = new ArrayList<>();
        expected.add(new Graph.Edge(0, 1, 4));
        expected.add(new Graph.Edge(0, 7, 8));
        expected.add(new Graph.Edge(7, 6, 1));
        expected.add(new Graph.Edge(6, 5, 2));
        expected.add(new Graph.Edge(5, 2, 4));
        expected.add(new Graph.Edge(2, 8, 2));
        expected.add(new Graph.Edge(2, 3, 7));
        expected.add(new Graph.Edge(3, 4, 9));

        Graph.ReturnObject returnObject = graph.PrimsAlgorithmParallel(0);
        assertEquals(37, returnObject.getSumOfPaths());
        assertEquals(8, returnObject.getEdges().size());
        assertEquals(expected, returnObject.getEdges());
    }
}
