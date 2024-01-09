import javafx.animation.KeyFrame;
import javafx.animation.Timeline;
import javafx.application.Platform;
import javafx.scene.layout.Pane;
import javafx.scene.paint.Color;
import javafx.scene.shape.Circle;
public class RandomCircleGenerator {
    private Pane pane;
    private volatile boolean paused = false;
    private Thread animationThread;
    public RandomCircleGenerator(Pane pane) {
        this.pane = pane;
    }
    public Pane getPane() {
        return pane;
    }
    public void initializeCircles() {
        animationThread = new Thread(() -> {
            while (!animationThread.isInterrupted()) {
                synchronized (animationThread) {
                    while (paused) {
                        try {
                            animationThread.wait();
                        } catch (InterruptedException e) {
                            Thread.currentThread().interrupt();
                        }
                    }
                }

                createRandomCircle();

                try {
                    Thread.sleep(500); // Очікування протягом 0.5 секунди
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            }
        });

        animationThread.start();
    }
    private void createRandomCircle() {
        if (paused) {
            return;
        }
        double radius = Math.random() * 30 + 10;
        double x = Math.random() * (pane.getWidth() - radius * 2) + radius;
        double y = Math.random() * (pane.getHeight() - radius * 2) + radius;

        Circle circle = new Circle(x, y, radius);
        circle.setFill(Color.color(Math.random(), Math.random(), Math.random()));

        Platform.runLater(() -> {
            pane.getChildren().add(circle);
        });
    }
    public void pauseAnimation() {
        paused = true;
    }
    public void resumeAnimation() {
        paused = false;
        synchronized (animationThread) {
            animationThread.notify();
        }
    }
    public void stopAnimation() {
        if (animationThread != null && animationThread.isAlive()) {
            animationThread.interrupt();
        }
    }
}