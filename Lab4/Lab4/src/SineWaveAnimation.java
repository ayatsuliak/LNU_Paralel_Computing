import javafx.application.Platform;
import javafx.scene.layout.Pane;
import javafx.scene.shape.MoveTo;
import javafx.scene.shape.Path;
import javafx.scene.shape.QuadCurveTo;
import javafx.scene.shape.StrokeType;
public class SineWaveAnimation {
    private Pane pane;
    private Path sineWave;
    private volatile boolean paused = false;
    private Thread animationThread;
    private double amplitude;
    private double frequency;
    private double width;
    private double height;
    private double phase;
    public SineWaveAnimation(Pane pane, double amplitude, double frequency, double width, double height) {
        this.pane = pane;
        this.amplitude = amplitude;
        this.frequency = frequency;
        this.width = width;
        this.height = height;
        this.phase = 0.0;
    }
    public void initializeSineWave() {
        sineWave = new Path();
        sineWave.setStrokeWidth(2);
        sineWave.setStrokeType(StrokeType.CENTERED);
        sineWave.getElements().add(new MoveTo(0, height / 2)); // Початкова точка
        updateSineWave();
        pane.getChildren().add(sineWave);
    }
    public Pane getPane() {
        return pane;
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
    public void startAnimation() {
        animationThread = new Thread(() -> {
            long lastUpdateTime = 0;

            while (!Thread.currentThread().isInterrupted()) {
                synchronized (animationThread) {
                    while (paused) {
                        try {
                            animationThread.wait();
                        } catch (InterruptedException e) {
                            Thread.currentThread().interrupt();
                        }
                    }
                }

                // Оновлення синусоїди
                phase += 0.01;
                updateSineWave();

                try {
                    Thread.sleep(16); // Затримка для визначення кадру (приблизно 60 FPS)
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            }
        });

        animationThread.start();
    }
    private void updateSineWave() {
        Platform.runLater(() -> {
            sineWave.getElements().clear();
            sineWave.getElements().add(new MoveTo(0, height / 2)); // Початкова точка синусоїди

            for (double x = 0; x < width; x += 1) {
                if (x + 1 < width) {
                    double y = height / 2 - amplitude * Math.sin(2 * Math.PI * frequency * x / width + phase);
                    sineWave.getElements().add(new QuadCurveTo(x, y, x + 1, y));
                }
            }
        });
    }
}