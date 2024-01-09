import javafx.application.Platform;
import javafx.scene.layout.Pane;
import javafx.scene.paint.Color;
import javafx.scene.shape.Rectangle;
public class RectangleAnimation {
    private Rectangle rectangle;
    private Pane pane;
    private volatile boolean paused = false;
    private Thread animationThread;
    private double x, y;
    private double width;
    private double height;
    private double velocityX;
    private double velocityY;
    public RectangleAnimation(Pane pane, double x, double y, double width, double height, double velocityX, double velocityY) {
        this.pane = pane;
        this.x = x;
        this.y = y;
        this.width = width;
        this.height = height;
        this.velocityX = velocityX;
        this.velocityY = velocityY;
    }
    public Rectangle initializeRectangle() {
        rectangle = new Rectangle(x, y, width, height);
        rectangle.setFill(Color.BLUE);
        pane.getChildren().add(rectangle); // Додаємо прямокутник на переданий Pane
        return rectangle; // Повертаємо об'єкт Rectangle
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
            animationThread.interrupt(); // Використовуємо interrupt() для завершення потоку
        }
    }
    public void startAnimation() {
        animationThread = new Thread(() -> {
            long lastUpdateTime = 0;
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

                long currentTime = System.nanoTime();

                if (lastUpdateTime == 0) {
                    lastUpdateTime = currentTime;
                    continue;
                }

                double elapsedTime = (currentTime - lastUpdateTime) / 1_000_000_000.0;

                double newWidth = rectangle.getWidth() + velocityX * elapsedTime;
                double newHeight = rectangle.getHeight() + velocityY * elapsedTime;

                // Перевірка на зіткнення з межами екрану та зміну напрямку
                if (newWidth <= 0 || newWidth + x >= pane.getWidth()) {
                    velocityX = -velocityX; // Зміна напрямку по ширині
                }
                if (newHeight <= 0 || newHeight + y >= pane.getHeight()) {
                    velocityY = -velocityY; // Зміна напрямку по висоті
                }

                // Встановлення нових розмірів для прямокутника
                Platform.runLater(() -> {
                    rectangle.setWidth(newWidth);
                    rectangle.setHeight(newHeight);
                });

                // Оновлення останнього часу оновлення
                lastUpdateTime = currentTime;

                try {
                    Thread.sleep(16); // Затримка для визначення кадру (приблизно 60 FPS)
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            }
        });

        animationThread.start();
    }
}