import javafx.application.Application;
import javafx.application.Platform;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Pane;
import javafx.stage.Stage;
public class Main extends Application {

    @Override
    public void start(Stage primaryStage) {
        // Створюємо кореневий контейнер для головного вікна
        Pane root = new Pane();
        Scene mainScene = new Scene(root, 665, 50);
        primaryStage.setScene(mainScene);

        Button startBallButton = new Button("Start Ball Animation");
        Button startRectangleButton = new Button("Start Rectangle Animation");
        Button startSineWaveButton = new Button("Start Sine Wave Animation");
        Button startRandomCircleButton = new Button("Start Random Circle Animation");

        startBallButton.setOnAction(event -> {
            Stage ballStage = new Stage();
            BallAnimation ballAnimation = new BallAnimation(new Pane(), 30, 30, 30, 140, 2, 2);
            ballAnimation.initializeBall();
            Scene ballScene = new Scene(ballAnimation.getPane(), 400, 300);
            ballStage.setScene(ballScene);
            ballStage.show();

            Button pauseBallButton = new Button("Pause");
            Button resumeBallButton = new Button("Resume");
            Button stopBallButton = new Button("Stop");

            pauseBallButton.setOnAction(e -> {
                ballAnimation.pauseAnimation();
            });

            resumeBallButton.setOnAction(e -> {
                ballAnimation.resumeAnimation();
            });

            stopBallButton.setOnAction(e -> {
                ballAnimation.stopAnimation();
                ballStage.close();
            });

            HBox ballButtons = new HBox(pauseBallButton, resumeBallButton, stopBallButton);
            ballButtons.setLayoutX(10);
            ballButtons.setLayoutY(10);
            ((Pane) ballScene.getRoot()).getChildren().addAll(ballButtons);

            Thread ballThread = new Thread(() -> {
                ballAnimation.startAnimation();
            });
            ballThread.start();
            ballStage.setTitle("First form");
        });
        
        startRectangleButton.setOnAction(event -> {
            Stage rectangleStage = new Stage();
            RectangleAnimation rectangleAnimation = new RectangleAnimation(new Pane(), 100, 100, 100, 100, 4, 4);
            rectangleAnimation.initializeRectangle();
            Scene rectangleScene = new Scene(rectangleAnimation.getPane(), 600, 400);
            rectangleStage.setScene(rectangleScene);
            rectangleStage.show();

            Button pauseRectangleButton = new Button("Pause");
            Button resumeRectangleButton = new Button("Resume");
            Button stopRectangleButton = new Button("Stop");

            pauseRectangleButton.setOnAction(e -> {
                rectangleAnimation.pauseAnimation();
            });

            resumeRectangleButton.setOnAction(e -> {
                rectangleAnimation.resumeAnimation();
            });

            stopRectangleButton.setOnAction(e -> {
                rectangleAnimation.stopAnimation();
                rectangleStage.close();
            });

            HBox rectangleButtons = new HBox(pauseRectangleButton, resumeRectangleButton, stopRectangleButton);
            rectangleButtons.setLayoutX(10);
            rectangleButtons.setLayoutY(10);
            ((Pane) rectangleScene.getRoot()).getChildren().addAll(rectangleButtons);

            Thread rectangleThread = new Thread(() -> {
                rectangleAnimation.startAnimation();
            });
            rectangleThread.start();
            rectangleStage.setTitle("Second form");
        });

        startSineWaveButton.setOnAction(event -> {
            Stage sineWaveStage = new Stage();
            SineWaveAnimation sineWaveAnimation = new SineWaveAnimation(new Pane(), 30, 100, 10000, 400);
            sineWaveAnimation.initializeSineWave();
            Pane sineWavePane = sineWaveAnimation.getPane();
            Scene sineWaveScene = new Scene(sineWavePane, 800, 400);
            sineWaveStage.setScene(sineWaveScene);
            sineWaveStage.show();

            Button pauseSineWaveButton = new Button("Pause");
            Button resumeSineWaveButton = new Button("Resume");
            Button stopSineWaveButton = new Button("Stop");

            pauseSineWaveButton.setOnAction(e -> {
                sineWaveAnimation.pauseAnimation();
            });

            resumeSineWaveButton.setOnAction(e -> {
                sineWaveAnimation.resumeAnimation();
            });

            stopSineWaveButton.setOnAction(e -> {
                sineWaveAnimation.stopAnimation();
                sineWaveStage.close();
            });

            HBox sineWaveButtons = new HBox(pauseSineWaveButton, resumeSineWaveButton, stopSineWaveButton);
            sineWaveButtons.setLayoutX(10);
            sineWaveButtons.setLayoutY(10);
            sineWavePane.getChildren().addAll(sineWaveButtons);

            Thread sineWaveThread = new Thread(() -> {
                sineWaveAnimation.startAnimation();
            });
            sineWaveThread.start();
            sineWaveStage.setTitle("Third form");
        });

        startRandomCircleButton.setOnAction(event -> {
            Stage randomCircleStage = new Stage();
            RandomCircleGenerator randomCircleGenerator = new RandomCircleGenerator(new Pane());
            randomCircleGenerator.initializeCircles();
            Scene randomCircleScene = new Scene(randomCircleGenerator.getPane(), 800, 400);
            randomCircleStage.setScene(randomCircleScene);
            randomCircleStage.show();

            Button pauseRandomCircleButton = new Button("Pause");
            Button resumeRandomCircleButton = new Button("Resume");
            Button stopRandomCircleButton = new Button("Stop");

            pauseRandomCircleButton.setOnAction(e -> {
                randomCircleGenerator.pauseAnimation();
            });

            resumeRandomCircleButton.setOnAction(e -> {
                randomCircleGenerator.resumeAnimation();
            });

            stopRandomCircleButton.setOnAction(e -> {
                randomCircleGenerator.stopAnimation();
                randomCircleStage.close();
            });

            HBox randomCircleButtons = new HBox(pauseRandomCircleButton, resumeRandomCircleButton, stopRandomCircleButton);
            randomCircleButtons.setLayoutX(10);
            randomCircleButtons.setLayoutY(10);
            ((Pane) randomCircleScene.getRoot()).getChildren().addAll(randomCircleButtons);

            Thread randomCircleThread = new Thread(() -> {
                randomCircleGenerator.initializeCircles();
            });
            randomCircleThread.start();
            randomCircleStage.setTitle("Fourth form");
        });

        Button exitButton = new Button("Exit");
        exitButton.setOnAction(event -> {
            primaryStage.close();
            Platform.exit();
        });

        HBox buttons = new HBox(startBallButton, startRectangleButton, startSineWaveButton, startRandomCircleButton, exitButton);
        buttons.setLayoutX(10);
        buttons.setLayoutY(10);
        root.getChildren().addAll(buttons);

        primaryStage.setTitle("Main");
        primaryStage.show();
    }
    public static void main(String[] args) {
        launch(args);
    }
}