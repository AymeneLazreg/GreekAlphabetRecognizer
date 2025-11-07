// SymbolRecognizer.java
// Java 11+ (requiert JavaFX). Single-file JavaFX app with template matching and Auto-tune N.
// Version mise à jour : agrégation des résultats d'auto-tune pour une décision plus stable.

import javafx.application.Application;
import javafx.application.Platform;
import javafx.embed.swing.SwingFXUtils;
import javafx.geometry.Insets;
import javafx.scene.Scene;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.control.*;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.layout.*;
import javafx.scene.paint.Color;
import javafx.stage.DirectoryChooser;
import javafx.stage.FileChooser;
import javafx.stage.Stage;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.*;
import java.util.List;
import java.util.stream.Collectors;

public class SymbolRecognizer extends Application {

    // UI fields
    private Stage primaryStage;
    private ImageView imageView;
    private Canvas overlay;
    private Label statusLabel;
    private Spinner<Integer> spinnerN;

    // Template data
    private Map<String, List<double[]>> templates = new HashMap<>(); // label -> list of feature vectors (for LoadTemplates)
    private final Map<Integer, Map<String, List<double[]>>> templatesCache = new HashMap<>(); // cache per N for autotune
    private File currentTemplatesDir = null; // set when user loads templates folder
    private File lastLoadedImageFile = null;

    // Params
    private int RESAMPLE_N = 64; // default N
    private final int AUTOTUNE_STEPS_DEFAULT = 50;
    private final int AUTOTUNE_MINN_DEFAULT = 8;
    private final int AUTOTUNE_MAXN_DEFAULT = 120;

    public static void main(String[] args) {
        launch(args);
    }

    @Override
    public void start(Stage stage) {
        this.primaryStage = stage;
        stage.setTitle("Symbol Recognizer (template matching) - Auto-tune N (agrégation)");

        imageView = new ImageView();
        imageView.setFitWidth(520);
        imageView.setPreserveRatio(true);
        imageView.setSmooth(true);

        overlay = new Canvas(520, 520);

        StackPane imagePane = new StackPane();
        imagePane.getChildren().addAll(imageView, overlay);
        imagePane.setPrefSize(520, 520);

        Button loadTemplatesBtn = new Button("Charger dossier templates");
        Button loadImageBtn = new Button("Charger image à reconnaître");
        Button recognizeBtn = new Button("Reconnaître");
        Button autoTuneBtn = new Button("Auto-tune N (agg.)");
        spinnerN = new Spinner<>(8, 512, RESAMPLE_N, 8);
        spinnerN.valueProperty().addListener((obs, oldV, newV) -> RESAMPLE_N = newV);

        statusLabel = new Label("Templates: 0 | Image: none");

        HBox controls = new HBox(10, loadTemplatesBtn, loadImageBtn, new Label("N:"), spinnerN, recognizeBtn, autoTuneBtn);
        controls.setPadding(new Insets(8));

        VBox root = new VBox(8, imagePane, controls, statusLabel);
        root.setPadding(new Insets(10));

        // Actions
        loadTemplatesBtn.setOnAction(e -> {
            DirectoryChooser dc = new DirectoryChooser();
            dc.setTitle("Sélectionner dossier contenant les templates (ex: alpha_01.jpg)");
            File dir = dc.showDialog(primaryStage);
            if (dir != null && dir.isDirectory()) {
                currentTemplatesDir = dir;
                int count = loadTemplatesFromDirectory(dir, RESAMPLE_N);
                statusLabel.setText("Templates: " + count + " | Image: " + (lastLoadedImageFile == null ? "none" : lastLoadedImageFile.getName()));
                templatesCache.clear();
            }
        });

        loadImageBtn.setOnAction(e -> {
            FileChooser fc = new FileChooser();
            fc.getExtensionFilters().addAll(
                    new FileChooser.ExtensionFilter("Images", "*.jpg", "*.jpeg", "*.png", "*.bmp")
            );
            File f = fc.showOpenDialog(primaryStage);
            if (f != null) {
                try {
                    BufferedImage bi = ImageIO.read(f);
                    Image fx = SwingFXUtils.toFXImage(bi, null);
                    imageView.setImage(fx);
                    overlay.setWidth(imageView.getFitWidth());
                    double scale = imageView.getFitWidth() / bi.getWidth();
                    overlay.setHeight(bi.getHeight() * scale);
                    lastLoadedImageFile = f;
                    statusLabel.setText("Templates: " + templatesCount() + " | Image: " + f.getName());
                    clearOverlay();
                } catch (Exception ex) {
                    showAlert("Erreur", "Impossible de charger l'image : " + ex.getMessage());
                }
            }
        });

        recognizeBtn.setOnAction(e -> {
            if (lastLoadedImageFile == null) {
                showAlert("Info", "Charge d'abord une image à reconnaître.");
                return;
            }
            if (currentTemplatesDir == null || templates.isEmpty()) {
                showAlert("Info", "Charge d'abord un dossier de templates.");
                return;
            }
            try {
                BufferedImage bi = ImageIO.read(lastLoadedImageFile);
                RecognizeResult res = processAndRecognize(bi, RESAMPLE_N);
                drawOverlay(res.resampledPoints, bi.getWidth(), bi.getHeight());
                statusLabel.setText(String.format("Match: %s (score %.4f) — dist=%.4f (N=%d)",
                        res.bestLabel, res.similarityScore, res.bestDistance, RESAMPLE_N));
            } catch (Exception ex) {
                showAlert("Erreur", "Problème pendant la reconnaissance : " + ex.getMessage());
                ex.printStackTrace();
            }
        });

        autoTuneBtn.setOnAction(ev -> {
            if (currentTemplatesDir == null) { showAlert("Erreur", "Charge d'abord le dossier de templates."); return; }
            if (lastLoadedImageFile == null) { showAlert("Erreur", "Charge d'abord une image à optimiser."); return; }
            // Run auto-tune in background thread
            new Thread(() -> {
                try {
                    AggregationDecision decision = autoTuneAndAggregate(currentTemplatesDir, lastLoadedImageFile, AUTOTUNE_STEPS_DEFAULT, AUTOTUNE_MINN_DEFAULT, AUTOTUNE_MAXN_DEFAULT);
                    Platform.runLater(() -> {
                        // show aggregated decision
                        String msg = String.format("Décision agrégée : %s\nConfiance (poids) = %.3f\nFréquence = %.2f%%\nmeanDist=%.4f stdDist=%.4f\nrecommended N = %d",
                                decision.chosenLabel, decision.confidence, 100.0 * decision.frequencyFraction, decision.meanDist, decision.stdDist, decision.recommendedN);
                        showAlert("Auto-tune (agrégé) terminé", msg);
                        // apply recommended N
                        spinnerN.getValueFactory().setValue(decision.recommendedN);
                        RESAMPLE_N = decision.recommendedN;
                        // run recognition with recommended N and draw overlay
                        try {
                            BufferedImage bi = ImageIO.read(lastLoadedImageFile);
                            RecognizeResult res = processAndRecognize(bi, RESAMPLE_N);
                            drawOverlay(res.resampledPoints, bi.getWidth(), bi.getHeight());
                            statusLabel.setText(String.format("After Auto-tune -> Match: %s (score %.4f) — dist=%.4f (N=%d)",
                                    res.bestLabel, res.similarityScore, res.bestDistance, RESAMPLE_N));
                        } catch (Exception ex) {
                            ex.printStackTrace();
                        }
                    });
                } catch (Exception ex) {
                    ex.printStackTrace();
                    Platform.runLater(() -> showAlert("Erreur Auto-tune", ex.getMessage()));
                }
            }).start();
        });

        Scene scene = new Scene(root, 760, 720);
        stage.setScene(scene);
        stage.show();
    }

    // ----------------- Template loading -----------------
    private int loadTemplatesFromDirectory(File dir, int n) {
        templates.clear();
        File[] files = dir.listFiles((d, name) -> {
            String nm = name.toLowerCase();
            return nm.endsWith(".jpg") || nm.endsWith(".jpeg") || nm.endsWith(".png") || nm.endsWith(".bmp");
        });
        if (files == null) return 0;
        int count = 0;
        for (File f : files) {
            try {
                BufferedImage bi = ImageIO.read(f);
                double[] feat = imageToFeature(bi, n);
                if (feat == null) continue;
                String label = inferLabelFromFilename(f.getName());
                templates.computeIfAbsent(label, k -> new ArrayList<>()).add(feat);
                count++;
            } catch (Exception ex) {
                System.err.println("Failed to load template " + f.getName() + ": " + ex.getMessage());
            }
        }
        // clear cache because templates changed
        templatesCache.clear();
        return count;
    }

    private int templatesCount() {
        return templates.values().stream().mapToInt(List::size).sum();
    }

    private String inferLabelFromFilename(String name) {
        String s = name;
        int dot = s.lastIndexOf('.');
        if (dot > 0) s = s.substring(0, dot);
        for (String sep : new String[]{"_", "-", " "}) {
            int k = s.indexOf(sep);
            if (k > 0) return s.substring(0, k).toLowerCase();
        }
        return s.toLowerCase();
    }

    // ----------------- Core pipeline: image -> feature -----------------
    // returns null if no valid component found
    private double[] imageToFeature(BufferedImage img, int N) {
        boolean[][] bin = binarize(img);
        if (bin == null) return null;
        int[][] labels = connectedComponents(bin);
        if (labels == null) return null;
        int largest = largestComponentLabel(labels);
        if (largest <= 0) return null;
        boolean[][] mask = componentMask(labels, largest);
        List<Point> boundary = extractBoundaryPixels(mask);
        if (boundary.isEmpty()) return null;
        List<Point> ordered = orderBoundaryByAngle(boundary);
        List<Point> resampled = resamplePolyline(ordered, N);
        normalizePoints(resampled);
        return toFeatureVector(resampled);
    }

    // wrapper that also returns boundary/resampled points for drawing
    private RecognizeResult processAndRecognize(BufferedImage img, int N) {
        double[] feat = imageToFeature(img, N);
        if (feat == null) throw new RuntimeException("Aucun symbole détecté dans l'image.");
        boolean[][] bin = binarize(img);
        int[][] labels = connectedComponents(bin);
        int largest = largestComponentLabel(labels);
        boolean[][] mask = componentMask(labels, largest);
        List<Point> boundary = extractBoundaryPixels(mask);
        List<Point> ordered = orderBoundaryByAngle(boundary);
        List<Point> resampled = resamplePolyline(ordered, N);
        normalizePoints(resampled);

        // match using currently loaded templates (if available), otherwise fallback to cache for this N
        String bestLabel = null;
        double bestDist = Double.POSITIVE_INFINITY;

        // prefer using templatesCache if currentTemplatesDir != null and cache has N
        Map<String, List<double[]>> candidateTemplates = templates;
        if (templatesCache.containsKey(N)) candidateTemplates = templatesCache.get(N);
        for (Map.Entry<String, List<double[]>> e : candidateTemplates.entrySet()) {
            for (double[] tfeat : e.getValue()) {
                if (tfeat.length != feat.length) continue;
                double d = euclideanDistance(feat, tfeat);
                if (d < bestDist) {
                    bestDist = d;
                    bestLabel = e.getKey();
                }
            }
        }
        double simScore = 1.0 / (1.0 + bestDist);
        return new RecognizeResult(bestLabel, bestDist, simScore, resampled, ordered);
    }

    // ----------------- Image processing helpers -----------------
    // Convert to binary (foreground = true) using grayscale + Otsu
    private boolean[][] binarize(BufferedImage img) {
        int w = img.getWidth(), h = img.getHeight();
        int[] hist = new int[256];
        int[] gray = new int[w * h];
        int idx = 0;
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int rgb = img.getRGB(x, y);
                int r = (rgb >> 16) & 0xff;
                int g = (rgb >> 8) & 0xff;
                int b = rgb & 0xff;
                int gr = (int) Math.round(0.299 * r + 0.587 * g + 0.114 * b);
                gray[idx++] = gr;
                hist[gr]++;
            }
        }
        int total = w * h;
        int thresh = otsuThreshold(hist, total);
        boolean[][] bin = new boolean[h][w];
        idx = 0;
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                // foreground = dark (symbol is black)
                bin[y][x] = (gray[idx++] < thresh);
            }
        }
        return bin;
    }

    private int otsuThreshold(int[] hist, int total) {
        double sum = 0;
        for (int t = 0; t < 256; t++) sum += t * hist[t];
        double sumB = 0;
        int wB = 0;
        int wF;
        double varMax = 0;
        int threshold = 128;
        for (int t = 0; t < 256; t++) {
            wB += hist[t];
            if (wB == 0) continue;
            wF = total - wB;
            if (wF == 0) break;
            sumB += (double) (t * hist[t]);
            double mB = sumB / wB;
            double mF = (sum - sumB) / wF;
            double varBetween = (double) wB * (double) wF * (mB - mF) * (mB - mF);
            if (varBetween > varMax) {
                varMax = varBetween;
                threshold = t;
            }
        }
        return threshold;
    }

    // Connected components labeling (8-neighbors). returns labels[y][x] with 0 = background
    private int[][] connectedComponents(boolean[][] bin) {
        int h = bin.length, w = bin[0].length;
        int[][] labels = new int[h][w];
        int currentLabel = 0;
        int[] dx = {-1, 0, 1, -1, 1, -1, 0, 1};
        int[] dy = {-1, -1, -1, 0, 0, 1, 1, 1};
        int[] queue = new int[w * h];
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                if (!bin[y][x] || labels[y][x] != 0) continue;
                currentLabel++;
                int qh = 0, qt = 0;
                queue[qt++] = y * w + x;
                labels[y][x] = currentLabel;
                while (qh < qt) {
                    int v = queue[qh++];
                    int cy = v / w, cx = v % w;
                    for (int k = 0; k < 8; k++) {
                        int nx = cx + dx[k], ny = cy + dy[k];
                        if (nx >= 0 && nx < w && ny >= 0 && ny < h) {
                            if (bin[ny][nx] && labels[ny][nx] == 0) {
                                labels[ny][nx] = currentLabel;
                                queue[qt++] = ny * w + nx;
                            }
                        }
                    }
                }
            }
        }
        return labels;
    }

    private int largestComponentLabel(int[][] labels) {
        if (labels == null) return -1;
        int h = labels.length, w = labels[0].length;
        Map<Integer, Integer> counts = new HashMap<>();
        for (int y = 0; y < h; y++) for (int x = 0; x < w; x++) {
            int v = labels[y][x];
            if (v > 0) counts.put(v, counts.getOrDefault(v, 0) + 1);
        }
        int best = -1, bestCount = 0;
        for (Map.Entry<Integer, Integer> e : counts.entrySet()) {
            if (e.getValue() > bestCount) { best = e.getKey(); bestCount = e.getValue(); }
        }
        return best;
    }

    private boolean[][] componentMask(int[][] labels, int label) {
        int h = labels.length, w = labels[0].length;
        boolean[][] mask = new boolean[h][w];
        for (int y = 0; y < h; y++) for (int x = 0; x < w; x++) mask[y][x] = (labels[y][x] == label);
        return mask;
    }

    // boundary pixels = pixels in mask that have at least one 8-neighbor outside mask
    private List<Point> extractBoundaryPixels(boolean[][] mask) {
        int h = mask.length, w = mask[0].length;
        List<Point> boundary = new ArrayList<>();
        for (int y = 0; y < h; y++) for (int x = 0; x < w; x++) {
            if (!mask[y][x]) continue;
            boolean isBoundary = false;
            for (int dy = -1; dy <= 1 && !isBoundary; dy++) for (int dx = -1; dx <= 1 && !isBoundary; dx++) {
                if (dx == 0 && dy == 0) continue;
                int nx = x + dx, ny = y + dy;
                if (nx < 0 || nx >= w || ny < 0 || ny >= h || !mask[ny][nx]) isBoundary = true;
            }
            if (isBoundary) boundary.add(new Point(x, y));
        }
        return boundary;
    }

    // Order boundary pixels by angle around centroid
    private List<Point> orderBoundaryByAngle(List<Point> boundary) {
        double cx = 0, cy = 0;
        for (Point p : boundary) { cx += p.x; cy += p.y; }
        cx /= boundary.size(); cy /= boundary.size();
        final double centerX = cx;
        final double centerY = cy;
        List<Point> copy = new ArrayList<>(boundary);
        copy.sort(Comparator.comparingDouble(p -> Math.atan2(p.y - centerY, p.x - centerX)));
        return copy;
    }

    // Resample polyline (closed) to N points equally spaced along the loop
    private List<Point> resamplePolyline(List<Point> poly, int N) {
        if (poly.size() < 2) {
            List<Point> res = new ArrayList<>();
            Point p = poly.isEmpty() ? new Point(0, 0) : poly.get(0);
            for (int i = 0; i < N; i++) res.add(new Point(p.x, p.y));
            return res;
        }
        List<Point> pts = new ArrayList<>(poly);
        pts.add(poly.get(0));
        double[] seg = new double[pts.size() - 1];
        double total = 0;
        for (int i = 0; i < pts.size() - 1; i++) {
            double dx = pts.get(i + 1).x - pts.get(i).x;
            double dy = pts.get(i + 1).y - pts.get(i).y;
            seg[i] = Math.hypot(dx, dy);
            total += seg[i];
        }
        if (total == 0) {
            List<Point> res = new ArrayList<>();
            Point p = pts.get(0);
            for (int i = 0; i < N; i++) res.add(new Point(p.x, p.y));
            return res;
        }
        double step = total / (N - 1.0);
        List<Point> out = new ArrayList<>();
        out.add(new Point(pts.get(0).x, pts.get(0).y));
        int segIdx = 0;
        for (int k = 1; k < N - 1; k++) {
            double target = k * step;
            double run = 0;
            for (int s = 0; s < segIdx; s++) run += seg[s];
            while (segIdx < seg.length && run + seg[segIdx] < target) {
                run += seg[segIdx];
                segIdx++;
            }
            if (segIdx >= seg.length) {
                out.add(new Point(pts.get(pts.size() - 2).x, pts.get(pts.size() - 2).y));
                continue;
            }
            double segStartDist = run;
            double localT = (target - segStartDist) / (seg[segIdx] == 0 ? 1.0 : seg[segIdx]);
            Point a = pts.get(segIdx);
            Point b = pts.get(segIdx + 1);
            double ix = a.x + localT * (b.x - a.x);
            double iy = a.y + localT * (b.y - a.y);
            out.add(new Point(ix, iy));
        }
        out.add(new Point(pts.get(pts.size() - 2).x, pts.get(pts.size() - 2).y));
        return out;
    }

    // Normalize points (translate centroid to 0, scale by max absolute coordinate -> [-1,1])
    private void normalizePoints(List<Point> pts) {
        double cx = 0, cy = 0;
        for (Point p : pts) { cx += p.x; cy += p.y; }
        cx /= pts.size(); cy /= pts.size();
        double maxAbs = 1e-9;
        for (Point p : pts) {
            p.x -= cx; p.y -= cy;
            maxAbs = Math.max(maxAbs, Math.abs(p.x));
            maxAbs = Math.max(maxAbs, Math.abs(p.y));
        }
        if (maxAbs < 1e-9) maxAbs = 1.0;
        for (Point p : pts) { p.x /= maxAbs; p.y /= maxAbs; }
    }

    private double[] toFeatureVector(List<Point> pts) {
        double[] f = new double[pts.size() * 2];
        for (int i = 0; i < pts.size(); i++) {
            f[2 * i] = pts.get(i).x;
            f[2 * i + 1] = pts.get(i).y;
        }
        return f;
    }

    // ----------------- Matching helpers -----------------
    private double euclideanDistance(double[] a, double[] b) {
        if (a.length != b.length) throw new IllegalArgumentException("Length mismatch");
        double s = 0;
        for (int i = 0; i < a.length; i++) {
            double d = a[i] - b[i];
            s += d * d;
        }
        return Math.sqrt(s);
    }

    // ----------------- Drawing overlay -----------------
    private void drawOverlay(List<Point> resampled, int imgW, int imgH) {
        clearOverlay();
        Image fx = imageView.getImage();
        if (fx == null) return;
        double displayW = imageView.getFitWidth();
        double scale = displayW / imgW;
        overlay.setWidth(displayW);
        overlay.setHeight(imgH * scale);

        GraphicsContext gc = overlay.getGraphicsContext2D();
        gc.clearRect(0, 0, overlay.getWidth(), overlay.getHeight());

        gc.setLineWidth(2);
        gc.setStroke(Color.RED);
        if (!resampled.isEmpty()) {
            Point p0 = resampled.get(0);
            gc.beginPath();
            gc.moveTo(p0.x * scale, p0.y * scale);
            for (int i = 1; i < resampled.size(); i++) {
                Point p = resampled.get(i);
                gc.lineTo(p.x * scale, p.y * scale);
            }
            gc.closePath();
            gc.stroke();
        }
        gc.setFill(Color.BLUE.deriveColor(1, 1, 1, 0.7));
        for (Point p : resampled) {
            gc.fillOval(p.x * scale - 2, p.y * scale - 2, 4, 4);
        }
    }

    private void clearOverlay() {
        GraphicsContext gc = overlay.getGraphicsContext2D();
        gc.clearRect(0, 0, overlay.getWidth(), overlay.getHeight());
    }

    private void showAlert(String title, String message) {
        Alert a = new Alert(Alert.AlertType.INFORMATION);
        a.setTitle(title);
        a.setHeaderText(null);
        a.setContentText(message);
        a.showAndWait();
    }

    // ----------------- Auto-tune and aggregation -----------------

    private int[] linspaceInt(int min, int max, int k) {
        if (k <= 1) return new int[]{min};
        int[] res = new int[k];
        for (int i = 0; i < k; i++) {
            double t = (double) i / (k - 1);
            res[i] = min + (int) Math.round(t * (max - min));
            if (res[i] < min) res[i] = min;
            if (res[i] > max) res[i] = max;
        }
        return res;
    }

    private void ensureTemplatesCached(File templatesDir, int N) {
        if (templatesCache.containsKey(N)) return;
        Map<String, List<double[]>> localMap = new HashMap<>();
        File[] files = templatesDir.listFiles((d, name) -> {
            String nm = name.toLowerCase();
            return nm.endsWith(".jpg") || nm.endsWith(".jpeg") || nm.endsWith(".png") || nm.endsWith(".bmp");
        });
        if (files == null) {
            templatesCache.put(N, localMap);
            return;
        }
        for (File f : files) {
            try {
                BufferedImage bi = ImageIO.read(f);
                double[] feat = imageToFeature(bi, N);
                if (feat == null) continue;
                String label = inferLabelFromFilename(f.getName());
                localMap.computeIfAbsent(label, k -> new ArrayList<>()).add(feat);
            } catch (Exception ex) {
                System.err.println("Erreur chargement template " + f.getName() + " pour N=" + N + " : " + ex.getMessage());
            }
        }
        templatesCache.put(N, localMap);
    }

    private RecognizeResultExtended recognizeWithCachedTemplates(BufferedImage img, int N) {
        ensureTemplatesCached(currentTemplatesDir, N);
        Map<String, List<double[]>> mapForN = templatesCache.getOrDefault(N, Collections.emptyMap());
        double[] feat = imageToFeature(img, N);
        if (feat == null) return null;
        String bestLabel = null;
        double bestDist = Double.POSITIVE_INFINITY;
        for (Map.Entry<String, List<double[]>> e : mapForN.entrySet()) {
            for (double[] tfeat : e.getValue()) {
                if (tfeat.length != feat.length) continue;
                double d = euclideanDistance(feat, tfeat);
                if (d < bestDist) {
                    bestDist = d;
                    bestLabel = e.getKey();
                }
            }
        }
        double simScore = 1.0 / (1.0 + bestDist);
        return new RecognizeResultExtended(bestLabel, bestDist, simScore, N);
    }

    /**
     * Test many N and aggregate the results across N.
     * Returns an AggregationDecision summarizing the aggregated decision.
     */
    private AggregationDecision autoTuneAndAggregate(File templatesDir, File imageFile, int numSteps, int minN, int maxN) throws Exception {
        if (templatesDir == null || imageFile == null) throw new IllegalArgumentException("templatesDir or imageFile null");
        this.currentTemplatesDir = templatesDir;
        BufferedImage img = ImageIO.read(imageFile);
        int[] Ns = linspaceInt(minN, maxN, numSteps);
        List<RecognizeResultExtended> all = new ArrayList<>();
        long t0 = System.currentTimeMillis();
        for (int N : Ns) {
            ensureTemplatesCached(templatesDir, N);
            RecognizeResultExtended res = recognizeWithCachedTemplates(img, N);
            if (res == null) continue;
            all.add(res);
            System.out.printf("N=%d -> label=%s dist=%.4f score=%.4f%n", N, res.label, res.distance, res.score);
        }
        long t1 = System.currentTimeMillis();
        System.out.printf("Auto-tune (collecte) terminé en %.2fs, testé %d valeurs.%n", (t1-t0)/1000.0, all.size());
        if (all.isEmpty()) throw new RuntimeException("Aucun résultat lors de l'auto-tune.");

        AggregationDecision decision = aggregateResults(all);
        // print top labels for diagnostics
        System.out.println("=== Top labels (par poids agrégé) ===");
        int rank = 1;
        for (Map.Entry<String, Double> e : decision.labelSumScores.entrySet()) {
            System.out.printf("Top %d: label=%s sumScore=%.4f count=%d meanDist=%.4f%n",
                    rank++, e.getKey(), e.getValue(), decision.labelCounts.get(e.getKey()), decision.labelMeanDist.get(e.getKey()));
        }
        return decision;
    }

    /**
     * Aggregate a list of RecognizeResultExtended (one per tested N) and return aggregated decision.
     * Strategy: group by label; compute sum(score), mean dist, std dist, count.
     * Choose label with max sum(score). recommendedN = N where that label had max individual score.
     */
    private AggregationDecision aggregateResults(List<RecognizeResultExtended> all) {
        Map<String, List<RecognizeResultExtended>> byLabel = new HashMap<>();
        double totalSumScore = 0.0;
        for (RecognizeResultExtended r : all) {
            if (r.label == null) continue;
            byLabel.computeIfAbsent(r.label, k -> new ArrayList<>()).add(r);
            totalSumScore += r.score;
        }
        // compute sums and stats
        Map<String, Double> sumScores = new HashMap<>();
        Map<String, Double> sumDists = new HashMap<>();
        Map<String, Double> sumSqDists = new HashMap<>();
        Map<String, Integer> counts = new HashMap<>();
        Map<String, Integer> bestNforLabel = new HashMap<>();
        Map<String, Double> bestScoreForLabel = new HashMap<>();

        for (Map.Entry<String, List<RecognizeResultExtended>> e : byLabel.entrySet()) {
            String label = e.getKey();
            double ss = 0.0, sd = 0.0, ssq = 0.0;
            int c = 0;
            double bestScore = -Double.MAX_VALUE; int bestN = -1;
            for (RecognizeResultExtended r : e.getValue()) {
                ss += r.score;
                sd += r.distance;
                ssq += r.distance * r.distance;
                c++;
                if (r.score > bestScore) { bestScore = r.score; bestN = r.N; }
            }
            sumScores.put(label, ss);
            sumDists.put(label, sd);
            sumSqDists.put(label, ssq);
            counts.put(label, c);
            bestNforLabel.put(label, bestN);
            bestScoreForLabel.put(label, bestScore);
        }

        // build sorted map of labels by decreasing sumScores
        Map<String, Double> sortedByScore = sumScores.entrySet().stream()
                .sorted((a,b) -> Double.compare(b.getValue(), a.getValue()))
                .collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue,
                        (x,y)->x, LinkedHashMap::new));

        // choose winner
        String winner = sortedByScore.entrySet().iterator().next().getKey();
        double winnerSumScore = sortedByScore.get(winner);
        int winnerCount = counts.get(winner);
        double winnerMeanDist = sumDists.get(winner) / winnerCount;
        double winnerMeanSq = sumSqDists.get(winner) / winnerCount;
        double winnerStd = Math.sqrt(Math.max(0.0, winnerMeanSq - winnerMeanDist * winnerMeanDist));

        // recommended N = the N where winner got its best individual score
        int recommendedN = bestNforLabel.get(winner);

        // prepare per-label maps for reporting
        Map<String, Double> labelMeanDist = new HashMap<>();
        Map<String, Integer> labelCounts = new HashMap<>();
        for (String label : sumScores.keySet()) {
            int c = counts.get(label);
            labelCounts.put(label, c);
            labelMeanDist.put(label, sumDists.get(label) / c);
        }

        double confidence = winnerSumScore / totalSumScore; // fraction of weight carried by winner
        double frequencyFrac = (double) winnerCount / (double) all.size();

        AggregationDecision dec = new AggregationDecision(winner, winnerSumScore, confidence, winnerCount, frequencyFrac, winnerMeanDist, winnerStd, recommendedN);
        dec.labelSumScores = sortedByScore;
        dec.labelCounts = labelCounts;
        dec.labelMeanDist = labelMeanDist;
        dec.totalSumScore = totalSumScore;
        return dec;
    }

    // Holder classes for aggregate decision and per-N result
    private static class RecognizeResultExtended {
        String label;
        double distance;
        double score;
        int N;
        RecognizeResultExtended(String label, double distance, double score, int N) {
            this.label = label; this.distance = distance; this.score = score; this.N = N;
        }
    }

    private static class AggregationDecision {
        String chosenLabel;
        double aggregatedScore; // sum of scores for chosen label
        double confidence; // aggregatedScore / totalSumScore
        int labelCount; // number of Ns where label was top
        double frequencyFraction; // labelCount / numNs
        double meanDist;
        double stdDist;
        int recommendedN; // N where this label achieved best individual score
        // additional reporting
        Map<String, Double> labelSumScores = new LinkedHashMap<>();
        Map<String, Integer> labelCounts = new HashMap<>();
        Map<String, Double> labelMeanDist = new HashMap<>();
        double totalSumScore;

        AggregationDecision(String chosenLabel, double aggregatedScore, double confidence, int labelCount, double frequencyFraction, double meanDist, double stdDist, int recommendedN) {
            this.chosenLabel = chosenLabel;
            this.aggregatedScore = aggregatedScore;
            this.confidence = confidence;
            this.labelCount = labelCount;
            this.frequencyFraction = frequencyFraction;
            this.meanDist = meanDist;
            this.stdDist = stdDist;
            this.recommendedN = recommendedN;
        }
    }

    // ----------------- Small helper classes -----------------
    private static class Point {
        double x, y;
        Point(double x, double y) { this.x = x; this.y = y; }
    }

    private static class RecognizeResult {
        String bestLabel;
        double bestDistance;
        double similarityScore;
        List<Point> resampledPoints;
        List<Point> boundaryPoints;
        RecognizeResult(String label, double dist, double sim, List<Point> res, List<Point> boundary) {
            this.bestLabel = label; this.bestDistance = dist; this.similarityScore = sim;
            this.resampledPoints = res; this.boundaryPoints = boundary;
        }
    }
}
