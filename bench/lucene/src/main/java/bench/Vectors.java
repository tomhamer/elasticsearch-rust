package bench;

import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.FilterCodec;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsFormat;
import org.apache.lucene.codecs.perfield.PerFieldKnnVectorsFormat;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StoredField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.Field;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.MMapDirectory;
import org.apache.lucene.index.StoredFields;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * {@code vectors}: HNSW build + search with Lucene's
 * {@code Lucene99HnswVectorsFormat(maxConn = 16, beamWidth = 100)} over the
 * base vectors of {@code vectors.f32}, EUCLIDEAN similarity.
 *
 * Lucene has no separate {@code ef} search parameter; {@code KnnFloatVectorQuery}
 * collects {@code k} candidates per segment, so the sweep sets k = ef
 * (50/100/200) and truncates to 10 before computing recall@10 against an
 * exact brute-force scan over the same float arrays.  The index is forced to
 * a single segment ({@code forceMerge(1)}) so a query walks one graph, like
 * the rustic benchmark.  Build time includes the force-merge.
 */
public final class Vectors {
    private Vectors() {}

    private static final String FIELD = "vec";

    public static void run(Common.Args args, Path results) throws Exception {
        Path vectorsPath = Path.of(args.require("vectors"));
        Path out = Path.of(args.get("out", "../indexes/lucene-vec"));
        int dim = args.getInt("dim", 128);
        int nQueries = args.getInt("queries", 20);
        String similarityName=args.get("similarity","l2");
        VectorSimilarityFunction similarity=switch(similarityName) { case "l2" -> VectorSimilarityFunction.EUCLIDEAN; case "cosine" -> VectorSimilarityFunction.COSINE; case "es-cosine" -> VectorSimilarityFunction.DOT_PRODUCT; default -> throw new IllegalArgumentException("unsupported similarity"); };
        String vectorFormat = args.get("vector-format", "lucene99");
        KnnVectorsFormat nativeFormat = switch (vectorFormat) {
            case "lucene99" -> new Lucene99HnswVectorsFormat(args.getInt("m",16), args.getInt("ef-construction",100));
            case "es93" -> EsNative.format(args.getInt("m",16), args.getInt("ef-construction",100));
            default -> throw new IllegalArgumentException("unsupported vector format");
        };
        int k = args.getInt("k", 10);
        int rounds = args.getInt("rounds", 20);
        int warmupRounds = args.getInt("warmup-rounds", 5);
        int m = args.getInt("m", 16);
        int efConstruction = args.getInt("ef-construction", 100);
        int[] efs = Arrays.stream(args.get("ef", "50,100,200").split(",")).mapToInt(s -> Integer.parseInt(s.trim())).toArray();

        // ---- load ----
        byte[] bytes = Files.readAllBytes(vectorsPath);
        if (bytes.length % (4L * dim) != 0) throw new IllegalArgumentException("file size not a multiple of dim*4");
        int total = (int) (bytes.length / (4L * dim));
        int n = total - nQueries;
        ByteBuffer bb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
        float[][] base = new float[n][dim];
        for (int i = 0; i < n; i++) for (int j = 0; j < dim; j++) base[i][j] = bb.getFloat();
        float[][] queries = new float[nQueries][dim];
        for (int i = 0; i < nQueries; i++) for (int j = 0; j < dim; j++) queries[i][j] = bb.getFloat();
        System.err.printf("[vectors] %d base vectors, %d queries, dim %d%n", n, nQueries, dim);

        // ---- build ----
        if (Files.exists(out)) {
            try (var s = Files.walk(out)) {
                s.sorted(Comparator.reverseOrder()).forEach(p -> {
                    try {
                        Files.delete(p);
                    } catch (IOException e) {
                        throw new RuntimeException(e);
                    }
                });
            }
        }
        Files.createDirectories(out);
        Codec base_ = Codec.getDefault();
        Codec codec = new FilterCodec(base_.getName(), base_) {
            @Override
            public KnnVectorsFormat knnVectorsFormat() {
                return new PerFieldKnnVectorsFormat() {
                    @Override
                    public KnnVectorsFormat getKnnVectorsFormatForField(String field) {
                        return nativeFormat;
                    }
                };
            }
        };
        long t0 = System.nanoTime();
        try (MMapDirectory dir = new MMapDirectory(out)) {
            IndexWriterConfig cfg = new IndexWriterConfig(Common.analyzer())
                    .setOpenMode(IndexWriterConfig.OpenMode.CREATE)
                    .setRAMBufferSizeMB(1024)
                    .setCodec(codec)
                    .setUseCompoundFile(false);
            try (IndexWriter w = new IndexWriter(dir, cfg)) {
                for (int i = 0; i < n; i++) {
                    Document d = new Document();
                    d.add(new StoredField("id", i));
                    d.add(new StringField("tag", "t" + (i % 10), Field.Store.NO));
                    d.add(new KnnFloatVectorField(FIELD, base[i], similarity));
                    w.addDocument(d);
                    if ((i + 1) % 50_000 == 0) {
                        System.err.printf("[vectors]   %d vectors, %.1fs%n", i + 1, (System.nanoTime() - t0) / 1e9);
                    }
                }
                w.forceMerge(1);
                w.commit();
            }
        }
        double buildS = (System.nanoTime() - t0) / 1e9;
        long graphBytes = Common.dirSize(out);
        System.err.printf("[vectors] built in %.2fs, %.1f MB on disk%n", buildS, graphBytes / 1e6);

        // ---- ground truth ----
        t0 = System.nanoTime();
        List<Set<Integer>> truth = new ArrayList<>();
        for (float[] q : queries) truth.add(bruteForce(base, q, k, similarity));
        double bruteQps = nQueries / ((System.nanoTime() - t0) / 1e9);
        System.err.printf("[vectors] brute force: %.1f QPS%n", bruteQps);

        // ---- search ----
        List<Map<String, Object>> runs = new ArrayList<>();
        try (MMapDirectory dir = new MMapDirectory(out); DirectoryReader reader = DirectoryReader.open(dir)) {
            IndexSearcher searcher = new IndexSearcher(reader);
            StoredFields stored = searcher.storedFields();
            for (int ef : efs) {
                for (int w = 0; w < warmupRounds; w++) { // untimed warm-up (JIT)
                    for (float[] q : queries) searcher.search(new KnnFloatVectorQuery(FIELD, q, ef), k);
                }
                List<Long> lat = new ArrayList<>();
                double recSum = 0;
                long ts = System.nanoTime();
                for (int r = 0; r < rounds; r++) {
                    for (int qi = 0; qi < nQueries; qi++) {
                        long a = System.nanoTime();
                        TopDocs td = searcher.search(new KnnFloatVectorQuery(FIELD, queries[qi], ef), k);
                        lat.add(System.nanoTime() - a);
                        if (r == 0) {
                            Set<Integer> found = new HashSet<>();
                            int lim = Math.min(k, td.scoreDocs.length);
                            for (int i = 0; i < lim; i++) {
                                ScoreDoc sd = td.scoreDocs[i];
                                found.add(stored.document(sd.doc).getField("id").numericValue().intValue());
                            }
                            Set<Integer> t = truth.get(qi);
                            long hit = found.stream().filter(t::contains).count();
                            recSum += t.isEmpty() ? 1.0 : (double) hit / t.size();
                        }
                    }
                }
                double totalS = (System.nanoTime() - ts) / 1e9;
                Map<String, Object> st = Common.latencyStats(lat);
                Map<String, Object> run = new LinkedHashMap<>();
                run.put("ef", ef);
                run.put("qps", lat.size() / totalS);
                run.put("recall_at_k", recSum / nQueries);
                run.put("latency", st);
                runs.add(run);
                System.err.printf("[vectors] ef=%d: %.0f QPS, p50 %.0fus, recall@%d %.4f%n",
                        ef, lat.size() / totalS, (double) st.get("p50_us"), k, recSum / nQueries);
            }
        }

        Map<String, Object> params = new LinkedHashMap<>();
        params.put("vectors", vectorsPath.toString());
        params.put("dim", dim);
        params.put("n", n);
        params.put("queries", nQueries);
        params.put("k", k);
        params.put("m", m);
        params.put("ef_construction", efConstruction);
        params.put("similarity", similarityName);
        params.put("vector_format", vectorFormat);
        params.put("rounds", rounds);
        params.put("warmup_rounds", warmupRounds);
        Map<String, Object> metrics = new LinkedHashMap<>();
        metrics.put("build_s", buildS);
        metrics.put("build_vectors_per_sec", n / buildS);
        metrics.put("graph_bytes", graphBytes);
        metrics.put("brute_force_qps", bruteQps);
        metrics.put("runs", runs);
        Path p = Common.writeResult(results, "vectors", params, metrics);
        System.out.println("wrote " + p);
    }

    /** Exact k nearest neighbours by squared L2 distance (ids, ties by lower id). */
    static Set<Integer> bruteForce(float[][] base, float[] q, int k) {
        return bruteForce(base,q,k,VectorSimilarityFunction.EUCLIDEAN);
    }

    static Set<Integer> bruteForce(float[][] base, float[] q, int k, VectorSimilarityFunction similarity) {
        int n = base.length;
        float[] d = new float[n];
        for (int i = 0; i < n; i++) {
            float s = 0;
            float[] v = base[i];
            for (int j = 0; j < q.length; j++) {
                float t = v[j] - q[j];
                s += t * t;
            }
            d[i] = similarity == VectorSimilarityFunction.COSINE ? -similarity.compare(base[i],q) : s;
        }
        Integer[] idx = new Integer[n];
        for (int i = 0; i < n; i++) idx[i] = i;
        Arrays.sort(idx, Comparator.<Integer>comparingDouble(i -> d[i]).thenComparingInt(i -> i));
        Set<Integer> out = new HashSet<>();
        for (int i = 0; i < Math.min(k, n); i++) out.add(idx[i]);
        return out;
    }
}
