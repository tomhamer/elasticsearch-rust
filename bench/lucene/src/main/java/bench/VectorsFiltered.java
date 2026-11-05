package bench;

import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.FilterCodec;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsFormat;
import org.apache.lucene.codecs.perfield.PerFieldKnnVectorsFormat;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.IntField;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StoredField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.StoredFields;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.MMapDirectory;

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
 * {@code vectors-filtered}: filtered k-NN with {@code KnnFloatVectorQuery(field, q, k = ef, filter)}.
 *
 * Every vector i gets {@code tag = i % 100} and {@code bucket = i % 10} as {@link IntField}
 * (points + doc values, {@code Store.NO}).  Filters per query qi: {@code tag == qi % 100} (1 %),
 * {@code bucket == qi % 10} (10 %), {@code bucket < 5} (50 %), built with
 * {@code IntField.newExactQuery} / {@code newRangeQuery}.  Lucene decides exact vs filtered
 * HNSW itself (exact when the filter cost is at most k, or when graph search visits more
 * nodes than the filter cost).  Same codec / HNSW params / forceMerge(1) as {@link Vectors}.
 * Ground truth: brute force among matching vectors.
 */
public final class VectorsFiltered {
    private VectorsFiltered() {}

    private static final String FIELD = "vec";

    record Spec(String name, double selectivity, String expr, String field, int[] lo, int[] hi) {}

    public static void run(Common.Args args, Path results) throws Exception {
        Path vectorsPath = Path.of(args.require("vectors"));
        Path out = Path.of(args.get("out", "../indexes/lucene-vec-filtered"));
        int dim = args.getInt("dim", 128);
        int nQueries = args.getInt("queries", 20);
        int k = args.getInt("k", 10);
        int rounds = args.getInt("rounds", 20);
        int warmupRounds = args.getInt("warmup-rounds", 5);
        int m = args.getInt("m", 16);
        int efConstruction = args.getInt("ef-construction", 100);
        int[] efs = Arrays.stream(args.get("ef", "100").split(",")).mapToInt(s -> Integer.parseInt(s.trim())).toArray();

        byte[] bytes = Files.readAllBytes(vectorsPath);
        int total = (int) (bytes.length / (4L * dim));
        int n = total - nQueries;
        ByteBuffer bb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
        float[][] base = new float[n][dim];
        for (int i = 0; i < n; i++) for (int j = 0; j < dim; j++) base[i][j] = bb.getFloat();
        float[][] queries = new float[nQueries][dim];
        for (int i = 0; i < nQueries; i++) for (int j = 0; j < dim; j++) queries[i][j] = bb.getFloat();

        if (Files.exists(out)) {
            try (var s = Files.walk(out)) {
                s.sorted(Comparator.reverseOrder()).forEach(p -> {
                    try { Files.delete(p); } catch (IOException e) { throw new RuntimeException(e); }
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
                        return new Lucene99HnswVectorsFormat(m, efConstruction);
                    }
                };
            }
        };
        long t0 = System.nanoTime();
        try (MMapDirectory dir = new MMapDirectory(out)) {
            IndexWriterConfig cfg = new IndexWriterConfig(Common.analyzer())
                    .setOpenMode(IndexWriterConfig.OpenMode.CREATE)
                    .setRAMBufferSizeMB(1024).setCodec(codec).setUseCompoundFile(false);
            try (IndexWriter w = new IndexWriter(dir, cfg)) {
                for (int i = 0; i < n; i++) {
                    Document d = new Document();
                    d.add(new StoredField("id", i));
                    d.add(new IntField("tag", i % 100, Field.Store.NO));
                    d.add(new IntField("bucket", i % 10, Field.Store.NO));
                    d.add(new KnnFloatVectorField(FIELD, base[i], VectorSimilarityFunction.EUCLIDEAN));
                    w.addDocument(d);
                }
                w.forceMerge(1);
                w.commit();
            }
        }
        double buildS = (System.nanoTime() - t0) / 1e9;
        System.err.printf("[vectors-filtered] built in %.2fs%n", buildS);

        int[] q100 = new int[nQueries], q10 = new int[nQueries], zero = new int[nQueries], four = new int[nQueries];
        for (int i = 0; i < nQueries; i++) { q100[i] = i % 100; q10[i] = i % 10; zero[i] = 0; four[i] = 4; }
        List<Spec> specs = List.of(
                new Spec("tag_eq", 0.01, "tag == qi % 100", "tag", q100, q100),
                new Spec("bucket_eq", 0.10, "bucket == qi % 10", "bucket", q10, q10),
                new Spec("bucket_lt5", 0.50, "bucket < 5", "bucket", zero, four));

        List<Map<String, Object>> runs = new ArrayList<>();
        try (MMapDirectory dir = new MMapDirectory(out); DirectoryReader reader = DirectoryReader.open(dir)) {
            IndexSearcher searcher = new IndexSearcher(reader);
            StoredFields stored = searcher.storedFields();
            for (Spec sp : specs) {
                List<Set<Integer>> truth = new ArrayList<>();
                Query[] filters = new Query[nQueries];
                for (int qi = 0; qi < nQueries; qi++) {
                    int lo = sp.lo[qi], hi = sp.hi[qi];
                    boolean tag = sp.field.equals("tag");
                    truth.add(bruteForce(base, queries[qi], k, i -> { int v = tag ? i % 100 : i % 10; return v >= lo && v <= hi; }));
                    filters[qi] = lo == hi ? IntField.newExactQuery(sp.field, lo) : IntField.newRangeQuery(sp.field, lo, hi);
                }
                for (int ef : efs) {
                    for (int w = 0; w < warmupRounds; w++)
                        for (int qi = 0; qi < nQueries; qi++)
                            searcher.search(new KnnFloatVectorQuery(FIELD, queries[qi], ef, filters[qi]), ef);
                    List<Long> lat = new ArrayList<>();
                    double recSum = 0;
                    long ts = System.nanoTime();
                    for (int r = 0; r < rounds; r++) {
                        for (int qi = 0; qi < nQueries; qi++) {
                            long a = System.nanoTime();
                            TopDocs td = searcher.search(new KnnFloatVectorQuery(FIELD, queries[qi], ef, filters[qi]), ef);
                            lat.add(System.nanoTime() - a);
                            if (r == 0) {
                                Set<Integer> found = new HashSet<>();
                                for (int i = 0; i < Math.min(k, td.scoreDocs.length); i++)
                                    found.add(stored.document(td.scoreDocs[i].doc).getField("id").numericValue().intValue());
                                Set<Integer> t = truth.get(qi);
                                recSum += t.isEmpty() ? 1.0 : (double) found.stream().filter(t::contains).count() / t.size();
                            }
                        }
                    }
                    double totalS = (System.nanoTime() - ts) / 1e9;
                    Map<String, Object> st = Common.latencyStats(lat);
                    Map<String, Object> run = new LinkedHashMap<>();
                    run.put("filter", sp.name);
                    run.put("expr", sp.expr);
                    run.put("selectivity", sp.selectivity);
                    run.put("ef", ef);
                    run.put("qps", lat.size() / totalS);
                    run.put("recall_at_k", recSum / nQueries);
                    run.put("latency", st);
                    runs.add(run);
                    System.err.printf("[vectors-filtered] %s ef=%d: %.0f QPS, p50 %.0fus, recall@%d %.4f%n",
                            sp.name, ef, lat.size() / totalS, (double) st.get("p50_us"), k, recSum / nQueries);
                }
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
        params.put("similarity", "l2");
        params.put("rounds", rounds);
        params.put("warmup_rounds", warmupRounds);
        params.put("path", "KnnFloatVectorQuery(field, q, k=ef, IntField.newExactQuery/newRangeQuery)");
        params.put("payload", "tag = i % 100, bucket = i % 10 (IntField: points + doc values)");
        Map<String, Object> metrics = new LinkedHashMap<>();
        metrics.put("build_s", buildS);
        metrics.put("runs", runs);
        Path p = Common.writeResult(results, "vectors-filtered", params, metrics);
        System.out.println("wrote " + p);
    }

    static Set<Integer> bruteForce(float[][] base, float[] q, int k, java.util.function.IntPredicate accept) {
        int n = base.length;
        List<Integer> ids = new ArrayList<>();
        float[] d = new float[n];
        for (int i = 0; i < n; i++) {
            if (!accept.test(i)) continue;
            float s = 0;
            float[] v = base[i];
            for (int j = 0; j < q.length; j++) { float t = v[j] - q[j]; s += t * t; }
            d[i] = s;
            ids.add(i);
        }
        ids.sort(Comparator.<Integer>comparingDouble(i -> d[i]).thenComparingInt(i -> i));
        return new HashSet<>(ids.subList(0, Math.min(k, ids.size())));
    }
}
