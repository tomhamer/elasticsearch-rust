package bench;

import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.FilterCodec;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsFormat;
import org.apache.lucene.codecs.perfield.PerFieldKnnVectorsFormat;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.DoubleField;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.NumericDocValues;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TopScoreDocCollectorManager;
import org.apache.lucene.search.similarities.BM25Similarity;
import org.apache.lucene.store.MMapDirectory;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * {@code hybrid}: BM25 text + k-NN in one request; same workload as {@code compare hybrid}
 * (see {@code bench/compare/src/hybrid.rs}).
 *
 * <ul>
 *   <li>Index: {@code body} TextField (custom analyzer, BM25(1.2, 0.75)), {@code price}
 *       {@link DoubleField} (points + doc values), {@code id} doc values, {@code vec}
 *       {@link KnnFloatVectorField} EUCLIDEAN with Lucene99Hnsw(m=16, beamWidth=100);
 *       single indexing thread, forceMerge(1).</li>
 *   <li>{@code sum}: {@code BooleanQuery(SHOULD text, SHOULD knn)} where the knn clause is
 *       {@link TopKKnnQuery}: {@code KnnFloatVectorQuery(k = num_candidates = 100)} whose merged
 *       result is truncated to the global top k = 10 (ES's {@code knn} section: per-shard
 *       num_candidates, global top k, then {@code bool.should} with the query).</li>
 *   <li>{@code rrf}: text top-100 + {@code KnnFloatVectorQuery(k=100)} fused in Java with
 *       RRF (rank_constant 60, window 100), top 10.</li>
 *   <li>Filter: {@code DoubleField.newRangeQuery(price, -inf, q)} as FILTER clause of the text
 *       query (BooleanQuery MUST text + FILTER range) and as the knn {@code filter}.</li>
 *   <li>Query cache disabled ({@code setQueryCache(null)}) so repeated filters are not cached
 *       across rounds (rustic has no query cache). Exact total hits (TopScoreDocCollectorManager
 *       threshold MAX), as in the query benchmark.</li>
 * </ul>
 */
public final class Hybrid {
    private Hybrid() {}

    private static final String FIELD = "vec";
    static final float RANK_CONSTANT = 60f;

    /** KnnFloatVectorQuery with beam {@code numCandidates} whose merged result keeps only the top {@code topK}. */
    static final class TopKKnnQuery extends KnnFloatVectorQuery {
        private final int topK;

        TopKKnnQuery(String field, float[] target, int numCandidates, Query filter, int topK) {
            super(field, target, numCandidates, filter);
            this.topK = topK;
        }

        @Override
        protected TopDocs mergeLeafResults(TopDocs[] perLeafResults) {
            return TopDocs.merge(topK, perLeafResults);
        }

        @Override
        public boolean equals(Object o) {
            return super.equals(o) && ((TopKKnnQuery) o).topK == topK;
        }

        @Override
        public int hashCode() {
            return 31 * super.hashCode() + topK;
        }
    }

    static List<Common.Query> pairText(List<Common.Query> all, int n) {
        List<Common.Query> or2 = all.stream().filter(q -> q.clazz.equals("or2")).toList();
        List<Common.Query> term = all.stream().filter(q -> q.clazz.equals("term")).toList();
        List<Common.Query> out = new ArrayList<>();
        for (int i = 0; i < n; i++) out.add(i % 2 == 0 ? or2.get((i / 2) % or2.size()) : term.get((i / 2) % term.size()));
        return out;
    }

    static Query textQuery(Common.Query q, Double hi) {
        Query t;
        if (q.kind.equals("term")) {
            t = new TermQuery(new Term("body", q.term));
        } else if (q.kind.equals("or")) {
            BooleanQuery.Builder b = new BooleanQuery.Builder();
            for (String s : q.terms) b.add(new TermQuery(new Term("body", s)), BooleanClause.Occur.SHOULD);
            t = b.build();
        } else {
            throw new IllegalArgumentException("unsupported kind " + q.kind);
        }
        if (hi == null) return t;
        return new BooleanQuery.Builder()
                .add(t, BooleanClause.Occur.MUST)
                .add(priceFilter(hi), BooleanClause.Occur.FILTER)
                .build();
    }

    static Query priceFilter(double hi) {
        return DoubleField.newRangeQuery("price", Double.NEGATIVE_INFINITY, hi);
    }

    interface Exec { long[] run(int qi) throws IOException; }

    public static void run(Common.Args args, Path results) throws Exception {
        Path corpus = Path.of(args.require("corpus"));
        Path queriesPath = Path.of(args.require("queries"));
        Path vectorsPath = Path.of(args.require("vectors"));
        Path out = Path.of(args.get("out", "../indexes/lucene-hybrid"));
        int dim = args.getInt("dim", 128);
        int vecQueries = args.getInt("vec-queries", 100);
        int nQ = args.getInt("n", 100);
        int k = args.getInt("k", 10);
        int nc = args.getInt("num-candidates", 100);
        int window = args.getInt("window", 100);
        int rounds = args.getInt("rounds", 10);
        int warmupRounds = args.getInt("warmup-rounds", 3);
        int m = args.getInt("m", 16);
        int efConstruction = args.getInt("ef-construction", 100);
        boolean reuse = args.has("reuse");

        byte[] bytes = Files.readAllBytes(vectorsPath);
        int total = (int) (bytes.length / (4L * dim));
        int n = total - vecQueries;
        ByteBuffer bb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
        float[][] base = new float[n][dim];
        for (int i = 0; i < n; i++) for (int j = 0; j < dim; j++) base[i][j] = bb.getFloat();
        float[][] qv = new float[vecQueries][dim];
        for (int i = 0; i < vecQueries; i++) for (int j = 0; j < dim; j++) qv[i][j] = bb.getFloat();
        bytes = null;
        List<Common.Query> texts = pairText(Common.loadQueries(queriesPath), nQ);

        double[] prices = new double[n];
        Double buildS = null;
        boolean reused = reuse && Files.exists(out.resolve("hybrid.done"));
        if (reused) {
            int[] c = {0};
            Common.forEachDoc(corpus, d -> prices[c[0]++] = d.price());
            System.err.println("[hybrid] reusing index at " + out);
        } else {
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
                        .setSimilarity(new BM25Similarity(1.2f, 0.75f))
                        .setRAMBufferSizeMB(1024).setCodec(codec).setUseCompoundFile(false);
                try (IndexWriter w = new IndexWriter(dir, cfg)) {
                    int[] c = {0};
                    Common.forEachDoc(corpus, d -> {
                        int i = c[0]++;
                        prices[i] = d.price();
                        Document doc = new Document();
                        doc.add(new NumericDocValuesField("id", d.id()));
                        doc.add(new TextField("body", d.body(), Field.Store.NO));
                        doc.add(new DoubleField("price", d.price(), Field.Store.NO));
                        doc.add(new KnnFloatVectorField(FIELD, base[i], VectorSimilarityFunction.EUCLIDEAN));
                        try { w.addDocument(doc); } catch (IOException e) { throw new RuntimeException(e); }
                    });
                    if (c[0] != n) throw new IllegalStateException("corpus docs " + c[0] + " != base vectors " + n);
                    w.forceMerge(1);
                    w.commit();
                }
            }
            buildS = (System.nanoTime() - t0) / 1e9;
            Files.writeString(out.resolve("hybrid.done"), "ok");
            System.err.printf("[hybrid] built in %.2fs%n", buildS);
        }

        double[] sorted = prices.clone();
        Arrays.sort(sorted);
        double q10 = Common.percentileSorted(sorted, 10), q50 = Common.percentileSorted(sorted, 50);
        String[] fnames = {"none", "price_p10", "price_p50"};
        Double[] his = {null, q10, q50};

        List<Map<String, Object>> runs = new ArrayList<>();
        List<Map<String, Object>> filterMeta = new ArrayList<>();
        try (MMapDirectory dir = new MMapDirectory(out); DirectoryReader reader = DirectoryReader.open(dir)) {
            IndexSearcher searcher = new IndexSearcher(reader);
            searcher.setSimilarity(new BM25Similarity(1.2f, 0.75f));
            searcher.setQueryCache(null);
            // doc -> corpus id
            long[] idOf = new long[reader.maxDoc()];
            for (LeafReaderContext ctx : reader.leaves()) {
                NumericDocValues dv = ctx.reader().getNumericDocValues("id");
                for (int d = dv.nextDoc(); d != NumericDocValues.NO_MORE_DOCS; d = dv.nextDoc()) idOf[ctx.docBase + d] = dv.longValue();
            }
            for (int fi = 0; fi < fnames.length; fi++) {
                String fname = fnames[fi];
                Double hi = his[fi];
                long cnt = 0;
                for (double p : prices) if (hi == null || p <= hi) cnt++;
                double selectivity = (double) cnt / n;
                Map<String, Object> fm = new LinkedHashMap<>();
                fm.put("name", fname);
                fm.put("price_lte", hi);
                fm.put("selectivity", selectivity);
                filterMeta.add(fm);

                Query filter = hi == null ? null : priceFilter(hi);
                Query[] tqs = new Query[nQ];
                for (int qi = 0; qi < nQ; qi++) tqs[qi] = textQuery(texts.get(qi), hi);

                // kNN leg recall@k vs brute force (untimed).
                double rec = 0;
                for (int qi = 0; qi < nQ; qi++) {
                    final double h = hi == null ? Double.POSITIVE_INFINITY : hi;
                    Set<Integer> truth = VectorsFiltered.bruteForce(base, qv[qi], k, i -> prices[i] <= h);
                    TopDocs td = searcher.search(new TopKKnnQuery(FIELD, qv[qi], nc, filter, k), k);
                    long found = Arrays.stream(td.scoreDocs).filter(sd -> truth.contains((int) idOf[sd.doc])).count();
                    rec += truth.isEmpty() ? 1.0 : (double) found / truth.size();
                }
                double knnRecall = rec / nQ;

                Exec sum = qi -> {
                    Query q = new BooleanQuery.Builder()
                            .add(tqs[qi], BooleanClause.Occur.SHOULD)
                            .add(new TopKKnnQuery(FIELD, qv[qi], nc, filter, k), BooleanClause.Occur.SHOULD)
                            .build();
                    TopDocs td = searcher.search(q, new TopScoreDocCollectorManager(k, Integer.MAX_VALUE));
                    return toIds(td.scoreDocs, idOf, k);
                };
                Exec rrf = qi -> {
                    TopDocs text = searcher.search(tqs[qi], new TopScoreDocCollectorManager(window, Integer.MAX_VALUE));
                    TopDocs knn = searcher.search(new KnnFloatVectorQuery(FIELD, qv[qi], Math.max(nc, window), filter), window);
                    HashMap<Integer, Float> acc = new HashMap<>();
                    for (ScoreDoc[] list : new ScoreDoc[][]{text.scoreDocs, knn.scoreDocs}) {
                        for (int r = 0; r < Math.min(window, list.length); r++)
                            acc.merge(list[r].doc, 1f / (RANK_CONSTANT + r + 1f), Float::sum);
                    }
                    List<Map.Entry<Integer, Float>> e = new ArrayList<>(acc.entrySet());
                    e.sort((a, b) -> {
                        int c = Float.compare(b.getValue(), a.getValue());
                        return c != 0 ? c : Integer.compare(a.getKey(), b.getKey());
                    });
                    long[] ids = new long[Math.min(k, e.size())];
                    for (int i = 0; i < ids.length; i++) ids[i] = idOf[e.get(i).getKey()];
                    return ids;
                };
                for (String mode : new String[]{"sum", "rrf"}) {
                    Exec ex = mode.equals("sum") ? sum : rrf;
                    for (int w = 0; w < warmupRounds; w++) for (int qi = 0; qi < nQ; qi++) ex.run(qi);
                    List<Long> lat = new ArrayList<>();
                    List<List<Long>> first = new ArrayList<>();
                    long ts = System.nanoTime();
                    for (int r = 0; r < rounds; r++) {
                        for (int qi = 0; qi < nQ; qi++) {
                            long a = System.nanoTime();
                            long[] ids = ex.run(qi);
                            lat.add(System.nanoTime() - a);
                            if (r == 0) first.add(Arrays.stream(ids).boxed().toList());
                        }
                    }
                    double totalS = (System.nanoTime() - ts) / 1e9;
                    Map<String, Object> st = Common.latencyStats(lat);
                    Map<String, Object> run = new LinkedHashMap<>();
                    run.put("mode", mode);
                    run.put("filter", fname);
                    run.put("selectivity", selectivity);
                    run.put("qps", lat.size() / totalS);
                    run.put("latency", st);
                    run.put("knn_recall_at_k", knnRecall);
                    run.put("top_ids", first);
                    runs.add(run);
                    System.err.printf("[hybrid] lucene %s filter=%s (%.0f %%): %.0f QPS p50 %.0fus p99 %.0fus; knn recall@%d %.4f%n",
                            mode, fname, selectivity * 100, lat.size() / totalS, (double) st.get("p50_us"),
                            (double) st.get("p99_us"), k, knnRecall);
                }
            }
        }

        Map<String, Object> params = new LinkedHashMap<>();
        params.put("corpus", corpus.toString());
        params.put("vectors", vectorsPath.toString());
        params.put("dim", dim);
        params.put("n_docs", n);
        params.put("n_queries", nQ);
        params.put("k", k);
        params.put("num_candidates", nc);
        params.put("rrf_window", window);
        params.put("rank_constant", RANK_CONSTANT);
        params.put("m", m);
        params.put("ef_construction", efConstruction);
        params.put("similarity", "EUCLIDEAN (score 1/(1+d^2))");
        params.put("rounds", rounds);
        params.put("warmup_rounds", warmupRounds);
        params.put("index_reused", reused);
        params.put("sum_path", "BooleanQuery(SHOULD text, SHOULD KnnFloatVectorQuery(k=num_candidates, filter) truncated to top k)");
        params.put("rrf_path", "search(text, 100) + KnnFloatVectorQuery(k=100) + RRF in Java");
        params.put("filter_path", "DoubleField.newRangeQuery(price): FILTER clause on text, KnnFloatVectorQuery filter");
        params.put("query_cache", "disabled");
        Map<String, Object> metrics = new LinkedHashMap<>();
        metrics.put("build_s", buildS);
        metrics.put("filters", filterMeta);
        metrics.put("runs", runs);
        Path p = Common.writeResult(results, "hybrid", params, metrics);
        System.out.println("wrote " + p);
    }

    static long[] toIds(ScoreDoc[] sds, long[] idOf, int k) {
        long[] out = new long[Math.min(k, sds.length)];
        for (int i = 0; i < out.length; i++) out[i] = idOf[sds[i].doc];
        return out;
    }
}
