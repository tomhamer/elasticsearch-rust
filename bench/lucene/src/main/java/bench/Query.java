package bench;

import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.PhraseQuery;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TopScoreDocCollectorManager;
import org.apache.lucene.search.similarities.BM25Similarity;
import org.apache.lucene.store.MMapDirectory;
import org.apache.lucene.util.NumericUtils;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

/**
 * {@code query}: warm query latency per class (p50/p99 in µs, pooled over
 * all queries x iterations of the class) plus per-query exact hit counts.
 * One untimed pass over the whole query set runs first ({@code --global-warmup},
 * default 1) so JIT compilation does not land in the timed samples; the Rust
 * harness does the same pass for symmetry.
 *
 * Queries: TermQuery, BooleanQuery (MUST / SHOULD), PhraseQuery,
 * NumericDocValuesField.newSlowRangeQuery (inclusive), collected with
 * TopScoreDocCollectorManager(10, Integer.MAX_VALUE) so totalHits is exact.
 * The IndexSearcher has no executor, so segments are searched sequentially
 * like in the other two harnesses.  BM25 k1 = 1.2, b = 0.75.
 */
public final class Query {
    private Query() {}

    /** Result of one execution. */
    public record Result(long totalHits, float topScore) {}

    /** An opened index. */
    public static final class Searcher implements AutoCloseable {
        final MMapDirectory dir;
        final DirectoryReader reader;
        final IndexSearcher searcher;

        public Searcher(Path index) throws IOException {
            dir = new MMapDirectory(index);
            reader = DirectoryReader.open(dir);
            searcher = new IndexSearcher(reader);
            searcher.setSimilarity(new BM25Similarity(1.2f, 0.75f));
        }

        public int numSegments() {
            return reader.leaves().size();
        }

        public org.apache.lucene.search.Query build(Common.Query q) {
            switch (q.kind) {
                case "term":
                    return new TermQuery(new Term(q.field, q.term));
                case "and": {
                    BooleanQuery.Builder b = new BooleanQuery.Builder();
                    for (String t : q.terms) b.add(new TermQuery(new Term(q.field, t)), BooleanClause.Occur.MUST);
                    return b.build();
                }
                case "or": {
                    BooleanQuery.Builder b = new BooleanQuery.Builder();
                    for (String t : q.terms) b.add(new TermQuery(new Term(q.field, t)), BooleanClause.Occur.SHOULD);
                    return b.build();
                }
                case "phrase":
                    return new PhraseQuery(q.field, q.terms.toArray(new String[0]));
                case "range": {
                    double lo = q.lower == null ? -Double.MAX_VALUE : q.lower;
                    double hi = q.upper == null ? Double.MAX_VALUE : q.upper;
                    return NumericDocValuesField.newSlowRangeQuery(q.field,
                            NumericUtils.doubleToSortableLong(lo), NumericUtils.doubleToSortableLong(hi));
                }
                default:
                    throw new IllegalArgumentException("unknown query kind " + q.kind);
            }
        }

        public Result run(Common.Query q) throws IOException {
            org.apache.lucene.search.Query lq = build(q);
            TopDocs td = searcher.search(lq, new TopScoreDocCollectorManager(Common.TOP_K, Integer.MAX_VALUE));
            float top = td.scoreDocs.length > 0 ? td.scoreDocs[0].score : 0f;
            return new Result(td.totalHits.value(), top);
        }

        @Override
        public void close() throws IOException {
            reader.close();
            dir.close();
        }
    }

    public static void run(Common.Args args, Path results) throws Exception {
        Path index = Path.of(args.require("index"));
        Path queriesPath = Path.of(args.require("queries"));
        int warmup = args.getInt("warmup", 2);
        int iters = args.getInt("iters", 5);
        String schedule=args.get("schedule","grouped");
        if (!schedule.equals("grouped") && !schedule.equals("mixed")) throw new IllegalArgumentException("unknown schedule");
        int globalWarmup = args.getInt("global-warmup", 1);
        List<Common.Query> queries = Common.loadQueries(queriesPath);

        long t0 = System.nanoTime();
        try (Searcher s = new Searcher(index)) {
            System.err.printf("[query] lucene opened %d segment(s) in %.1f ms; %d queries, warmup %d, iters %d%n",
                    s.numSegments(), (System.nanoTime() - t0) / 1e6, queries.size(), warmup, iters);

            // Untimed pass(es) over the whole query set first (JIT / page cache),
            // then the per-query warm-up + timed iterations.
            for (int g = 0; g < globalWarmup; g++) for (Common.Query q : queries) s.run(q);

            Map<String, List<Long>> classLat = new TreeMap<>();
            Map<String, Long> classHits = new TreeMap<>();
            Map<String, Integer> classN = new TreeMap<>();
            List<Map<String, Object>> perQuery = new ArrayList<>();
            long benchStart = System.nanoTime();
            List<List<Long>> mixedLat=new ArrayList<>();
            List<Result> mixedLast=new ArrayList<>();
            for (int qi=0;qi<queries.size();qi++) { mixedLat.add(new ArrayList<>()); mixedLast.add(new Result(0,0f)); }
            if (schedule.equals("mixed")) {
                for (int r=0;r<warmup+iters;r++) for (int qi=0;qi<queries.size();qi++) {
                    long start=System.nanoTime(); Result result=s.run(queries.get(qi)); long elapsed=System.nanoTime()-start;
                    if (r>=warmup) { mixedLat.get(qi).add(elapsed); mixedLast.set(qi,result); }
                }
            }
            for (int qi=0;qi<queries.size();qi++) {
                Common.Query q=queries.get(qi);
                for (int i = 0; i < (schedule.equals("grouped")?warmup:0); i++) s.run(q);
                List<Long> lat = new ArrayList<>(iters);
                Result last = new Result(0, 0f);
                for (int i = 0; i < (schedule.equals("grouped")?iters:0); i++) {
                    long a = System.nanoTime();
                    last = s.run(q);
                    lat.add(System.nanoTime() - a);
                }
                if (schedule.equals("mixed")) { lat=mixedLat.get(qi); last=mixedLast.get(qi); }
                Map<String, Object> st = Common.latencyStats(lat);
                classLat.computeIfAbsent(q.clazz, k -> new ArrayList<>()).addAll(lat);
                classHits.merge(q.clazz, last.totalHits(), Long::sum);
                classN.merge(q.clazz, 1, Integer::sum);
                Map<String, Object> pq = new LinkedHashMap<>();
                pq.put("id", q.id);
                pq.put("class", q.clazz);
                pq.put("query", q.describe());
                pq.put("hits", last.totalHits());
                pq.put("top_score", last.topScore());
                pq.put("p50_us", st.get("p50_us"));
                pq.put("min_us", st.get("min_us"));
                perQuery.add(pq);
            }
            double benchWall = (System.nanoTime() - benchStart) / 1e9;

            Map<String, Object> classes = new LinkedHashMap<>();
            System.out.printf("%-8s %5s %12s %12s %12s %14s%n", "class", "n", "p50_us", "p99_us", "mean_us", "total_hits");
            for (var e : classLat.entrySet()) {
                Map<String, Object> st = Common.latencyStats(e.getValue());
                System.out.printf("%-8s %5d %12.1f %12.1f %12.1f %14d%n", e.getKey(), classN.get(e.getKey()),
                        (double) st.get("p50_us"), (double) st.get("p99_us"), (double) st.get("mean_us"),
                        classHits.get(e.getKey()));
                Map<String, Object> c = new LinkedHashMap<>();
                c.put("n_queries", classN.get(e.getKey()));
                c.put("total_hits", classHits.get(e.getKey()));
                c.put("latency", st);
                classes.put(e.getKey(), c);
            }

            Map<String, Object> params = new LinkedHashMap<>();
            params.put("index", index.toString());
            params.put("queries", queriesPath.toString());
            params.put("schedule",schedule);
            params.put("warmup", warmup);
            params.put("iters", iters);
            params.put("global_warmup", globalWarmup);
            params.put("top_k", Common.TOP_K);
            Map<String, Object> metrics = new LinkedHashMap<>();
            metrics.put("segments", s.numSegments());
            metrics.put("bench_wall_s", benchWall);
            metrics.put("classes", classes);
            metrics.put("per_query", perQuery);
            Path p = Common.writeResult(results, "query", params, metrics);
            System.out.println("wrote " + p);
            System.out.println("run `compare report --inputs results/*.json --out results/RESULTS.md` to cross-check hit counts");
        }
    }
}
