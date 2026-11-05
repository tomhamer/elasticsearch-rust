package bench;

import java.lang.management.ManagementFactory;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * {@code startup}: cold open + first query, fresh JVM per run.
 *
 * The parent spawns {@code java -cp <same classpath> bench.Main startup-child ...}
 * {@code --runs} times (default 5), attempting {@code sudo -n purge} / {@code purge}
 * before each run to drop the OS page cache (needs root on macOS; when that
 * fails the result is labelled warm-page-cache).  The child reports:
 * <ul>
 *   <li>{@code jvm_startup_ms}: JVM start to {@code main()} entry
 *       (RuntimeMXBean start time), reported separately because it is a JVM
 *       cost the Rust systems do not pay;</li>
 *   <li>{@code open_ms}: {@code DirectoryReader.open} + IndexSearcher creation;</li>
 *   <li>{@code first_query_ms}: the first term query of queries.jsonl.</li>
 * </ul>
 * The parent also records total process wall time.
 */
public final class Startup {
    private Startup() {}

    private static boolean tryRun(String... cmd) {
        try {
            Process p = new ProcessBuilder(cmd).redirectErrorStream(true).start();
            p.getInputStream().readAllBytes();
            return p.waitFor() == 0;
        } catch (Exception e) {
            return false;
        }
    }

    /** Returns [purged?, label]. */
    private static Object[] purge() {
        if (tryRun("sudo", "-n", "purge")) return new Object[] {true, "purged via `sudo -n purge`"};
        if (tryRun("purge")) return new Object[] {true, "purged via `purge`"};
        return new Object[] {false, "NOT purged (`purge` needs sudo; numbers are warm-page-cache)"};
    }

    public static void run(Common.Args args, Path results) throws Exception {
        Path index = Path.of(args.require("index"));
        Path queriesPath = Path.of(args.require("queries"));
        int runs = args.getInt("runs", 5);
        boolean noPurge = args.has("no-purge");
        List<Common.Query> queries = Common.loadQueries(queriesPath);
        Common.Query q = queries.stream().filter(x -> "term".equals(x.clazz)).findFirst()
                .orElseThrow(() -> new IllegalStateException("queries.jsonl has no term query"));

        String java = ProcessHandle.current().info().command().orElse("java");
        String cp = System.getProperty("java.class.path");

        List<Double> openMs = new ArrayList<>(), firstMs = new ArrayList<>(), openFirstMs = new ArrayList<>();
        List<Double> jvmMs = new ArrayList<>(), procMs = new ArrayList<>();
        List<Map<String, Object>> purgeLabels = new ArrayList<>();
        long hits = 0;
        for (int i = 0; i < runs; i++) {
            Object[] pr = noPurge ? new Object[] {false, "not attempted (--no-purge)"} : purge();
            Map<String, Object> pl = new LinkedHashMap<>();
            pl.put("run", i);
            pl.put("purged", pr[0]);
            pl.put("label", pr[1]);
            purgeLabels.add(pl);

            long t0 = System.nanoTime();
            // Same JVM flags as the parent (see run.sh / build.gradle.kts).
            Process p = new ProcessBuilder(java, "-Xms4g", "-Xmx4g", "-XX:+UseParallelGC",
                    "--enable-native-access=ALL-UNNAMED", "--add-modules", "jdk.incubator.vector",
                    "-cp", cp, "bench.Main", "startup-child",
                    "--index", index.toString(), "--field", q.field, "--term", q.term)
                    .redirectError(ProcessBuilder.Redirect.INHERIT).start();
            String out = new String(p.getInputStream().readAllBytes(), StandardCharsets.UTF_8);
            int code = p.waitFor();
            double wall = (System.nanoTime() - t0) / 1e6;
            if (code != 0) {
                System.err.println("startup-child failed with exit code " + code);
                System.exit(1);
            }
            var o = com.google.gson.JsonParser.parseString(out.trim()).getAsJsonObject();
            double open = o.get("open_ms").getAsDouble();
            double first = o.get("first_query_ms").getAsDouble();
            double jvm = o.get("jvm_startup_ms").getAsDouble();
            hits = o.get("hits").getAsLong();
            openMs.add(open);
            firstMs.add(first);
            openFirstMs.add(open + first);
            jvmMs.add(jvm);
            procMs.add(wall);
            System.err.printf("[startup] run %d: jvm %.0f ms, open %.2f ms, first query %.2f ms, process wall %.1f ms (%s)%n",
                    i, jvm, open, first, wall, pr[1]);
        }
        boolean anyPurged = purgeLabels.stream().anyMatch(m -> Boolean.TRUE.equals(m.get("purged")));
        String cache = anyPurged ? "purged" : "warm";
        System.out.printf("lucene: open median %.2f ms, open+first-query median %.2f ms, JVM startup median %.0f ms, process wall median %.1f ms [page cache: %s]%n",
                Common.median(openMs), Common.median(openFirstMs), Common.median(jvmMs), Common.median(procMs), cache);

        Map<String, Object> params = new LinkedHashMap<>();
        params.put("index", index.toString());
        params.put("queries", queriesPath.toString());
        params.put("runs", runs);
        params.put("first_query", q.describe());
        Map<String, Object> metrics = new LinkedHashMap<>();
        metrics.put("page_cache", cache);
        metrics.put("purge", purgeLabels);
        metrics.put("open_ms", openMs);
        metrics.put("first_query_ms", firstMs);
        metrics.put("open_plus_first_query_ms", openFirstMs);
        metrics.put("process_wall_ms", procMs);
        metrics.put("jvm_startup_ms", jvmMs);
        metrics.put("open_ms_median", Common.median(openMs));
        metrics.put("open_ms_min", openMs.stream().mapToDouble(Double::doubleValue).min().orElse(0));
        metrics.put("first_query_ms_median", Common.median(firstMs));
        metrics.put("open_plus_first_query_ms_median", Common.median(openFirstMs));
        metrics.put("process_wall_ms_median", Common.median(procMs));
        metrics.put("jvm_startup_ms_median", Common.median(jvmMs));
        metrics.put("first_query_hits", hits);
        Path p = Common.writeResult(results, "startup", params, metrics);
        System.out.println("wrote " + p);
    }

    /** Child: open, run one term query, print JSON to stdout. */
    public static void child(Common.Args args) throws Exception {
        double jvmStartup = System.currentTimeMillis() - ManagementFactory.getRuntimeMXBean().getStartTime();
        Path index = Path.of(args.require("index"));
        Common.Query q = new Common.Query();
        q.id = 0;
        q.clazz = "term";
        q.kind = "term";
        q.field = args.require("field");
        q.term = args.require("term");

        long t0 = System.nanoTime();
        try (Query.Searcher s = new Query.Searcher(index)) {
            double open = (System.nanoTime() - t0) / 1e6;
            long t1 = System.nanoTime();
            Query.Result r = s.run(q);
            double first = (System.nanoTime() - t1) / 1e6;
            System.out.printf(java.util.Locale.ROOT,
                    "{\"jvm_startup_ms\":%.3f,\"open_ms\":%.4f,\"first_query_ms\":%.4f,\"hits\":%d,\"segments\":%d}%n",
                    jvmStartup, open, first, r.totalHits(), s.numSegments());
        }
    }
}
