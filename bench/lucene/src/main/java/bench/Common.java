package bench;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.google.gson.stream.JsonReader;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.LowerCaseFilter;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.Tokenizer;
import org.apache.lucene.analysis.util.CharTokenizer;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;

/**
 * Shared helpers: analyzer, corpus/query readers, argument parsing, statistics,
 * machine info and the JSON result envelope (identical to the Rust harness).
 */
public final class Common {
    private Common() {}

    public static final int TOP_K = 10;
    public static final Gson GSON = new GsonBuilder().setPrettyPrinting().serializeSpecialFloatingPointValues().create();

    // ------------------------------------------------------------------ analyzer

    /**
     * Lowercase + split on non-letter-or-digit: the same token stream rustic's
     * {@code standard} analyzer and tantivy's {@code default} tokenizer produce
     * for this corpus.  (Lucene's StandardAnalyzer would also split on
     * punctuation but applies Unicode word-break rules; SimpleAnalyzer splits on
     * digits.  A custom CharTokenizer avoids both differences.)
     */
    public static Analyzer analyzer() {
        return new Analyzer() {
            @Override
            protected TokenStreamComponents createComponents(String fieldName) {
                Tokenizer tok = CharTokenizer.fromTokenCharPredicate(Character::isLetterOrDigit);
                TokenStream ts = new LowerCaseFilter(tok);
                return new TokenStreamComponents(tok, ts);
            }
        };
    }

    // ------------------------------------------------------------------ records

    /** One corpus document (gen.py). */
    public record Doc(long id, String title, String body, double price, long ts, String tag) {}

    /** One query (queries.py). */
    public static final class Query {
        public int id;
        public String clazz; // "class" is reserved; mapped manually below
        public String field;
        public String kind;
        public String term;
        public List<String> terms = new ArrayList<>();
        public Double lower;
        public Double upper;

        public List<String> allTerms() {
            return term != null ? List.of(term) : terms;
        }

        public String describe() {
            switch (kind) {
                case "range":
                    return field + ":[" + lower + " TO " + upper + "]";
                case "phrase":
                    return field + ":\"" + String.join(" ", terms) + "\"";
                case "and":
                    return field + ":(" + String.join(" AND ", terms) + ")";
                case "or":
                    return field + ":(" + String.join(" OR ", terms) + ")";
                default:
                    return field + ":" + String.join(" ", allTerms());
            }
        }

        static Query fromJson(JsonObject o) {
            Query q = new Query();
            q.id = o.get("id").getAsInt();
            q.clazz = o.get("class").getAsString();
            q.field = o.get("field").getAsString();
            q.kind = o.get("kind").getAsString();
            if (o.has("term")) q.term = o.get("term").getAsString();
            if (o.has("terms")) o.getAsJsonArray("terms").forEach(e -> q.terms.add(e.getAsString()));
            if (o.has("lower")) q.lower = o.get("lower").getAsDouble();
            if (o.has("upper")) q.upper = o.get("upper").getAsDouble();
            return q;
        }
    }

    /** Stream the JSONL corpus. Returns the number of docs. */
    public static long forEachDoc(Path corpus, Consumer<Doc> f) throws IOException {
        long n = 0;
        try (BufferedReader br = Files.newBufferedReader(corpus, StandardCharsets.UTF_8)) {
            String line;
            while ((line = br.readLine()) != null) {
                if (line.isBlank()) continue;
                JsonReader jr = new JsonReader(new java.io.StringReader(line));
                JsonObject o = JsonParser.parseReader(jr).getAsJsonObject();
                f.accept(new Doc(
                        o.get("id").getAsLong(),
                        o.get("title").getAsString(),
                        o.get("body").getAsString(),
                        o.get("price").getAsDouble(),
                        o.get("ts").getAsLong(),
                        o.get("tag").getAsString()));
                n++;
            }
        }
        return n;
    }

    public static List<Query> loadQueries(Path path) throws IOException {
        List<Query> out = new ArrayList<>();
        for (String line : Files.readAllLines(path, StandardCharsets.UTF_8)) {
            if (line.isBlank()) continue;
            out.add(Query.fromJson(JsonParser.parseString(line).getAsJsonObject()));
        }
        return out;
    }

    // ------------------------------------------------------------------ args

    /** Minimal {@code --flag value} parser mirroring the Rust harness. */
    public static final class Args {
        public final String subcommand;
        private final Map<String, String> map = new LinkedHashMap<>();
        public final List<String> positional = new ArrayList<>();

        public Args(String[] argv) {
            subcommand = argv.length > 0 ? argv[0] : "";
            for (int i = 1; i < argv.length; i++) {
                String a = argv[i];
                if (a.startsWith("--")) {
                    String name = a.substring(2);
                    int eq = name.indexOf('=');
                    if (eq >= 0) {
                        map.put(name.substring(0, eq), name.substring(eq + 1));
                    } else if (i + 1 < argv.length && !argv[i + 1].startsWith("--")) {
                        map.put(name, argv[++i]);
                    } else {
                        map.put(name, "true");
                    }
                } else {
                    positional.add(a);
                }
            }
        }

        public String get(String name, String def) {
            return map.getOrDefault(name, def);
        }

        public String require(String name) {
            String v = map.get(name);
            if (v == null) {
                System.err.println("missing required argument --" + name);
                System.exit(2);
            }
            return v;
        }

        public int getInt(String name, int def) {
            return map.containsKey(name) ? Integer.parseInt(map.get(name)) : def;
        }

        public boolean has(String name) {
            return map.containsKey(name);
        }
    }

    // ------------------------------------------------------------------ stats

    /** Nearest-rank percentile of a sorted array. */
    public static double percentileSorted(double[] sorted, double p) {
        if (sorted.length == 0) return 0.0;
        int rank = (int) Math.ceil(p / 100.0 * sorted.length);
        rank = Math.max(1, Math.min(sorted.length, rank));
        return sorted[rank - 1];
    }

    /** Latency summary in microseconds, same keys as the Rust {@code LatencyStats}. */
    public static Map<String, Object> latencyStats(List<Long> nanos) {
        double[] us = nanos.stream().mapToDouble(n -> n / 1000.0).sorted().toArray();
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("n", us.length);
        m.put("p50_us", percentileSorted(us, 50));
        m.put("p90_us", percentileSorted(us, 90));
        m.put("p99_us", percentileSorted(us, 99));
        m.put("mean_us", us.length == 0 ? 0.0 : Arrays.stream(us).average().orElse(0));
        m.put("min_us", us.length == 0 ? 0.0 : us[0]);
        m.put("max_us", us.length == 0 ? 0.0 : us[us.length - 1]);
        return m;
    }

    public static double median(List<Double> v) {
        double[] s = v.stream().mapToDouble(Double::doubleValue).sorted().toArray();
        return percentileSorted(s, 50);
    }

    // ------------------------------------------------------------------ machine / results

    private static String sysctl(String name) {
        try {
            Process p = new ProcessBuilder("sysctl", "-n", name).redirectErrorStream(true).start();
            String out = new String(p.getInputStream().readAllBytes(), StandardCharsets.UTF_8).trim();
            p.waitFor();
            return out;
        } catch (Exception e) {
            return "";
        }
    }

    private static String run(String... cmd) {
        try {
            Process p = new ProcessBuilder(cmd).redirectErrorStream(true).start();
            String out = new String(p.getInputStream().readAllBytes(), StandardCharsets.UTF_8).trim();
            p.waitFor();
            return out;
        } catch (Exception e) {
            return "";
        }
    }

    private static String procField(String path, String key) {
        try {
            for (String l : Files.readAllLines(Path.of(path))) {
                int i = l.indexOf(':');
                if (i > 0 && l.substring(0, i).trim().equals(key)) return l.substring(i + 1).trim();
            }
        } catch (Exception e) {
            // not Linux
        }
        return null;
    }

    public static Map<String, Object> machineInfo() {
        Map<String, Object> m = new LinkedHashMap<>();
        String label = System.getenv("BENCH_MACHINE");
        if (procField("/proc/cpuinfo", "model name") != null) {
            m.put("cpu", procField("/proc/cpuinfo", "model name"));
            m.put("logical_cores", Runtime.getRuntime().availableProcessors());
            long cores = 0;
            try {
                cores = Files.readAllLines(Path.of("/proc/cpuinfo")).stream()
                        .filter(l -> l.startsWith("core id")).distinct().count();
            } catch (Exception e) { /* ignore */ }
            m.put("physical_cores", cores);
            String mem = procField("/proc/meminfo", "MemTotal");
            m.put("ram_bytes", mem == null ? 0 : parseLongOr(mem.split("\\s+")[0], 0) * 1024);
            String os = "Linux";
            try {
                for (String l : Files.readAllLines(Path.of("/etc/os-release")))
                    if (l.startsWith("PRETTY_NAME=")) os = l.substring(12).replace("\"", "");
            } catch (Exception e) { /* ignore */ }
            m.put("os", os);
            m.put("arch", "amd64".equals(System.getProperty("os.arch")) ? "x86_64" : System.getProperty("os.arch"));
            m.put("jvm", System.getProperty("java.vm.name") + " " + System.getProperty("java.version"));
            if (label != null && !label.isEmpty()) m.put("label", label);
            return m;
        }
        m.put("cpu", sysctl("machdep.cpu.brand_string"));
        m.put("logical_cores", parseLongOr(sysctl("hw.ncpu"), Runtime.getRuntime().availableProcessors()));
        m.put("physical_cores", parseLongOr(sysctl("hw.physicalcpu"), 0));
        m.put("ram_bytes", parseLongOr(sysctl("hw.memsize"), 0));
        String os = System.getProperty("os.name");
        if (os != null && os.toLowerCase().contains("mac")) {
            os = run("sw_vers", "-productName") + " " + run("sw_vers", "-productVersion");
        }
        m.put("os", os);
        m.put("arch", System.getProperty("os.arch"));
        m.put("jvm", System.getProperty("java.vm.name") + " " + System.getProperty("java.version"));
        if (label != null && !label.isEmpty()) m.put("label", label);
        return m;
    }

    private static long parseLongOr(String s, long def) {
        try {
            return Long.parseLong(s.trim());
        } catch (Exception e) {
            return def;
        }
    }

    /** Write {@code <results>/lucene-<subcommand>.json} in the shared envelope. */
    public static Path writeResult(Path resultsDir, String subcommand, Map<String, Object> params,
                                   Map<String, Object> metrics) throws IOException {
        Map<String, Object> env = new LinkedHashMap<>();
        env.put("system", "lucene");
        env.put("subcommand", subcommand);
        env.put("timestamp_unix", System.currentTimeMillis() / 1000);
        env.put("machine", machineInfo());
        env.put("params", params);
        env.put("metrics", metrics);
        Files.createDirectories(resultsDir);
        Path out = resultsDir.resolve("lucene-" + subcommand + ".json");
        Files.writeString(out, GSON.toJson(env));
        return out;
    }

    public static long dirSize(Path dir) throws IOException {
        try (var s = Files.walk(dir)) {
            return s.filter(Files::isRegularFile).mapToLong(p -> {
                try {
                    return Files.size(p);
                } catch (IOException e) {
                    return 0;
                }
            }).sum();
        }
    }
}
