package bench;

import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.sun.net.httpserver.HttpServer;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Map;
import java.util.concurrent.Executors;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.index.Term;
import org.apache.lucene.store.MMapDirectory;

/** Paired native-query diagnostic. Query time excludes HTTP, parsing and ID fetch. */
public final class VectorServer {
    public static void run(Common.Args args) throws Exception {
        System.setProperty("sun.net.httpserver.nodelay", "true");
        boolean esQuery = args.get("query-path", "lucene").equals("es");
        String early = args.get("early-termination", "true");
        if (!early.equals("true") && !early.equals("false")) throw new IllegalArgumentException("early-termination must be true or false");
        boolean earlyTermination = Boolean.parseBoolean(early);
        if (!esQuery && !earlyTermination) throw new IllegalArgumentException("early-termination is an ES query diagnostic");
        if (esQuery) EsNative.initialize();
        var directory = new MMapDirectory(Path.of(args.require("index")));
        var reader = DirectoryReader.open(directory);
        var searcher = new IndexSearcher(reader);
        searcher.setQueryCache(null);
        var server = HttpServer.create(new InetSocketAddress("127.0.0.1", args.getInt("port", 19230)), 64);
        server.setExecutor(Executors.newSingleThreadExecutor());
        server.createContext("/_search", exchange -> {
            try {
                long start = System.nanoTime();
                JsonObject body = JsonParser.parseString(new String(exchange.getRequestBody().readAllBytes(), StandardCharsets.UTF_8)).getAsJsonObject();
                JsonObject knn = body.getAsJsonObject("knn");
                if (knn.has("rescore_vector") || body.get("_source").getAsBoolean())
                    throw new IllegalArgumentException("float KNN with source disabled required");
                var array = knn.getAsJsonArray("query_vector");
                float[] vector = new float[array.size()];
                for (int i = 0; i < vector.length; i++) vector[i] = array.get(i).getAsFloat();
                int k = knn.get("k").getAsInt();
                Query filter = knn.has("filter") ? filter(knn.getAsJsonObject("filter")) : null;
                Query query = esQuery ? EsNative.query("vec", vector, k, knn.get("num_candidates").getAsInt(), filter, earlyTermination)
                    : new KnnFloatVectorQuery("vec", vector, knn.get("num_candidates").getAsInt(), filter);
                long parsed = System.nanoTime();
                var top = searcher.search(query, k);
                long searched = System.nanoTime();
                var hits = new ArrayList<Map<String, Object>>();
                var stored = searcher.storedFields();
                for (var hit : top.scoreDocs) {
                    String id = stored.document(hit.doc).getField("id").numericValue().toString();
                    hits.add(Map.of("_index", "native-lucene", "_id", id, "_score", hit.score));
                }
                var response = Map.of("hits", Map.of("total", Map.of("value", hits.size(), "relation", "eq"), "hits", hits),
                    "_shards", Map.of("total", 1, "successful", 1, "failed", 0),
                    "_poc", Map.of("parse_us", (parsed-start)/1000.0, "query_us", (searched-parsed)/1000.0,
                                    "fetch_us", (System.nanoTime()-searched)/1000.0));
                byte[] bytes = Common.GSON.toJson(response).getBytes(StandardCharsets.UTF_8);
                exchange.getResponseHeaders().set("Content-Type", "application/json");
                exchange.sendResponseHeaders(200, bytes.length);
                exchange.getResponseBody().write(bytes);
            } catch (Exception e) {
                byte[] bytes = e.toString().getBytes(StandardCharsets.UTF_8);
                exchange.sendResponseHeaders(400, bytes.length);
                exchange.getResponseBody().write(bytes);
            } finally {
                exchange.close();
            }
        });
        server.start();
        System.err.println("Native vector server ready on " + server.getAddress() + "; query-path=" + (esQuery ? "es" : "lucene") + "; early-termination=" + earlyTermination);
    }

    private static Query filter(JsonObject value) {
        if (value.has("term")) {
            var term = value.getAsJsonObject("term");
            if (term.size() != 1 || !term.has("tag")) throw new IllegalArgumentException("tag filter required");
            return new TermQuery(new Term("tag", term.get("tag").getAsString()));
        }
        var bool = value.getAsJsonObject("bool");
        if (bool == null || bool.size() != 1 || !bool.has("should"))
            throw new IllegalArgumentException("simple should disjunction required");
        var query = new BooleanQuery.Builder();
        for (var clause : bool.getAsJsonArray("should")) query.add(filter(clause.getAsJsonObject()), BooleanClause.Occur.SHOULD);
        return query.build();
    }
}
