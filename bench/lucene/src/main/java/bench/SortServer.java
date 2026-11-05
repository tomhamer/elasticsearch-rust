package bench;

import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.sun.net.httpserver.HttpServer;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.concurrent.Executors;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.FieldDoc;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.MatchAllDocsQuery;
import org.apache.lucene.search.Sort;
import org.apache.lucene.search.SortField;
import org.apache.lucene.search.SortedNumericSelector;
import org.apache.lucene.search.SortedNumericSortField;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.TopFieldCollectorManager;
import org.apache.lucene.store.MMapDirectory;

/** Native field-sort diagnostic with exact counts; query time excludes transport/fetch. */
public final class SortServer {
    private static final com.google.gson.Gson GSON = new com.google.gson.GsonBuilder().serializeNulls().create();
    public static void run(Common.Args args) throws Exception {
        System.setProperty("sun.net.httpserver.nodelay", "true");
        var directory = new MMapDirectory(Path.of(args.require("index")));
        var reader = DirectoryReader.open(directory);
        var searcher = new IndexSearcher(reader);
        searcher.setQueryCache(null);
        var server = HttpServer.create(new InetSocketAddress("127.0.0.1", args.getInt("port", 19252)), 64);
        server.setExecutor(Executors.newSingleThreadExecutor());
        server.createContext("/_search", exchange -> {
            try {
                long start = System.nanoTime();
                JsonObject body = JsonParser.parseString(new String(exchange.getRequestBody().readAllBytes(), StandardCharsets.UTF_8)).getAsJsonObject();
                if (body.get("_source").getAsBoolean() || !body.get("track_total_hits").getAsBoolean())
                    throw new IllegalArgumentException("source disabled and exact totals required");
                org.apache.lucene.search.Query query = new MatchAllDocsQuery();
                if (body.has("query") && !body.getAsJsonObject("query").has("match_all")) {
                    var term = body.getAsJsonObject("query").getAsJsonObject("term");
                    if (term == null || term.size() != 1 || !term.has("tag"))
                        throw new IllegalArgumentException("match_all or tag term required");
                    query = new TermQuery(new Term("tag", term.get("tag").getAsString()));
                }
                var fields = new ArrayList<SortField>();
                for (var entry : body.getAsJsonArray("sort")) {
                    var object = entry.getAsJsonObject();
                    if (object.size() != 1) throw new IllegalArgumentException("one-field sort required");
                    var field = object.entrySet().iterator().next();
                    String direction = field.getValue().getAsString();
                    if (!direction.equals("asc") && !direction.equals("desc")) throw new IllegalArgumentException("asc/desc required");
                    boolean reverse = direction.equals("desc");
                    fields.add(switch (field.getKey()) {
                        // The index uses ES-style sortable-long numeric doc values.
                        case "price" -> new SortedNumericSortField("price", SortField.Type.DOUBLE, reverse,
                            reverse ? SortedNumericSelector.Type.MAX : SortedNumericSelector.Type.MIN);
                        case "ts" -> new SortField("ts", SortField.Type.LONG, reverse);
                        default -> throw new IllegalArgumentException("price/ts required");
                    });
                }
                var sort = new Sort(fields.toArray(SortField[]::new));
                int size = body.get("size").getAsInt();
                long parsed = System.nanoTime();
                var top = searcher.search(query, new TopFieldCollectorManager(sort, size, Integer.MAX_VALUE));
                long searched = System.nanoTime();
                var hits = new ArrayList<Map<String, Object>>();
                var stored = searcher.storedFields();
                for (var hit : top.scoreDocs) {
                    var result = new LinkedHashMap<String, Object>();
                    result.put("_index", "native-lucene");
                    result.put("_id", stored.document(hit.doc).getField("id").numericValue().toString());
                    result.put("_score", null);
                    result.put("sort", ((FieldDoc)hit).fields);
                    hits.add(result);
                }
                var hitBody = new LinkedHashMap<String, Object>();
                hitBody.put("total", Map.of("value", top.totalHits.value(), "relation", "eq"));
                hitBody.put("max_score", null);
                hitBody.put("hits", hits);
                var response = Map.of("hits", hitBody,
                    "_shards", Map.of("total", 1, "successful", 1, "failed", 0),
                    "_poc", Map.of("parse_us", (parsed-start)/1000.0, "query_us", (searched-parsed)/1000.0,
                                    "fetch_us", (System.nanoTime()-searched)/1000.0));
                // Gson's default omits null map values; preserve ES's null score contract.
                byte[] bytes = GSON.toJson(response).getBytes(StandardCharsets.UTF_8);
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
        System.err.println("Native Lucene sort server ready on " + server.getAddress());
    }
}
