package bench;

import java.nio.file.Path;

/**
 * Entry point: {@code index | startup | startup-child | query | vectors}.
 *
 * <pre>
 *   ./gradlew run --args="index   --corpus ../corpus/corpus.jsonl --out ../indexes/lucene [--results ../results]"
 *   ./gradlew run --args="startup --index ../indexes/lucene --queries ../corpus/queries.jsonl [--runs 5] [--no-purge]"
 *   ./gradlew run --args="query   --index ../indexes/lucene --queries ../corpus/queries.jsonl [--warmup 2] [--iters 5]"
 *   ./gradlew run --args="hybrid  --corpus ../corpus/corpus.jsonl --queries ../corpus/queries.jsonl --vectors ../corpus/vectors-q100.f32 [--vec-queries 100] [--reuse]"
 *   ./gradlew run --args="vectors --vectors ../corpus/vectors.f32 --out ../indexes/lucene-vec [--dim 128] [--queries 20]"
 * </pre>
 *
 * Every subcommand writes {@code <results>/lucene-<subcommand>.json} in the same
 * envelope as the Rust harness so {@code compare report} can merge them.
 */
public final class Main {
    public static void main(String[] argv) throws Exception {
        Common.Args args = new Common.Args(argv);
        Path results = Path.of(args.get("results", "../results"));
        switch (args.subcommand) {
            case "index" -> Index.run(args, results);
            case "startup" -> Startup.run(args, results);
            case "startup-child" -> Startup.child(args);
            case "query" -> Query.run(args, results);
            case "vectors" -> Vectors.run(args, results);
            case "vectors-serve" -> VectorServer.run(args);
            case "sort-serve" -> SortServer.run(args);
            case "vectors-filtered" -> VectorsFiltered.run(args, results);
            case "hybrid" -> Hybrid.run(args, results);
            default -> {
                System.err.println("usage: bench.Main <index|startup|query|vectors> [--flags]  (see Main.java)");
                System.exit(2);
            }
        }
    }
}
