package bench;

import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.document.SortedDocValuesField;
import org.apache.lucene.document.StoredField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.store.MMapDirectory;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.NumericUtils;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * {@code index}: build the Lucene index from the JSONL corpus.
 *
 * Schema (mirrors the rustic / tantivy schemas):
 * <ul>
 *   <li>{@code id}:    StoredField + NumericDocValues</li>
 *   <li>{@code title}, {@code body}: TextField (positions), stored, custom
 *       lowercase + letter-or-digit analyzer</li>
 *   <li>{@code price}: NumericDocValues (sortable-long encoded double) only,
 *       i.e. a columnar "fast field" like the other two systems; range queries
 *       use {@code NumericDocValuesField.newSlowRangeQuery}</li>
 *   <li>{@code ts}:    NumericDocValues</li>
 *   <li>{@code tag}:   StringField (inverted, not analyzed) + SortedDocValues</li>
 * </ul>
 * Writer: MMapDirectory, 1 GB RAM buffer, single thread, default
 * TieredMergePolicy, one commit at the end (optionally {@code --commit-every N}).
 */
public final class Index {
    private Index() {}

    public static void run(Common.Args args, Path results) throws Exception {
        Path corpus = Path.of(args.require("corpus"));
        Path out = Path.of(args.require("out"));
        int ramMb = args.getInt("heap-mb", 1024);
        int commitEvery = args.getInt("commit-every", 0);

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

        System.err.printf("[index] lucene -> %s (%d MB RAM buffer, 1 thread, commit every %s)%n",
                out, ramMb, commitEvery > 0 ? String.valueOf(commitEvery) : "end only");
        long t0 = System.nanoTime();
        long[] n = {0};
        int segments;
        try (MMapDirectory dir = new MMapDirectory(out)) {
            IndexWriterConfig cfg = new IndexWriterConfig(Common.analyzer())
                    .setOpenMode(IndexWriterConfig.OpenMode.CREATE)
                    .setRAMBufferSizeMB(ramMb)
                    .setMaxBufferedDocs(IndexWriterConfig.DISABLE_AUTO_FLUSH)
                    .setUseCompoundFile(false);
            try (IndexWriter writer = new IndexWriter(dir, cfg)) {
                Common.forEachDoc(corpus, d -> {
                    Document doc = new Document();
                    doc.add(new StoredField("id", d.id()));
                    doc.add(new NumericDocValuesField("id", d.id()));
                    doc.add(new TextField("title", d.title(), Field.Store.YES));
                    doc.add(new TextField("body", d.body(), Field.Store.YES));
                    doc.add(new NumericDocValuesField("price", NumericUtils.doubleToSortableLong(d.price())));
                    doc.add(new org.apache.lucene.document.DoublePoint("price", d.price()));
                    doc.add(new NumericDocValuesField("ts", d.ts()));
                    doc.add(new org.apache.lucene.document.LongPoint("ts", d.ts()));
                    doc.add(new StringField("tag", d.tag(), Field.Store.NO));
                    doc.add(new SortedDocValuesField("tag", new BytesRef(d.tag())));
                    try {
                        writer.addDocument(doc);
                    } catch (IOException e) {
                        throw new RuntimeException(e);
                    }
                    n[0]++;
                    if (commitEvery > 0 && n[0] % commitEvery == 0) {
                        try {
                            writer.commit();
                        } catch (IOException e) {
                            throw new RuntimeException(e);
                        }
                    }
                    if (n[0] % 100_000 == 0) {
                        System.err.printf("[index]   %d docs, %.1fs%n", n[0], (System.nanoTime() - t0) / 1e9);
                    }
                });
                writer.commit();
            }
            try (DirectoryReader r = DirectoryReader.open(dir)) {
                segments = r.leaves().size();
            }
        }
        double wall = (System.nanoTime() - t0) / 1e9;
        long bytes = Common.dirSize(out);
        double dps = n[0] / wall;
        System.out.printf("lucene: %d docs in %.2fs = %.0f docs/s, %.1f MB on disk, %d segment(s)%n",
                n[0], wall, dps, bytes / 1e6, segments);

        Map<String, Object> params = new LinkedHashMap<>();
        params.put("corpus", corpus.toString());
        params.put("out", out.toString());
        params.put("threads", 1);
        params.put("heap_mb", ramMb);
        params.put("commit_every", commitEvery);
        Map<String, Object> metrics = new LinkedHashMap<>();
        metrics.put("docs", n[0]);
        metrics.put("wall_s", wall);
        metrics.put("docs_per_sec", dps);
        metrics.put("index_bytes", bytes);
        metrics.put("segments", segments);
        Path p = Common.writeResult(results, "index", params, metrics);
        System.out.println("wrote " + p);
    }
}
