import org.elasticsearch.h3.CellBoundary;
import org.elasticsearch.h3.H3;
import org.elasticsearch.h3.LatLng;

import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;

/**
 * Fixtures for the Rust port of ES's H3 library: points (random, on cell
 * edges and vertices, around pentagons, at the poles and the dateline) with
 * their cells at several resolutions, and cells with their boundaries,
 * centres, parents, children, non-intersecting children and rings.
 *
 * java -cp elasticsearch-h3-9.5.5.jar Gen.java points.txt cells.txt
 */
public class Gen {
    static long seed = 0x9E3779B97F4A7C15L;

    static double next() {
        seed = seed * 6364136223846793005L + 1442695040888963407L;
        return (seed >>> 11) * 0x1.0p-53;
    }

    static String b(double d) {
        return Long.toHexString(Double.doubleToRawLongBits(d));
    }

    static final int[] RES = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };

    public static void main(String[] args) throws Exception {
        List<double[]> pts = new ArrayList<>();
        for (int i = 0; i < 1200; i++) {
            pts.add(new double[] { next() * 180 - 90, next() * 360 - 180 });
        }
        // Vertices, edge midpoints and points a hair off them.
        List<Long> cells = new ArrayList<>();
        for (int i = 0; i < 160; i++) {
            int res = (int) (next() * 16);
            long c = H3.geoToH3(next() * 180 - 90, next() * 360 - 180, res);
            cells.add(c);
            CellBoundary bd = H3.h3ToGeoBoundary(c);
            for (int v = 0; v < bd.numPoints(); v++) {
                LatLng a = bd.getLatLon(v);
                LatLng z = bd.getLatLon((v + 1) % bd.numPoints());
                pts.add(new double[] { a.getLatDeg(), a.getLonDeg() });
                double mlat = (a.getLatDeg() + z.getLatDeg()) / 2, mlon = (a.getLonDeg() + z.getLonDeg()) / 2;
                if (Math.abs(a.getLonDeg() - z.getLonDeg()) < 180) {
                    pts.add(new double[] { mlat, mlon });
                    pts.add(new double[] { Math.nextUp(mlat), mlon });
                    pts.add(new double[] { mlat, Math.nextDown(mlon) });
                }
            }
        }
        // Around the pentagons.
        for (long base : H3.getLongRes0Cells()) {
            if (H3.isPentagon(base) == false) {
                continue;
            }
            long c = base;
            for (int res = 0; res <= 15; res++) {
                if (res > 0) {
                    c = H3.childPosToH3(c, 0);
                }
                cells.add(c);
                if (res % 3 == 0) {
                    LatLng ctr = H3.h3ToLatLng(c);
                    double d = 10.0 / Math.pow(2.6, res);
                    for (int k = 0; k < 6; k++) {
                        double ang = k * Math.PI / 3 + 0.1;
                        pts.add(new double[] { ctr.getLatDeg() + d * Math.sin(ang), ctr.getLonDeg() + d * Math.cos(ang) });
                    }
                    pts.add(new double[] { ctr.getLatDeg(), ctr.getLonDeg() });
                }
            }
        }
        // Poles and the dateline.
        for (double lon : new double[] { -180, -179.999999, -90, 0, 45.5, 90, 179.999999, 180 }) {
            for (double lat : new double[] { 90, 89.9999999, 89.99, 88, -88, -89.99, -89.9999999, -90 }) {
                pts.add(new double[] { lat, lon });
            }
        }
        for (double lat : new double[] { -60, -10.5, 0, 0.000001, 33.3, 71 }) {
            pts.add(new double[] { lat, 180 });
            pts.add(new double[] { lat, -180 });
            pts.add(new double[] { lat, Math.nextDown(180.0) });
            pts.add(new double[] { lat, Math.nextUp(-180.0) });
        }
        try (PrintWriter w = new PrintWriter(args[0])) {
            for (double[] p : pts) {
                StringBuilder sb = new StringBuilder(b(p[0])).append(' ').append(b(p[1]));
                for (int r : RES) {
                    sb.append(' ').append(Long.toHexString(H3.geoToH3(p[0], p[1], r)));
                }
                w.println(sb);
            }
        }
        for (long c : H3.getLongRes0Cells()) {
            cells.add(c);
        }
        for (int i = 0; i < 300; i++) {
            cells.add(H3.geoToH3(next() * 180 - 90, next() * 360 - 180, (int) (next() * 16)));
        }
        for (int res = 0; res <= 15; res++) {
            cells.add(H3.northPolarH3(res));
            cells.add(H3.southPolarH3(res));
        }
        // cell | centre | boundary | parent | children | no-children | ring
        try (PrintWriter w = new PrintWriter(args[1])) {
            for (long c : cells) {
                StringBuilder sb = new StringBuilder(Long.toHexString(c));
                LatLng ctr = H3.h3ToLatLng(c);
                sb.append('|').append(b(ctr.getLatRad())).append(' ').append(b(ctr.getLonRad())).append('|');
                CellBoundary bd = H3.h3ToGeoBoundary(c);
                for (int v = 0; v < bd.numPoints(); v++) {
                    if (v > 0) sb.append(' ');
                    sb.append(b(bd.getLatLon(v).getLatRad())).append(' ').append(b(bd.getLatLon(v).getLonRad()));
                }
                sb.append('|').append(H3.getResolution(c) > 0 ? Long.toHexString(H3.h3ToParent(c)) : "-").append('|');
                if (H3.getResolution(c) < 15) {
                    sb.append(String.join(" ", hex(H3.h3ToChildren(c)))).append('|');
                    sb.append(String.join(" ", hex(H3.h3ToNoChildrenIntersecting(c)))).append('|');
                } else {
                    sb.append("-|-|");
                }
                sb.append(String.join(" ", hex(H3.hexRing(c))));
                w.println(sb);
            }
        }
    }

    static List<String> hex(long[] v) {
        List<String> out = new ArrayList<>();
        for (long x : v) {
            out.add(Long.toHexString(x));
        }
        return out;
    }
}
