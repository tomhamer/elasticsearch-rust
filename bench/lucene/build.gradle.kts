// Lucene 10 benchmark harness. See ../README.md for install + run instructions.
//
//   brew install openjdk@21 gradle
//   gradle wrapper --gradle-version 8.10      # (once) generates ./gradlew
//   ./gradlew run --args="index --corpus ../corpus/corpus.jsonl --out ../indexes/lucene"
//
// Bump `luceneVersion` to the newest 10.x release if the one below is gone.
plugins {
    application
}

val luceneVersion = providers.gradleProperty("luceneVersion").getOrElse("10.5.1")

repositories {
    mavenCentral()
}

dependencies {
    implementation("org.apache.lucene:lucene-core:$luceneVersion")
    implementation("org.apache.lucene:lucene-analysis-common:$luceneVersion")
    implementation("org.apache.lucene:lucene-queryparser:$luceneVersion")
    implementation("com.google.code.gson:gson:2.11.0")
}

java {
    toolchain {
        languageVersion.set(JavaLanguageVersion.of(21))
    }
}

application {
    mainClass.set("bench.Main")
    // MMapDirectory on JDK 21 uses the Panama memory-segment API; a 1 GB RAM
    // buffer plus the mapped index comfortably fits in a 4 GB heap.
    applicationDefaultJvmArgs = listOf(
        "-Xms4g", "-Xmx4g", "-XX:+UseParallelGC",
        // Panama native access (MMapDirectory) and the incubating Vector API
        // that Lucene uses for SIMD postings decoding / vector distance.
        "--enable-native-access=ALL-UNNAMED", "--add-modules", "jdk.incubator.vector",
    )
}

tasks.named<JavaExec>("run") {
    // Let `./gradlew run --args=...` print straight to the terminal.
    standardOutput = System.out
    errorOutput = System.err
    workingDir = projectDir
}
