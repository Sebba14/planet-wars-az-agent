import com.github.jengelman.gradle.plugins.shadow.tasks.ShadowJar

plugins {
    kotlin("jvm") version "2.1.0"
    kotlin("plugin.serialization") version "2.1.0"
    application
    id("com.github.johnrengelman.shadow") version "8.1.1"
}



repositories {
    mavenCentral()
}

group = "sml"
version = "1.0-SNAPSHOT"

dependencies {
    // Kotlin Standard Library
    implementation("org.jetbrains.kotlin:kotlin-stdlib:2.1.0")

    // Ktor dependencies
    implementation("io.ktor:ktor-server-core:2.3.3")
    implementation("io.ktor:ktor-server-netty:2.3.3")
    implementation("io.ktor:ktor-server-websockets:2.3.3")
    implementation("io.ktor:ktor-client-core:2.3.3")
    implementation("io.ktor:ktor-client-cio:2.3.3")
    implementation("io.ktor:ktor-server-content-negotiation:2.3.3")
    implementation("io.ktor:ktor-serialization-kotlinx-json:2.3.3")

    // Kotlinx Serialization
    implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.5.1")

    // Coroutines
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.7.3")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-jdk8:1.7.3")

    // Testing
    testImplementation("org.jetbrains.kotlin:kotlin-test-junit5")
    testImplementation("org.junit.jupiter:junit-jupiter-engine:5.10.0")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")

    // Additional Libraries
    implementation("com.google.guava:guava:32.1.2-jre")

    // ONNX Runtime for JVM (bundles native libs for win/linux/mac x64) — net-guided search
    implementation("com.microsoft.onnxruntime:onnxruntime:1.20.0")
}
tasks.withType<Test>().configureEach {
    useJUnitPlatform()
}

tasks.withType<Jar> {
    manifest {
        // AZ entry: java -jar uses the JAR manifest Main-Class, so point it at our
        // WebSocket agent server (NOT the upstream stub client_server.MultiRTSServer).
        attributes["Main-Class"] = "az.RunAZServerKt"
    }
}

tasks.named<ShadowJar>("shadowJar") {
    archiveBaseName.set("client-server")
    archiveClassifier.set("")
    archiveVersion.set("")
}


java {
    toolchain {
        languageVersion = JavaLanguageVersion.of(21) // Java 21 LTS - best compatibility
    }
}

application {
    mainClass.set("az.RunAZServerKt") // AZ competition entry (WebSocket agent server on :8080)
}

kotlin {
    jvmToolchain(21) // Ensure Kotlin targets JVM 21 as well
}

tasks.register<JavaExec>("runGate") {
    // AZ promotion gate: net-MCTS(onnx file) vs SFP. gradlew :app:runGate -Pargs="<onnx> <games> [budgetMs]"
    mainClass.set("az.RunGateKt")
    classpath = sourceSets["main"].runtimeClasspath
    args = listOf(project.findProperty("args")?.toString() ?: "runs/current.onnx 20 20")
}

tasks.register<JavaExec>("runOnnxSmoke") {
    // onnxruntime-java inference smoke test (loads bundled /az/model.onnx)
    mainClass.set("az.RunOnnxSmokeKt")
    classpath = sourceSets["main"].runtimeClasspath
}

tasks.register<JavaExec>("runSelfPlay") {
    // AlphaZero self-play data generation: gradlew :app:runSelfPlay -Pargs="8 selfplay_out 1 gen0"
    mainClass.set("az.RunSelfPlayKt")
    classpath = sourceSets["main"].runtimeClasspath
    args = listOf(project.findProperty("args")?.toString() ?: "4 selfplay_out 1 gen0")
}

tasks.register<JavaExec>("runBench") {
    // Forward-model speed benchmark (gates SFP search budget)
    mainClass.set("az.RunBenchKt")
    classpath = sourceSets["main"].runtimeClasspath
}

tasks.register<JavaExec>("runRemoteClient") {
    // Connect to an already-running agent server (Docker/java -jar): gradlew :app:runRemoteClient -Pargs=8080
    mainClass.set("az.RunRemoteClientKt")
    classpath = sourceSets["main"].runtimeClasspath
    args = listOf(project.findProperty("args")?.toString() ?: "8080")
}

tasks.register<JavaExec>("runRemoteSmoke") {
    // AZ end-to-end WebSocket smoke test (server + RemoteAgent client, 50ms timeout)
    mainClass.set("az.RunRemoteSmokeTestKt")
    classpath = sourceSets["main"].runtimeClasspath
}

tasks.register<JavaExec>("runLocalLeague") {
    // AZ local gauntlet vs sample agents: gradlew :app:runLocalLeague -Pargs=50
    mainClass.set("az.RunLocalLeagueKt")
    classpath = sourceSets["main"].runtimeClasspath
    args = listOf(project.findProperty("args")?.toString() ?: "50")
}

tasks.register<JavaExec>("runEvaluation") {
    mainClass.set("games.planetwars.runners.EvaluateAgentKt")
    classpath = sourceSets["main"].runtimeClasspath
    args = listOf(project.findProperty("args")?.toString() ?: "49875")
}

tasks.register<JavaExec>("runUnifiedExample") {
    mainClass.set("games.planetwars.runners.UnifiedGameRunnerKt")
    classpath = sourceSets["main"].runtimeClasspath
    standardInput = System.`in`
}
tasks.register<JavaExec>("runRemotePairEvaluation") {
    // Kotlin entry point above
    mainClass.set("games.planetwars.runners.RunRemotePairEvaluationKt")
    classpath = sourceSets["main"].runtimeClasspath

    // Support `--args=portA,portB,gpp,timeout` (comes in as a project property)
    val raw = project.findProperty("args")?.toString()
    args = if (raw != null) listOf(raw) else listOf("5001,5002,10,50")
}
