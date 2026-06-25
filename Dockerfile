# Self-contained multi-stage build for the AZ Planet Wars agent.
# Stage 1 builds the fat JAR with the project's Gradle wrapper (Gradle 8.12) on JDK 21.
# Stage 2 runs it: `java -jar app.jar` launches az.RunAZServerKt — the WebSocket
# agent server on port 8080 (JAR Main-Class set in app/build.gradle.kts).
#
# Build:  docker build -t az-planetwars .
# Run:    docker run --rm -p 8080:8080 az-planetwars

# ---------- Stage 1: build ----------
FROM eclipse-temurin:21-jdk AS builder
WORKDIR /home/app
COPY . .
RUN chmod +x ./gradlew && ./gradlew :app:shadowJar -x test --no-daemon

# ---------- Stage 2: runtime ----------
FROM eclipse-temurin:21-jre
WORKDIR /app
COPY --from=builder /home/app/app/build/libs/client-server.jar app.jar
EXPOSE 8080
CMD ["java", "-jar", "app.jar"]
