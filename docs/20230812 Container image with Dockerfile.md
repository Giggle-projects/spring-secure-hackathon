## Container image with dockerfile

### Our container registry
```
ghcr.io/giggle-projects/spring-secure-hackathon
```

### Gradle build
```
./gradlew build
```

### Container build with Dockerfile
```
FROM adoptopenjdk/openjdk11
VOLUME /tmp
ARG JAR_FILE
COPY build/libs/*.jar /app.jar
ENTRYPOINT ["java","-jar","/app.jar"]
```

### Build container image and publish
```
docker build -t ghcr.io/giggle-projects/spring-secure-hackathon:latest .
docker push ghcr.io/giggle-projects/spring-secure-hackathon
```
