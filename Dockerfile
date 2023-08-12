# ./gradlew build
# docker build -t ghcr.io/giggle-projects/spring-secure-hackathon:latest .

FROM adoptopenjdk/openjdk11
VOLUME /tmp
ARG JAR_FILE
COPY build/libs/*.jar /app.jar
ENTRYPOINT ["java","-jar","/app.jar"]
