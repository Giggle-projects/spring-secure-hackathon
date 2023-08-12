## Gradle jib

### Our container registry
```
ghcr.io/giggle-projects/spring-secure-hackathon
```

### Why jib
`https://cloud.google.com/java/getting-started/jib?hl=ko`
Jib는 애플리케이션을 종속 항목, 리소스, 클래스 등 별개의 레이어로 구성하고 Docker 이미지 레이어 캐싱을 활용하여 변경사항만 다시 빌드함으로써 빌드를 빠르게 유지합니다. Jib 레이어 구성과 작은 기본 이미지는 전체 이미지 크기를 작게 유지하여 성능과 휴대성을 향상시킵니다.

### Gradle config

```
plugins {
    id 'com.google.cloud.tools.jib' version '3.3.2'
}
```

```
jib {
    from.image = "adoptopenjdk/openjdk11:jre-11.0.10_9-alpine"
    to.image = "ghcr.io/giggle-projects/spring-secure-hackathon"
    to.tags = ["latest"]
    // ./gradlew jib
}
```

### Gradle build
```
./gradlew jib
```

the end
