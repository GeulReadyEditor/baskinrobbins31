# maven이미지에 mvn package로 *.jar 파일 생성
FROM maven:3.8-jdk-11-slim AS builder
WORKDIR /app
COPY ./ ./
RUN mvn package

# openjdk 이미지에 builder에서 생성된 *.jar파일을 복사
FROM openjdk:11-jre-slim
WORKDIR /app
ARG --from=builder JAR_FILE=/app/target/*.jar
COPY --from=builder ${JAR_FILE} app.jar
CMD ["java", "-jar", "app.jar"]