package se.ton.t210;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
public class T210Application {

    public static void main(String[] args) {
        SpringApplication app = new SpringApplication(T210Application.class);
        app.setAdditionalProfiles("prod");
        app.run(args);
    }
}
