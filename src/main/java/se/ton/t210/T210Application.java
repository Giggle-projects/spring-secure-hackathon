package se.ton.t210;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.web.servlet.ServletComponentScan;

@ServletComponentScan
@SpringBootApplication
public class T210Application {

    public static void main(String[] args) {
        SpringApplication app = new SpringApplication(T210Application.class);
        app.setAdditionalProfiles("dev");
        app.run(args);
    }
}

