package se.ton.t210.configuration;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.servlet.config.annotation.CorsRegistry;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;

@Configuration
public class WebCorsConfig implements WebMvcConfigurer {

    @Value("${web.cors.allow.origins:http://localhost:63342}")
    private String[] allowOrigins;

    @Override
    public void addCorsMappings(CorsRegistry registry) {
        registry.addMapping("/**")
                .allowedOrigins(
                    "https://t210.ecsimsw.com:8080",
                    "https://t210.ecsimsw.com:8443",
                    "http://127.0.0.1:8443",
                    "http://127.0.0.1:8080",
                    "http://127.0.0.1:80"
                )
                .allowedMethods("*")
                .allowCredentials(true);
    }
}
