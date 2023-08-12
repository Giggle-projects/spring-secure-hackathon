package se.ton.t210.configuration;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import se.ton.t210.domain.TokenSecret;

@Configuration
public class TokenConfig {

    @Value("${jwt.secret}")
    private String secret;

    @Bean
    public TokenSecret tokenSecret() {
        return new TokenSecret(secret);
    }
}
