package se.ton.t210.configuration;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.web.servlet.FilterRegistrationBean;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import se.ton.t210.configuration.filter.TokenFilter;
import se.ton.t210.domain.TokenSecret;

import javax.servlet.Filter;

@Configuration
public class TokenFilterConfig {

    @Value("${auth.jwt.token.access.cookie.key:accessToken}")
    private String accessTokenCookieKey;

    @Value("${auth.jwt.token.refresh.cookie:refreshToken}")
    private String refreshTokenCookieKey;

    @Value("${auth.jwt.token.email.cookie:emailAuthToken}")
    private String emailAuthTokenCookieKey;

    private final TokenSecret tokenSecret;

    public TokenFilterConfig(TokenSecret tokenSecret) {
        this.tokenSecret = tokenSecret;
    }

    @Bean
    public FilterRegistrationBean<Filter> addAccessTokenFilter() {
        FilterRegistrationBean<Filter> filterRegistrationBean = new FilterRegistrationBean<>();
        filterRegistrationBean.setFilter(new TokenFilter(tokenSecret, accessTokenCookieKey));
        filterRegistrationBean.addUrlPatterns("/api/auth/reissue/token");
        return filterRegistrationBean;
    }

    @Bean
    public FilterRegistrationBean<Filter> addRefreshTokenFilter() {
        FilterRegistrationBean<Filter> filterRegistrationBean = new FilterRegistrationBean<>();
        filterRegistrationBean.setFilter(new TokenFilter(tokenSecret, refreshTokenCookieKey));
        filterRegistrationBean.addUrlPatterns("/api/auth/reissue/token");
        return filterRegistrationBean;
    }

    @Bean
    public FilterRegistrationBean<Filter> addEmailTokenFilter() {
        FilterRegistrationBean<Filter> filterRegistrationBean = new FilterRegistrationBean<>();
        filterRegistrationBean.setFilter(new TokenFilter(tokenSecret, emailAuthTokenCookieKey));
        filterRegistrationBean.addUrlPatterns("/api/auth/signUp");
        return filterRegistrationBean;
    }
}
