package se.ton.t210.configuration;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.web.servlet.FilterRegistrationBean;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import se.ton.t210.filter.AccessTokenValidationFilter;
import se.ton.t210.token.JwtUtils;

import javax.servlet.Filter;

@Configuration
public class SecurityConfig {

    @Value("${jwt.secret}")
    private String secret;

    @Bean
    public JwtUtils jwtProvider() {
        return new JwtUtils(secret);
    }

    @Bean
    public FilterRegistrationBean<Filter> addTokenFilter() {
        FilterRegistrationBean<Filter> filterRegistrationBean = new FilterRegistrationBean<>();
        filterRegistrationBean.setFilter(new AccessTokenValidationFilter(jwtProvider()));
        filterRegistrationBean.setOrder(1);
        filterRegistrationBean.addUrlPatterns("/*");
        return filterRegistrationBean;
    }
}
