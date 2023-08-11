package se.ton.t210.configuration;

import org.springframework.boot.web.servlet.FilterRegistrationBean;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import se.ton.t210.configuration.filter.AccessTokenValidationFilter;
import se.ton.t210.configuration.filter.RefreshTokenValidationFilter;

import javax.servlet.Filter;

@Configuration
public class TokenFilterConfig {

    @Bean
    public FilterRegistrationBean<Filter> addAccessTokenFilter(AccessTokenValidationFilter accessTokenValidationFilter) {
        FilterRegistrationBean<Filter> filterRegistrationBean = new FilterRegistrationBean<>();
        filterRegistrationBean.setFilter(accessTokenValidationFilter);
        filterRegistrationBean.addUrlPatterns("/api/me/*");
        return filterRegistrationBean;
    }

    @Bean
    public FilterRegistrationBean<Filter> addRefreshTokenFilter(RefreshTokenValidationFilter refreshTokenValidationFilter) {
        FilterRegistrationBean<Filter> filterRegistrationBean = new FilterRegistrationBean<>();
        filterRegistrationBean.setFilter(refreshTokenValidationFilter);
        filterRegistrationBean.addUrlPatterns("/api/auth/reissue/token");
        return filterRegistrationBean;
    }
}
