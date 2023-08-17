package se.ton.t210.configuration;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.web.servlet.FilterRegistrationBean;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import se.ton.t210.configuration.filter.AccessTokenFilter;
import se.ton.t210.configuration.filter.EmailAuthTokenFilter;
import se.ton.t210.domain.TokenSecret;
import se.ton.t210.service.token.TokenService;

import javax.servlet.Filter;

@Configuration
public class TokenFilterConfig {

    @Value("${auth.jwt.token.access.cookie.key:accessToken}")
    private String accessTokenCookieKey;

    @Value("${auth.jwt.token.refresh.cookie:refreshToken}")
    private String refreshTokenCookieKey;

    @Value("${auth.jwt.token.email.cookie:emailAuthToken}")
    private String emailAuthTokenCookieKey;

    private final TokenService tokenService;

    public TokenFilterConfig(TokenService tokenService) {
        this.tokenService = tokenService;
    }

    @Bean
    public FilterRegistrationBean<Filter> addAccessTokenFilter() {
        FilterRegistrationBean<Filter> filterRegistrationBean = new FilterRegistrationBean<>();
        filterRegistrationBean.setFilter(new AccessTokenFilter(accessTokenCookieKey, refreshTokenCookieKey, tokenService));
        filterRegistrationBean.addUrlPatterns(
            "/html/dashboard.html",
            "/html/personal-information.html",
            "/html/record.html",
            "/html/application-information1.html",
            "/html/application-information2.html",
            "/html/application-information3.html",
            "/html/application-information4.html",
            "/html/application-information5.html",
            "/html/application-information6.html",
            "/html/setting-account.html"
        );
        return filterRegistrationBean;
    }

    @Bean
    public FilterRegistrationBean<Filter> addEmailTokenFilter() {
        FilterRegistrationBean<Filter> filterRegistrationBean = new FilterRegistrationBean<>();
        filterRegistrationBean.setFilter(new EmailAuthTokenFilter(emailAuthTokenCookieKey, tokenService));
        filterRegistrationBean.addUrlPatterns(
            "/api/member/signUp"
        );
        return filterRegistrationBean;
    }
}
