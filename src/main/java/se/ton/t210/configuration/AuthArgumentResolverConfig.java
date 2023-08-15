package se.ton.t210.configuration;

import org.springframework.context.annotation.Configuration;
import org.springframework.web.method.support.HandlerMethodArgumentResolver;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;
import se.ton.t210.configuration.interceptor.LoginUserArgumentResolver;

import java.util.List;

@Configuration
public class AuthArgumentResolverConfig implements WebMvcConfigurer {

    private final LoginUserArgumentResolver resolver;

    public AuthArgumentResolverConfig(LoginUserArgumentResolver resolver) {
        this.resolver = resolver;
    }

    @Override
    public void addArgumentResolvers(List<HandlerMethodArgumentResolver> resolvers) {
        resolvers.add(resolver);
    }
}
