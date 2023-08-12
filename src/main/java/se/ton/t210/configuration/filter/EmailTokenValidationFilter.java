package se.ton.t210.configuration.filter;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Component;
import org.springframework.web.filter.OncePerRequestFilter;
import se.ton.t210.exception.AuthException;
import se.ton.t210.service.JwtUtils;

import javax.servlet.FilterChain;
import javax.servlet.ServletException;
import javax.servlet.http.Cookie;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.IOException;
import java.util.Arrays;

@Component
public class EmailTokenValidationFilter extends OncePerRequestFilter {

    private static final Logger LOGGER = LoggerFactory.getLogger(EmailTokenValidationFilter.class);

    @Value("${auth.jwt.token.email.cookie:emailAuthToken}")
    private String emailAuthTokenCookieKey;

    private final JwtUtils jwtUtils;

    public EmailTokenValidationFilter(JwtUtils jwtUtils) {
        this.jwtUtils = jwtUtils;
    }

    @Override
    protected void doFilterInternal(HttpServletRequest request, HttpServletResponse response, FilterChain filterChain) throws ServletException, IOException {
        LOGGER.info("EmailTokenValidationFilter");
        try {
            String emailAuthToken = getTokenFromCookies(emailAuthTokenCookieKey, request.getCookies());
            jwtUtils.validateToken(emailAuthToken);

        } catch (AuthException e) {
            response.setStatus(HttpStatus.UNAUTHORIZED.value());
            response.getOutputStream().write(e.getMessage().getBytes());
            return;
        }

        filterChain.doFilter(request, response);
    }

    private String getTokenFromCookies(String keyName, Cookie[] cookies) {
        if (cookies == null) {
            throw new AuthException(HttpStatus.UNAUTHORIZED, "Cookie corrupted");
        }
        return Arrays.stream(cookies)
                .filter(cookie -> cookie.getName().equals(keyName))
                .map(Cookie::getValue)
                .findFirst()
                .orElseThrow(() -> new AuthException(HttpStatus.UNAUTHORIZED, "JWT Token is not found"));
    }
}
