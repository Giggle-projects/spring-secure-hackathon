package se.ton.t210.filter;

import org.springframework.http.HttpStatus;
import org.springframework.web.filter.OncePerRequestFilter;
import se.ton.t210.exception.AuthException;
import se.ton.t210.token.JwtUtils;

import javax.servlet.FilterChain;
import javax.servlet.ServletException;
import javax.servlet.http.Cookie;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;

public class TokenValidationFilter extends OncePerRequestFilter {

    private final JwtUtils jwtUtils;

    private static final String ACCESS_TOKEN = "accessToken";

    public TokenValidationFilter(JwtUtils jwtUtils) {
        this.jwtUtils = jwtUtils;
    }

    @Override
    protected void doFilterInternal(HttpServletRequest request, HttpServletResponse response, FilterChain filterChain) throws ServletException, IOException {
        try {
            String accessToken = getAccessTokenByCookie(request.getCookies());
            jwtUtils.validateToken(accessToken);

        } catch (AuthException e) {
            response.setStatus(HttpStatus.UNAUTHORIZED.value());
            response.getOutputStream().write(e.getMessage().getBytes());
            return;
        }

        filterChain.doFilter(request, response);
    }

    private String getAccessTokenByCookie(Cookie[] cookies) {
        if (cookies != null) {
            return Arrays.stream(cookies)
                    .filter(cookie -> cookie.getName().equals(ACCESS_TOKEN))
                    .map(Cookie::getValue)
                    .findFirst()
                    .orElseThrow(() -> new AuthException(HttpStatus.UNAUTHORIZED, "JWT Token is not found"));
        } else {
            throw new AuthException(HttpStatus.UNAUTHORIZED, "Cookie corrupted");
        }
    }

    @Override
    protected boolean shouldNotFilter(HttpServletRequest request) {
        List<String> excludedUrls = List.of("/api/auth/signUp", "/api/auth/signIn", "/api/auth/reissue/token");
        return excludedUrls.stream().anyMatch(request.getRequestURI()::contains);
    }
}
