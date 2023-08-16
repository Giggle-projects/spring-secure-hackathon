package se.ton.t210.configuration.filter;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.HttpStatus;
import org.springframework.web.filter.OncePerRequestFilter;
import se.ton.t210.domain.TokenSecret;
import se.ton.t210.dto.MemberTokens;
import se.ton.t210.exception.AuthException;
import se.ton.t210.service.token.TokenService;
import se.ton.t210.utils.http.CookieUtils;

import javax.servlet.FilterChain;
import javax.servlet.ServletException;
import javax.servlet.http.Cookie;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.IOException;
import java.util.Arrays;

import static se.ton.t210.utils.http.CookieUtils.getTokenFromCookies;

public class AccessTokenFilter extends OncePerRequestFilter {

    private static final Logger LOGGER = LoggerFactory.getLogger(AccessTokenFilter.class);

    private final String accessTokenCookieKey;
    private final String refreshTokenCookieKey;
    private final TokenSecret secret;
    private final TokenService tokenService;

    public AccessTokenFilter(TokenSecret secret, String accessTokenCookieKey, String refreshTokenCookieKey, TokenService tokenService) {
        this.secret = secret;
        this.accessTokenCookieKey = accessTokenCookieKey;
        this.refreshTokenCookieKey = refreshTokenCookieKey;
        this.tokenService = tokenService;
    }

    @Override
    protected void doFilterInternal(HttpServletRequest request, HttpServletResponse response, FilterChain filterChain) throws ServletException, IOException {
        String accessToken = null;
        try {
            accessToken = getTokenFromCookies(accessTokenCookieKey, request.getCookies());
            secret.validateToken(accessToken);
        } catch (AuthException e) {
            try {
                if (accessToken == null) {
                    redirectLoginPage(response, e);
                    return;
                }
                refreshTokenReIssue(request, response);
                LOGGER.info("re-login with refresh token");
            } catch (AuthException ae) {
                ae.printStackTrace();
                redirectLoginPage(response, e);
            }
        }
        filterChain.doFilter(request, response);
    }

    private void refreshTokenReIssue(HttpServletRequest request, HttpServletResponse response) {
        final String accessToken = getTokenFromCookies(accessTokenCookieKey, request.getCookies());
        final String refreshToken = getTokenFromCookies(refreshTokenCookieKey, request.getCookies());
        final MemberTokens tokens = tokenService.reissue(accessToken, refreshToken);
        CookieUtils.loadHttpOnlyCookie(response, accessTokenCookieKey, tokens.getAccessToken());
        CookieUtils.loadHttpOnlyCookie(response, refreshTokenCookieKey, tokens.getRefreshToken());
    }

    private void redirectLoginPage(HttpServletResponse response, AuthException e) throws IOException {
        response.setStatus(HttpStatus.UNAUTHORIZED.value());
        response.sendRedirect("/html/sign-in.html");
        response.getOutputStream().write(e.getMessage().getBytes());
    }
}
