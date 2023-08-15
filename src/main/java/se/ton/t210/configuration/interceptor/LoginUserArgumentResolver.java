package se.ton.t210.configuration.interceptor;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.MethodParameter;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Component;
import org.springframework.web.bind.support.WebDataBinderFactory;
import org.springframework.web.context.request.NativeWebRequest;
import org.springframework.web.method.support.HandlerMethodArgumentResolver;
import org.springframework.web.method.support.ModelAndViewContainer;
import se.ton.t210.configuration.annotation.LoginMember;
import se.ton.t210.domain.MemberRepository;
import se.ton.t210.domain.TokenSecret;
import se.ton.t210.dto.LoginMemberInfo;
import se.ton.t210.exception.AuthException;

import javax.servlet.http.Cookie;
import javax.servlet.http.HttpServletRequest;
import java.util.Arrays;

@Component
public class LoginUserArgumentResolver implements HandlerMethodArgumentResolver {

    @Value("${auth.jwt.token.access.cookie.key:accessToken}")
    private String accessTokenCookieKey;

    @Value("${auth.jwt.token.payload.email.key:email}")
    private String accessTokenPayloadKey;

    private final MemberRepository memberRepository;
    private final TokenSecret tokenSecret;

    public LoginUserArgumentResolver(MemberRepository memberRepository, TokenSecret tokenSecret) {
        this.memberRepository = memberRepository;
        this.tokenSecret = tokenSecret;
    }

    @Override
    public boolean supportsParameter(MethodParameter parameter) {
        return parameter.hasParameterAnnotation(LoginMember.class);
    }

    @Override
    public LoginMemberInfo resolveArgument(MethodParameter parameter, ModelAndViewContainer mavContainer, NativeWebRequest webRequest, WebDataBinderFactory binderFactory) {
        final Cookie[] cookies = ((HttpServletRequest) webRequest.getNativeRequest()).getCookies();
        final Cookie authCookie = Arrays.stream(cookies)
            .filter(it -> it.getName().equals(accessTokenCookieKey))
            .findAny()
            .orElseThrow(() -> new AuthException(HttpStatus.UNAUTHORIZED, "Unauthorized access"));
        final String email = tokenSecret.getPayloadValue(accessTokenPayloadKey, authCookie.getValue());
        return LoginMemberInfo.of(memberRepository.findByEmail(email).orElseThrow(() -> new AuthException(HttpStatus.UNAUTHORIZED, "Invalid user")));
    }
}
