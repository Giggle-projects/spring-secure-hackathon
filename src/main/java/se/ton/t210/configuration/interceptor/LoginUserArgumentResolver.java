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
import se.ton.t210.domain.Member;
import se.ton.t210.domain.MemberRepository;
import se.ton.t210.domain.TokenSecret;
import se.ton.t210.dto.LoginMemberInfo;
import se.ton.t210.exception.AuthException;

import javax.servlet.http.HttpServletRequest;
import java.util.Base64;

@Component
public class LoginUserArgumentResolver implements HandlerMethodArgumentResolver {

    @Value("${auth.jwt.token.access.cookie.key:accessToken}")
    private String accessTokenCookieKey;

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
        final String accessToken = ((HttpServletRequest) webRequest.getNativeRequest()).getHeader(accessTokenCookieKey);
        if (accessToken == null) {
            throw new AuthException(HttpStatus.UNAUTHORIZED, "Unauthorized access");
        }
        final String email = tokenSecret.getPayloadValue(accessTokenCookieKey, accessToken);
        return LoginMemberInfo.of(memberRepository.findByEmail(email).orElseThrow(() -> new AuthException(HttpStatus.UNAUTHORIZED, "Invalid user")));
    }
}
