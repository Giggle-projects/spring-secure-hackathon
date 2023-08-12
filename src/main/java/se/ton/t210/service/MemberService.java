package se.ton.t210.service;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import se.ton.t210.domain.Member;
import se.ton.t210.domain.MemberRepository;
import se.ton.t210.dto.MemberTokens;
import se.ton.t210.dto.SignUpRequest;
import se.ton.t210.exception.AuthException;
import se.ton.t210.service.token.MemberTokenService;
import se.ton.t210.utils.http.CookieUtils;

import javax.servlet.http.HttpServletResponse;

@Transactional
@Service
public class MemberService {

    @Value("${auth.jwt.token.access.cookie.key:accessToken}")
    private String accessTokenCookieKey;

    @Value("${auth.jwt.token.refresh.cookie:refreshToken}")
    private String refreshTokenCookieKey;

    private final MemberRepository memberRepository;
    private final MemberTokenService memberTokenService;

    public MemberService(MemberRepository memberRepository, MemberTokenService memberTokenService) {
        this.memberRepository = memberRepository;
        this.memberTokenService = memberTokenService;
    }

    public void signUp(SignUpRequest request, String tokenEmail, HttpServletResponse response) {
        if (memberRepository.existsByEmail(request.getEmail())) {
            throw new AuthException(HttpStatus.CONFLICT, "Email is already exists");
        }
        if (!request.getEmail().equals(tokenEmail)) {
            throw new AuthException(HttpStatus.FORBIDDEN, "It is different from the previous email information you entered.");
        }
        final Member member = memberRepository.save(request.toEntity());
        final MemberTokens tokens = memberTokenService.createTokensByEmail(member.getEmail());
        responseTokens(response, tokens);
    }

    private void responseTokens(HttpServletResponse response, MemberTokens tokens) {
        CookieUtils.loadHttpOnlyCookie(response, accessTokenCookieKey, tokens.getAccessToken());
        CookieUtils.loadHttpOnlyCookie(response, refreshTokenCookieKey, tokens.getRefreshToken());
    }
}
